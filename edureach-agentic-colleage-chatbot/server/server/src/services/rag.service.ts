import path from "node:path";
import { fileURLToPath } from "node:url";
import { readdirSync, readFileSync } from "node:fs";
import { MongoClient } from "mongodb";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { SystemMessage, HumanMessage, AIMessage, ToolMessage } from "@langchain/core/messages";
import { initiateOutboundCall } from "./vapi.service.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let mongoClient: MongoClient | null = null;

const timeoutPromise = <T>(promise: Promise<T>, ms: number, timeoutError: Error): Promise<T> => {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(timeoutError);
    }, ms);

    promise
      .then((res) => {
        clearTimeout(timer);
        resolve(res);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
};

const getMongoClient = async (): Promise<MongoClient> => {
  if (!mongoClient) {
    mongoClient = new MongoClient(process.env.MONGODB_URI || "");
    await mongoClient.connect();
  }
  return mongoClient;
};


// gemini-embedding-001 → default 3072 dimensions 
const getEmbeddings = () => {
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY is not set in .env!");
  }
  return new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_API_KEY,
    model: "gemini-embedding-001",
  });
};

// ---- Vector Store ----
const getVectorStore = async () => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs");

  return new MongoDBAtlasVectorSearch(getEmbeddings(), {
    collection: collection as any,
    indexName: "edureach_vector_index",
    textKey: "pageContent",
    embeddingKey: "embedding",
  });
};

// Helper to generate embeddings using Google REST API directly with automatic retries and rate limit handling
const embedTextsDirectly = async (texts: string[]): Promise<number[][]> => {
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error("GOOGLE_API_KEY is not set in .env!");
  }

  // Use gemini-embedding-001 or gemini-embedding-2. gemini-embedding-001 is standard 3072D.
  const model = "gemini-embedding-001";
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:batchEmbedContents?key=${apiKey}`;
  const BATCH_SIZE = 50; // LangChain's default of sending all 149 at once fails because Google limits to max 100 per batch.
  const allVectors: number[][] = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);
    const requests = batch.map((text) => ({
      model: `models/${model}`,
      content: { parts: [{ text }] }
    }));

    let success = false;
    let attempt = 1;
    const maxAttempts = 5;
    let batchVectors: number[][] = [];

    while (attempt <= maxAttempts && !success) {
      try {
        const batchNum = Math.floor(i / BATCH_SIZE) + 1;
        const totalBatches = Math.ceil(texts.length / BATCH_SIZE);
        console.log(`      Embedding batch ${batchNum}/${totalBatches} (chunks ${i + 1}-${Math.min(i + BATCH_SIZE, texts.length)}, attempt ${attempt}/${maxAttempts})...`);

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ requests })
        });

        if (response.status === 429) {
          console.warn(`      Rate limited (429). Waiting ${attempt * 5}s before retry...`);
          await new Promise((resolve) => setTimeout(resolve, attempt * 5000));
          attempt++;
          continue;
        }

        if (response.status !== 200) {
          const errText = await response.text();
          throw new Error(`API returned status ${response.status}: ${errText}`);
        }

        const data = await response.json();
        if (data.embeddings && Array.isArray(data.embeddings)) {
          batchVectors = data.embeddings.map((e: any) => e.values);
          const hasEmpty = batchVectors.some((v) => !v || v.length === 0);
          if (hasEmpty) {
            throw new Error("Received empty vectors from API");
          }
          success = true;
        } else {
          throw new Error("Unexpected API response format");
        }
      } catch (err: any) {
        console.error(`      Attempt ${attempt} failed: ${err.message || err}`);
        if (attempt === maxAttempts) {
          throw err;
        }
        await new Promise((resolve) => setTimeout(resolve, attempt * 3000));
        attempt++;
      }
    }

    allVectors.push(...batchVectors);

    // Generous delay between successful batches to be super gentle on the free tier rate limit
    if (i + BATCH_SIZE < texts.length) {
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }

  return allVectors;
};

// A) INDEXING — runs ONCE at server startup

export const initializeKnowledgeBase = async (): Promise<void> => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs") as any;

  // Count how many .txt files are in the knowledge-base directory
  const kbDir = path.join(__dirname, "../../knowledge-base");
  const txtFiles = readdirSync(kbDir).filter((file) => file.endsWith(".txt"));

  // Get metadata about previous indexing
  const indexMetadata = await collection.findOne({ _id: "INDEX_METADATA" } as any);
  const previousFileCount = indexMetadata?.fileCount || 0;

  // Check if we have valid non-empty embeddings in MongoDB
  const docWithEmbedding = await collection.findOne({
    embedding: { $exists: true, $type: "array", $ne: [] }
  } as any);
  const hasValidEmbeddings = docWithEmbedding !== null;

  const existingCount = await collection.countDocuments({ _id: { $ne: "INDEX_METADATA" } } as any);

  // Skip indexing if data is already populated, matching the file counts and has embeddings
  if (existingCount > 0 && indexMetadata && previousFileCount === txtFiles.length && hasValidEmbeddings) {
    console.log(`✓ Knowledge base ready (${existingCount} chunks with embeddings from ${txtFiles.length} files)`);
    return;
  }

  console.log(" Indexing knowledge base starting...");

  // Clear everything to build a clean index
  await collection.deleteMany({} as any);

  // Verify API key exists
  console.log(" API configured - proceeding with indexing...");

  if (txtFiles.length === 0) {
    throw new Error("No .txt files found in knowledge-base directory");
  }

  console.log(`   Found ${txtFiles.length} knowledge base files: ${txtFiles.join(", ")}`);

  let docs: Document<Record<string, any>>[] = [];
  let totalCharacters = 0;

  // Load each .txt file
  for (const file of txtFiles) {
    const filePath = path.join(kbDir, file);
    const content = readFileSync(filePath, "utf-8");
    const doc = new Document({
      pageContent: content,
      metadata: { source: file },
    });
    docs.push(doc);
    totalCharacters += content.length;
    console.log(`    Loaded ${file} (${content.length} characters)`);
  }

  console.log(`    Total loaded: ${totalCharacters} characters from ${docs.length} files`);

  // SPLIT
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);
  console.log(`    Split into ${allSplits.length} chunks`);

  // EMBED + STORE — GENERATE EMBEDDINGS VIA ROBUST DIRECT API AND INSERT
  console.log(`    Generating embeddings for ${allSplits.length} chunks...`);
  try {
    const texts = allSplits.map((chunk) => chunk.pageContent);
    const allVectors = await embedTextsDirectly(texts);

    console.log(`    ✓ All ${allVectors.length} embeddings generated (${allVectors[0]?.length}D vectors)`);

    const docsToInsert = allSplits.map((chunk, index) => ({
      pageContent: chunk.pageContent,
      embedding: allVectors[index],
      source: chunk.metadata.source,
      loc: chunk.metadata.loc || {},
    }));

    const result = await collection.insertMany(docsToInsert);
    console.log(`    ✓ Inserted ${result.insertedCount} document chunks with valid ${allVectors[0]?.length}D embeddings`);

  } catch (error: any) {
    console.error(`\n     EMBEDDING ERROR: ${error.message || String(error)}`);
    console.error(`    The knowledge base was NOT indexed. Queries will use keyword fallback only.`);
    // Do not re-throw — let server start anyway; queries will use keyword fallback
    return;
  }

  // VERIFY — confirm vectors are actually stored
  const totalDocs = await collection.countDocuments({ _id: { $ne: "INDEX_METADATA" } } as any);
  const sampleDoc = await collection.findOne({ embedding: { $exists: true, $ne: [] } } as any);
  const embeddingDim = sampleDoc?.embedding?.length || 0;
  console.log(`    ✓ Verified: ${totalDocs} docs in MongoDB, embedding dimension: ${embeddingDim}D`);

  // Store metadata about this indexing session
  await collection.updateOne(
    { _id: "INDEX_METADATA" } as any,
    {
      $set: {
        fileCount: txtFiles.length,
        chunksCount: totalDocs,
        lastIndexed: new Date(),
        files: txtFiles,
        embeddingDim,
      },
    },
    { upsert: true }
  );
  console.log(`    ✓ Indexing complete with valid embeddings`);
};


// B) RAG RETRIEVAL ENGINE (used inside the Retrieve Tool)

const retrieveContext = async (question: string): Promise<string> => {
  const client = await getMongoClient();
  const collection = client.db("edureach_db").collection("knowledge_docs") as any;

  let retrievedDocs: { pageContent: string; metadata: { source: string } }[] = [];
  let usedFallback = false;

  // 1. ATTEMPT VECTOR SIMILARITY SEARCH FIRST (with strict timeout)
  try {
    const vectorStore = await getVectorStore();
    console.log(`      [RAG Agent] Querying vector similarity search for: "${question.substring(0, 50)}"...`);
    const searchPromise = vectorStore.similaritySearch(question, 4);
    const results = await timeoutPromise(
      searchPromise,
      3000,
      new Error("Vector search timed out (likely missing Atlas search index)")
    );

    if (results && results.length > 0) {
      retrievedDocs = results.map(doc => ({
        pageContent: doc.pageContent,
        metadata: { source: String(doc.metadata?.source || "unknown") }
      }));
      console.log(`      [RAG Agent] Vector search retrieved ${retrievedDocs.length} chunks`);
    } else {
      usedFallback = true;
    }
  } catch (vectorError: any) {
    console.warn(`      [RAG Agent] Vector search failed. Falling back... Reason: ${vectorError.message || vectorError}`);
    usedFallback = true;
  }

  // 2. FALLBACK HYBRID SEARCH
  if (usedFallback || retrievedDocs.length === 0) {
    console.log(`      [RAG Agent] Running robust keyword fallback search...`);
    const keywords = question.toLowerCase();

    const fileKeywords: Record<string, string[]> = {
      "01-about-us.txt": ["about us", "who are you", "what is edureach", "history", "established", "vision", "mission"],
      "02-departments.txt": ["department", "engineering", "management", "sciences"],
      "03-courses-offered.txt": ["course", "program", "degree", "offer", "btech", "mtech", "mba", "phd", "engineering", "management", "specialize", "what do you offer"],
      "04-admissions.txt": ["admission", "apply", "eligible", "entrance", "exam", "qualify", "requirement", "criteria"],
      "05-fee-structure.txt": ["fee", "cost", "price", "tuition", "payment", "expense", "charge", "structure"],
      "06-scholarships.txt": ["scholarship", "financial", "award", "funding", "grant", "merit", "assistance"],
      "07-placements.txt": ["placement", "job", "career", "company", "salary", "recruit", "placement rate", "companies visit"],
      "08-campus-life.txt": ["campus", "hostel", "club", "activity", "student life", "dorm", "accommodation", "sports", "facilities"],
      "09-faculty-mentors.txt": ["faculty", "teacher", "mentor", "professor", "instructor", "staff", "faculty ratio"],
      "10-rules-regulations.txt": ["rule", "regulation", "policy", "academic", "conduct", "disciplinary", "code", "attendance"],
      "11-research-innovation.txt": ["research", "innovation", "lab", "project", "publication", "patent"],
      "12-contact-information.txt": ["contact", "phone", "email", "address", "call", "reach", "location", "address", "website"]
    };

    const matchedFiles: Set<string> = new Set();
    for (const [file, kws] of Object.entries(fileKeywords)) {
      if (kws.some((kw) => keywords.includes(kw))) {
        matchedFiles.add(file);
      }
    }

    let dbMatchedDocs: any[] = [];
    if (matchedFiles.size > 0) {
      dbMatchedDocs = await collection
        .find({ source: { $in: Array.from(matchedFiles) } })
        .limit(8)
        .toArray();
    }

    const terms = question
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter((word) => word.length > 4);

    let regexMatchedDocs: any[] = [];
    if (terms.length > 0) {
      const orQueries = terms.map((term) => ({
        pageContent: { $regex: term, $options: "i" }
      }));
      regexMatchedDocs = await collection
        .find({ $or: orQueries })
        .limit(6)
        .toArray();
    }

    const combinedDocs = [...dbMatchedDocs];
    for (const doc of regexMatchedDocs) {
      if (!combinedDocs.some(d => String(d._id) === String(doc._id))) {
        combinedDocs.push(doc);
      }
    }

    if (combinedDocs.length === 0) {
      const defaultDocs = await collection
        .find({ source: { $in: ["01-about-us.txt", "03-courses-offered.txt"] } })
        .limit(4)
        .toArray();
      combinedDocs.push(...defaultDocs);
    }

    const topDocs = combinedDocs.slice(0, 4);
    retrievedDocs = topDocs.map(doc => ({
      pageContent: doc.pageContent,
      metadata: { source: String(doc.source || "unknown") }
    }));
    console.log(`      [RAG Agent] Keyword/regex search fetched ${retrievedDocs.length} chunks`);
  }

  if (retrievedDocs.length === 0) {
    return "No matching information found in the knowledge base.";
  }

  return retrievedDocs
    .map((doc) => `Source File: ${doc.metadata.source}\nContent: ${doc.pageContent}`)
    .join("\n\n");
};


// C) AGENT TOOLS

const retrieveTool = tool(
  async ({ query }: { query: string }) => {
    return await retrieveContext(query);
  },
  {
    name: "retrieve",
    description: "Retrieve official information from the EduReach College knowledge base. Use this tool ALWAYS first for any questions about courses, fees, admissions, rules, hostel, mentors, campus, placements.",
    schema: z.object({ query: z.string() })
  }
);

const requestCallTool = tool(
  async ({ phoneNumber, userName, userEmail, preferredCourse, queryTopic }: {
    phoneNumber: string;
    userName: string;
    userEmail: string;
    preferredCourse?: string;
    queryTopic?: string;
  }) => {
    try {
      console.log(`      [Call Tool] Triggering outbound Vapi call to ${phoneNumber} (${userName})...`);
      const result = await initiateOutboundCall({
        phoneNumber,
        userName,
        userEmail,
        preferredCourse,
        queryTopic
      });
      return `Successfully triggered and initiated outbound call via Ava (AI voice counselor). Call ID: ${result.id}. Current Status: ${result.status}`;
    } catch (err: any) {
      console.error(`      [Call Tool] Vapi error:`, err.message || err);
      return `Failed to trigger outbound call: ${err.message || err}`;
    }
  },
  {
    name: "request_call",
    description: "Trigger an automated outbound voice call to the student. If the student explicitly requests a call, wants someone to contact/reach them, or provides their phone number for counseling, collect their name, phone number, email, and preferred course/topic, and trigger this tool.",
    schema: z.object({
      phoneNumber: z.string().describe("The student's phone number"),
      userName: z.string().describe("The student's full name"),
      userEmail: z.string().describe("The student's email address"),
      preferredCourse: z.string().optional().describe("The course the student is interested in"),
      queryTopic: z.string().optional().describe("The topic of query, e.g., fee structure, admission requirements")
    })
  }
);


// D) AGENT EXECUTION LOOP

export const getRAGResponse = async (question: string): Promise<string> => {
  try {
    console.log(`\n[Agentic RAG] Processing query: "${question.substring(0, 50)}..."`);

    // 1. INITIALIZE MODEL & BIND TOOLS
    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "gemini-2.5-flash",
      temperature: 0.0, // Low temperature for deterministic RAG behavior
    });

    const tools = [retrieveTool, requestCallTool];
    const modelWithTools = model.bindTools(tools);

    // 2. CONSTRUCT AGENT CONVERSATION HISTORY WITH SYSTEM PROMPT
    const systemPrompt =
      "You are EduReach Bot, the official AI counselor for EduReach College, Hyderabad.\n" +
      "Your objective is to provide highly precise, friendly, professional, and completely grounded answers to student queries.\n\n" +
      "You have access to the following tools:\n" +
      "1. `retrieve`: Search the college knowledge base. ALWAYS use this tool first if the query is about courses, departments, admissions, rules, hostels, placements, scholarships, mentors, fees, or contact information.\n" +
      "2. `request_call`: Trigger an automated outbound voice call to the student. If the student explicitly requests a call, wants to talk to a person/counselor, or provides their phone number, collect their name, phone number, email, and preferred course, and trigger this tool to immediately call them.\n\n" +
      "CRITICAL RULES:\n" +
      "1. If you call `retrieve` and the retrieved information is insufficient, ambiguous, or empty, you MUST respond to the student with exactly: 'I don't have that information right now. Click Talk to Us to speak with a counselor.'\n" +
      "2. DO NOT extrapolate, speculate, or make up facts. No external knowledge should be used to guess details (e.g. specific names, numbers, rules, fees, dates) that are not explicitly stated in the context retrieved by your tools.\n" +
      "3. Keep your answer professional, clear, and well-structured using markdown bullet points if helpful.";

    const agentMessages: any[] = [
      new SystemMessage(systemPrompt),
      new HumanMessage(question)
    ];

    // 3. EXECUTE THE REACT REASONING LOOP (Max 5 iterations to prevent infinite runaways)
    const MAX_ITERATIONS = 5;
    let iteration = 0;
    let finalAnswer = "";

    while (iteration < MAX_ITERATIONS) {
      console.log(`[Agentic RAG] Agent thinking (iteration ${iteration + 1}/${MAX_ITERATIONS})...`);
      const response = await modelWithTools.invoke(agentMessages);
      agentMessages.push(response);

      // If the agent doesn't want to call tools, it has reached its final answer!
      if (!response.tool_calls || response.tool_calls.length === 0) {
        finalAnswer = String(response.content).trim();
        break;
      }

      // Execute tool calls in parallel/series
      for (const toolCall of response.tool_calls) {
        console.log(`[Agentic RAG] Model called tool: "${toolCall.name}"`);
        let toolResult = "";

        if (toolCall.name === "retrieve") {
          try {
            const { query } = toolCall.args as any;
            toolResult = await retrieveContext(query);
          } catch (err: any) {
            toolResult = `Error retrieving info: ${err.message || err}`;
          }
        } else if (toolCall.name === "request_call") {
          try {
            const args = toolCall.args as any;
            const callRes = await initiateOutboundCall({
              phoneNumber: args.phoneNumber,
              userName: args.userName,
              userEmail: args.userEmail,
              preferredCourse: args.preferredCourse,
              queryTopic: args.queryTopic
            });
            toolResult = `Successfully triggered and initiated outbound call via Vapi. Call ID: ${callRes.id}. Current Status: ${callRes.status}`;
          } catch (err: any) {
            toolResult = `Failed to trigger call: ${err.message || err}`;
          }
        } else {
          toolResult = `Tool "${toolCall.name}" is not supported.`;
        }

        // Push tool output message back to the conversation history
        agentMessages.push(new ToolMessage({
          content: toolResult,
          tool_call_id: toolCall.id!,
          name: toolCall.name
        }));
      }

      iteration++;
    }

    if (!finalAnswer) {
      finalAnswer = "I couldn't process your request in time. Please try again or click 'Talk to Us'.";
    }

    console.log(`[Agentic RAG] Response generated successfully`);
    return finalAnswer;

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error(`[Agentic RAG] Error during getRAGResponse: ${errorMsg}`);
    return "I'm having trouble right now. Please try again or click 'Talk to Us'.";
  }
};