// Mongoose reference for knowledge_docs collection.
// Actual data inserted by LangChain via native MongoDB driver.


import mongoose, { Schema } from "mongoose";
import type { Document } from "mongoose";

export interface IKnowledgeDoc extends Document {
  text: string;
  embedding: number[];
  metadata: Record<string, unknown>;
}

const KnowledgeDocSchema: Schema<IKnowledgeDoc> = new Schema({
  text:      { type: String, required: true },
  embedding: { type: [Number], default: [] },
  metadata:  { type: Schema.Types.Mixed, default: {} },
});

export default mongoose.model<IKnowledgeDoc>("KnowledgeDoc", KnowledgeDocSchema, "knowledge_docs");
