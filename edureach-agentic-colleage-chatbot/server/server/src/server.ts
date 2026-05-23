import app from "./app.ts";
import connectDB from "./config/database.config.ts";
import { initializeKnowledgeBase } from "./services/rag.service.ts";

const PORT = process.env.PORT || 5000;

const start = async (): Promise<void> => {
  try {
    // 1. Connect Mongoose (for users collection)
    await connectDB();

    // 2. Start Express FIRST (non-blocking)
    app.listen(PORT, () => {
      console.log(` EduReach Server is running!`);
      console.log(` URL: http://localhost:${PORT}`);
      console.log(` Node: ${process.version}`);
      console.log(` Press Ctrl+C to stop`);
    });

    // 3. Index knowledge base in background (doesn't block server startup)
    //    First run: loads .txt → splits → embeds → stores in MongoDB
    //    Subsequent runs: sees data exists, skips
    initializeKnowledgeBase().catch((error) => {
      console.error("Knowledge base initialization failed:", error);
    });
  } catch (error) {
    console.error("Failed to start server:", error);
    process.exit(1);
  }
};

start();














