// ============================================
// FILE: src/app.ts
// PURPOSE: Configure Express — middleware, routes, error handling
// ============================================

// ============================================
// FILE: src/app.ts
// PURPOSE: Configure Express — middleware, routes, error handling
// ============================================

import express from "express";
import type { Application, Request, Response } from "express";
import cors from "cors";
import chatRoutes from "./routes/chat.routes.ts";
import authRoutes from "./routes/auth.routes.ts";
import vapiRoutes from "./routes/vapi.routes.ts";

import errorHandler from "./middleware/error-handler.middleware.ts";

const app: Application = express();

// Middleware
// Configure CORS: allow client URL and common dev ports (5173, 5174)
const allowedOrigins = [process.env.CLIENT_URL, "http://localhost:5173", "http://localhost:5174"].filter(Boolean);
app.use(
  cors({
    origin: (origin, callback) => {
      // allow non-browser requests like curl/postman (no origin)
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin)) return callback(null, true);
      return callback(new Error("Not allowed by CORS"));
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "PATCH"],
    allowedHeaders: ["Content-Type", "Authorization"],
  }),
);
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/vapi", vapiRoutes);

// 404
app.use((_req: Request, res: Response) => {
  res.status(404).json({ success: false, message: "Route not found." });
});

// Global error handler (must be last)
app.use(errorHandler);

export default app;