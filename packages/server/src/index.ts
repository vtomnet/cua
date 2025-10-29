import express from "express";
import * as http from "http";
import dotenv from "dotenv";
// import Cerebras from "@cerebras/cerebras_cloud_sdk";
import { generateText, tool } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";

dotenv.config({ path: "../../.env" });

// const client = new Cerebras();

const tools = {
  open: tool({
    description: "Open a given URL, application, or file.",
    inputSchema: z.object({
      thing: z.string()
    }),
  }),
  scroll: tool({
    description: "Scroll on the current window. The distance is a percentage of the current view, default 70.",
    inputSchema: z.object({
      direction: z.enum(["up", "down", "left", "right"]),
      distance: z.number().optional()
    }),
  }),
  click: tool({
    description: "Click at the specified x, y coordinates on the screen.",
    inputSchema: z.object({
      x: z.number().int().min(0),
      y: z.number().int().min(0)
    }),
  }),
  keys: tool({
    description: 'Send a list of keypresses. You can use <ctrl>, <shift>, etc. E.g., "<cmd>+c". You may also pass a string of characters to type it, like "Hello world".',
    inputSchema: z.object({
      list: z.array(z.string()),
    }),
  }),
}

const app = express();
const server = http.createServer(app);

const allowedOrigins = new Set(
  process.env.ALLOWED_ORIGINS
    ? process.env.ALLOWED_ORIGINS.split(',').map(origin => origin.trim())
    : ["http://localhost:5173", "views://mainview"]
);

app.use((req, res, next) => {
  const origin = req.headers.origin;

  if (origin && allowedOrigins.has(origin)) {
    res.header("Access-Control-Allow-Origin", origin);
  }

  res.header("Vary", "Origin");
  res.header("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.header("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.sendStatus(204);
  }

  next();
});

app.use(express.json());
app.use(express.static("public"));

app.post("/generate", async (req, res) => {
  try {
    const { text, image } = req.body ?? {};
    console.log(req.body);

    if (typeof text !== "string") {
      return res.status(400).json({ error: "text must be a string" });
    }

    // Build messages array with text and optional image
    const userContent: Array<{ type: "text"; text: string } | { type: "image"; image: string }> = [];

    // Add text content
    userContent.push({
      type: "text",
      text: text,
    });

    // Add image content if provided
    if (image) {
      if (typeof image !== "string") {
        return res.status(400).json({ error: "image must be a base64 string" });
      }

      // Determine image type from base64 string or assume PNG
      let mimeType = "image/png";
      if (image.startsWith("data:")) {
        const matches = image.match(/^data:([^;]+);base64,/);
        if (matches) {
          mimeType = matches[1];
        }
      }

      userContent.push({
        type: "image",
        image: image.startsWith("data:") ? image : `data:${mimeType};base64,${image}`,
      });
    }

    const messages = [{
      role: "user" as const,
      content: userContent,
    }];

    const result = await generateText({
      model: anthropic("claude-sonnet-4-5"),
      messages: messages,
      tools: tools,
    });

    res.json({
      text: result.text,
      toolCalls: result.toolCalls,
      usage: result.usage,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to generate completion";
    console.error("Error generating completion", error);
    res.status(500).json({ error: message });
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
