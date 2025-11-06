import express from "express";
import * as http from "http";
import dotenv from "dotenv";
// import Cerebras from "@cerebras/cerebras_cloud_sdk";
import { generateText, tool, type ModelMessage } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { cerebras } from "@ai-sdk/cerebras";
import { groq } from "@ai-sdk/groq";
import { z } from "zod";

dotenv.config({ path: "../../.env" });

const systemPrompt = (currentDate: string, currentApp: string) => `
You are a helpful computer-use assistant.

## RULES

- You MUST NOT ask the user questions or prompt them for interaction. If you are unsure of something, take your best guess.
- Some tasks the user gives you will take several steps to do. Don't be discouraged; take your time with these.
- Be as concise as possible. The user will not be reading any of your text output.

## INFO

The current date and time is ${currentDate}.
The current open application is ${currentApp}.
The user is using macOS 26 Tahoe on a MacBook Pro.
The user's location is roughly Merced, CA.

## EXAMPLES

"open hacker news" -> call \`open\` tool with argument https://news.ycombinator.com
`;

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
  screenshot: tool({
    description: "Take a screenshot of the screen. Do this ONLY if necessary.",
    inputSchema: z.object({}),
  }),
  // keys: tool({
  //   description: 'Send a list of keypresses. You can use <ctrl>, <shift>, etc. E.g., "<cmd>+c". You may also pass a string of characters to type it, like "Hello world".',
  //   inputSchema: z.object({
  //     list: z.array(z.string()),
  //   }),
  // }),
};

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

app.use(express.json({ limit: '10mb' }));
app.use(express.static("public"));

app.post("/generate", async (req, res) => {
  try {
    const { info, messages: inputMessages, image } = req.body ?? {};

    // Log request size for diagnostics
    const requestSize = JSON.stringify(req.body).length;
    const imageSizeKB = image ? (image.length * 0.75 / 1024).toFixed(2) : 0;
    console.log(`Request received - Total size: ${(requestSize / 1024).toFixed(2)}KB, Image size: ${imageSizeKB}KB`);

    const messages: ModelMessage[] = [];
    if (false) // DEBUG
    if (image) messages.push({
      role: "user",
      content: [{
        type: "image",
        image: image,
      }],
    });
    messages.push(...inputMessages);

    console.log(`Sending messages:`, messages);
    console.log(`System prompt:`, systemPrompt(info.currentDate, info.currentApp));

    const result = await generateText({
      // model: anthropic("claude-sonnet-4-5"),
      // model: groq("meta-llama/llama-4-maverick-17b-128e-instruct"),
      model: cerebras("gpt-oss-120b"),
      system: systemPrompt(info.currentDate, info.currentApp),
      messages: messages,
      tools: tools,
    });

    res.json({
      messages: result.response.messages,
      usage: result.usage,
      finishReason: result.finishReason,
    });
  } catch (error) {
    const message = (error as Error).message || "Failed to generate completion";
    console.error("Error generating completion:", error);

    res.status(500).json({ error: message });
  }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
