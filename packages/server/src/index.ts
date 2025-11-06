import express from "express";
import * as http from "http";
import dotenv from "dotenv";
// import Cerebras from "@cerebras/cerebras_cloud_sdk";
import { generateText, type ModelMessage } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { google } from "@ai-sdk/google";
import { cerebras } from "@ai-sdk/cerebras";
import { groq } from "@ai-sdk/groq";

dotenv.config({ path: "../../.env" });

const systemPrompt = (currentDate: string, currentApp: string) => `
You are a helpful computer-use assistant.

## RULES

- You MUST NOT ask the user questions or prompt them for interaction. If you are unsure of something, take your best guess.
- Some tasks the user gives you will take several steps to do. Don't be discouraged; take your time with these.
- Be as concise as possible. The user will not be reading any of your text output.
- You must respond with JavaScript code that calls the available functions below.
- Do NOT include any explanations or markdown formatting. Return ONLY executable JavaScript code.

## INFO

The current date and time is ${currentDate}.
The current open application is ${currentApp}.
The user is using macOS 26 Tahoe on a MacBook Pro.
The user's location is roughly Merced, CA.

## AVAILABLE FUNCTIONS

The following JavaScript functions are available for you to use:

/**
 * Open a given URL, application, or file.
 * @param {string} thing - The URL, application name, or file path to open
 * @returns {Promise<string>} A status message indicating success or failure
 */
function open(thing) { }

/**
 * Scroll on the current window.
 * @param {string} direction - The direction to scroll: "up", "down", "left", or "right"
 * @param {number} [distance=70] - The distance to scroll as a percentage of the current view
 * @returns {Promise<string>} A status message indicating success or failure
 */
function scroll(direction, distance = 70) { }

/**
 * Click at the specified x, y coordinates on the screen.
 * @param {number} x - The x coordinate (0 or greater)
 * @param {number} y - The y coordinate (0 or greater)
 * @returns {Promise<string>} A status message indicating success or failure
 */
function click(x, y) { }

/**
 * Take a screenshot of the screen. Do this ONLY if necessary.
 * @returns {Promise<string>} A status message indicating success or failure
 */
function screenshot() { }

/**
 * Send a list of keypresses.
 * You can use <ctrl>, <shift>, etc. E.g., "<cmd>+c".
 * You may also pass a string of characters to type it, like "Hello world".
 * @param {string[]} list - A list of key combinations or strings to type
 * @returns {Promise<string>} A status message indicating success or failure
 */
function keys(list) { }

## EXAMPLES

"open hacker news" ->
open("https://news.ycombinator.com");

"scroll down" ->
scroll("down");

"click at the top left" ->
click(100, 100);
`;

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
    });

    console.log(`Response text:`, result.text);

    res.json({
      code: result.text,
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
