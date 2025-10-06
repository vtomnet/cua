import express from "express";
import * as http from "http";
import Cerebras from "@cerebras/cerebras_cloud_sdk";

const client = new Cerebras();

const tools = [
  {
    type: "function",
    function: {
      name: "open",
      description: "Open a given URL, application, or file.",
      parameters: {
        type: "object",
        properties: {
          thing: {
            type: "string",
          },
        },
        required: ["thing"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "scroll",
      description:
        "Scroll on the current window. The distance is a percentage of the current view, default 70.",
      parameters: {
        type: "object",
        properties: {
          direction: {
            type: "string",
            enum: ["up", "down", "left", "right"],
          },
          distance: {
            type: "number",
          },
        },
        required: ["direction"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "keys",
      parameters: {
        description:
          'Send a list of keypresses. You can use <ctrl>, <shift>, etc. E.g., "<cmd>+c". You may also pass a string of characters to type it, like "Hello world".',
        type: "object",
        properties: {
          list: {
            type: "array",
            items: {
              type: "string",
            },
          },
        },
      },
    }
  },
];

const app = express();
const server = http.createServer(app);

const allowedOrigins = new Set(["http://localhost:5173", "views://mainview"]);

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
    const { input } = req.body ?? {};

    if (typeof input !== "string") {
      return res.status(400).json({ error: "input must be a string" });
    }

    const completion = await client.chat.completions.create({
      messages: [{ role: "user", content: input }],
      model: "gpt-oss-120b",
      tools,
      reasoning_effort: "low",
    });

    res.json(completion);
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
