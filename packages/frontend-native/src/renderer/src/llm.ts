// Custom error for agent cancellation
export class AgentCancelledError extends Error {
  constructor(message = "Agent execution was cancelled") {
    super(message);
    this.name = "AgentCancelledError";
  }
}

// Track the current running agent invocation
let currentAbortController: AbortController | null = null;

interface Message {
  role: "user" | "assistant" | "tool";
  content: string | Array<any>;
}

interface SystemInfo {
  date: string;
  currentApp: string;
  url?: string;
  title?: string;
}

interface LLMRequest {
  messages: Array<Message>;
  info: SystemInfo;
  image?: string;
}

interface ToolCall {
  toolName: string;
  input: Record<string, unknown>;
}

interface LLMResponse {
  messages: Message[];
  finishReason: string;
}

interface ToolCallPart {
  type: 'tool-call';
  toolCallId: string;
  toolName: string;
  input: any;
}

interface ToolResultPart {
  type: 'tool-result';
  toolCallId: string;
  output: {
    type: 'text';
    value: string;
  };
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:3000";

const generate = async (requestBody: LLMRequest, signal?: AbortSignal): Promise<LLMResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
      signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    // Convert AbortError to AgentCancelledError
    if (error instanceof Error && error.name === 'AbortError') {
      throw new AgentCancelledError();
    }
    throw error;
  }
};

const tools: Record<string, (input: any) => Promise<string>> = {
  open: async (input: { thing: string }) => {
    const result = await window.electronAPI.openTool(input);
    return result.output;
  },
  scroll: async (input: { direction?: "up" | "down" | "left" | "right"; distance?: number }) => {
    const result = await window.electronAPI.scrollTool(input);
    return result.output;
  },
  click: async (input: { x: number; y: number }) => {
    const result = await window.electronAPI.clickTool(input);
    return result.output;
  },
  keys: async (input: { list: string[] }) => {
    const result = await window.electronAPI.keysTool(input);
    return result.output;
  },
};

const handleTool = async (toolPart: ToolCallPart): Promise<string> => {
  const tool = tools[toolPart.toolName];
  if (!tool) {
    throw new Error(`Unknown tool: ${toolPart.toolName}`);
  }
  return await tool(toolPart.input);
};

export const runAgent = async (input: string, info: SystemInfo) => {
  console.log("Running agent with input:", input);

  if (!input.trim()) {
    return;
  }

  // Abort any existing running invocation
  if (currentAbortController) {
    currentAbortController.abort();
  }

  // Create new AbortController for this invocation
  const abortController = new AbortController();
  currentAbortController = abortController;
  const { signal } = abortController;

  try {
    const messages: Message[] = [{ role: "user", content: input }];

    const responseLines: string[] = [];

    // Format browser info into a readable string if we have URL or title
    const formattedInfo: SystemInfo = {
      date: info.date,
      currentApp: info.currentApp
    };

    if (info.url || info.title) {
      // This is a browser with additional info
      const parts: string[] = [];

      if (info.title) {
        parts.push(info.title);
      }

      if (info.url) {
        parts.push(`(${info.url})`);
      }

      parts.push(`in ${info.currentApp}`);
      formattedInfo.currentApp = parts.join(' ');
    }

    while (true) {
      // Check if cancelled before starting new iteration
      if (signal.aborted) {
        throw new AgentCancelledError();
      }
      const requestBody: LLMRequest = { messages, info: formattedInfo };

      const screenshotResult = await window.electronAPI.takeScreenshot();
      if (screenshotResult.success && screenshotResult.image) {
        requestBody.image = `data:image/jpeg;base64,${screenshotResult.image}`;
      } else {
        console.warn('Screenshot failed:', screenshotResult.error);
      }

      console.log("Calling generate with requestBody:", requestBody);
      const response = await generate(requestBody, signal);

      const outputMessages = response.messages;
      console.log(outputMessages);

      let lastPart = null;

      for (const message of outputMessages) {
        responseLines.push(JSON.stringify(message));
        messages.push(message);
        for (const part of message.content) {
          // Check if cancelled before executing tools
          if (signal.aborted) {
            throw new AgentCancelledError();
          }

          if (part.type === 'tool-call') {
            const output = await handleTool(part);
            responseLines.push(output);
            messages.push({
              role: 'tool',
              content: [{
                type: 'tool-result',
                toolName: part.toolName,
                toolCallId: part.toolCallId,
                output: {
                  type: 'text',
                  value: output,
                },
              }]
            })
          } else if (part.type === 'text') {
            console.log(part.text);
          } else {
            console.warn(`Unknown part type: ${part.type}`);
          }
          lastPart = part;
        }
      }

      // timeout for actions to finish happening
      await new Promise(r => setTimeout(r, 1000));

      if (lastPart === null || lastPart.type !== 'tool-call') {
        break;
      }
    }

    if (!responseLines.length) {
      responseLines.push("No response received");
    }

    const responseText = responseLines.join("\n");
    console.log("LLM Response:", responseText);
    return responseText;
  } finally {
    // Clear the controller reference if this is still the current invocation
    if (currentAbortController === abortController) {
      currentAbortController = null;
    }
  }
};
