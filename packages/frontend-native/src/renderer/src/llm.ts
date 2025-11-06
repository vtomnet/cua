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

interface LLMResponse {
  code: string;
  finishReason: string;
  usage?: any;
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

// Execute JavaScript code in the sandbox
const executeJavaScriptCode = async (code: string): Promise<{ success: boolean; message?: string }> => {
  try {
    const result = await window.electronAPI.executeJavaScript(code);

    if (result.success) {
      return { success: true, message: result.message || 'Code executed successfully' };
    } else {
      const errorMsg = result.error || 'Unknown error';
      console.error('JavaScript execution error:', errorMsg);
      return { success: false, message: `Error: ${errorMsg}` };
    }
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('Failed to execute JavaScript code:', errorMsg);
    return { success: false, message: `Failed to execute: ${errorMsg}` };
  }
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

    // TODO: the finishReason logic is outdated. replace.
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

      console.log("Received JavaScript code:", response.code);
      console.log("Finish reason:", response.finishReason);
      responseLines.push(`Code:\n${response.code}`);

      // Check if cancelled before executing code
      if (signal.aborted) {
        throw new AgentCancelledError();
      }

      // Execute the JavaScript code
      const result = await executeJavaScriptCode(response.code);

      if (result.success) {
        const executionMessage = result.message || 'Code executed successfully';
        responseLines.push(`Execution: ${executionMessage}`);
        console.log("Execution result:", executionMessage);

        // Add the assistant's code and execution results to messages
        messages.push({
          role: 'assistant',
          content: response.code
        });

        // Only continue the loop if the model wants to continue (didn't finish naturally)
        // and the execution was successful
        if (response.finishReason === 'stop') {
          // Model indicated it's done
          responseLines.push("Task complete");
          break;
        }

        messages.push({
          role: 'user',
          content: `Execution: ${executionMessage}`
        });
      } else {
        // Execution failed
        const errorMessage = result.message || 'Unknown error';
        responseLines.push(`Execution failed: ${errorMessage}`);
        console.error("Execution failed:", errorMessage);

        messages.push({
          role: 'assistant',
          content: response.code
        });

        messages.push({
          role: 'user',
          content: `Execution failed: ${errorMessage}`
        });
      }

      // timeout for actions to finish happening
      await new Promise(r => setTimeout(r, 1000));

      // Continue loop to allow for multi-step tasks
      // Break if we've had too many iterations (safety limit)
      if (messages.length > 20) {
        responseLines.push("Reached maximum iteration limit");
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
