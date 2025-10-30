interface LLMRequest {
  text: string;
  image?: string;
}

interface ToolCall {
  toolName: string;
  input: Record<string, unknown>;
}

interface LLMResponse {
  text?: string;
  toolCalls?: ToolCall[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:3000";

export const sendToLLM = async (transcription: string) => {
  if (!transcription.trim()) {
    return;
  }

  try {
    // Take screenshot before sending to API
    let screenshotDataUri: string | undefined;
    try {
      const screenshotResult = await window.electronAPI.takeScreenshot();
      if (screenshotResult.success && screenshotResult.image) {
        // Convert base64 to proper JPEG data URI
        screenshotDataUri = `data:image/jpeg;base64,${screenshotResult.image}`;

        // Calculate approximate size in KB
        const sizeKB = (screenshotResult.image.length * 0.75 / 1024).toFixed(2);
        console.log(`Screenshot captured: ${screenshotResult.width}x${screenshotResult.height} (resized from ${screenshotResult.originalWidth}x${screenshotResult.originalHeight}), ~${sizeKB} KB`);
      } else {
        console.warn('Screenshot failed:', screenshotResult.error);
      }
    } catch (screenshotError) {
      console.error('Error taking screenshot:', screenshotError);
      // Continue without screenshot if it fails
    }

    // Build request body with text and optional image
    const requestBody: LLMRequest = {
      text: transcription,
    };

    if (screenshotDataUri) {
      requestBody.image = screenshotDataUri;
    }

    const response = await fetch(`${API_BASE_URL}/generate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    const data: LLMResponse = await response.json();
    const responseLines: string[] = [];

    // Handle the server response format
    const text = data?.text;
    if (typeof text === "string" && text.trim()) {
      responseLines.push(text.trim());
    }

    // Handle tool calls from the API format
    const toolCalls = Array.isArray(data?.toolCalls) ? data.toolCalls : [];
    for (const call of toolCalls) {
      console.log(call);
      const toolName = call?.toolName ?? "unknown_tool";
      const args = call?.input ?? {};
      const argsStr = JSON.stringify(args);
      responseLines.push(`Called ${toolName}(${argsStr})`);

      // Execute tool calls with proper error handling
      try {
        let result;
        if (toolName === "open") {
          result = await window.electronAPI.openTool(args as { thing: string });
        } else if (toolName === "scroll") {
          result = await window.electronAPI.scrollTool(args as { direction?: "up" | "down" | "left" | "right"; distance?: number });
        } else if (toolName === "click") {
          result = await window.electronAPI.clickTool(args as { x: number; y: number });
        } else if (toolName === "keys") {
          result = await window.electronAPI.keysTool(args as { list: string[] });
        } else {
          console.warn(`Unknown tool: ${toolName}`);
          continue;
        }

        console.log(`Result from ${toolName}:`, result);

        // Add result info to response if tool execution failed
        if (result && !result.success) {
          responseLines.push(`${toolName} failed: ${result.output || 'Unknown error'}`);
        }
      } catch (toolError) {
        console.error(`Error executing tool ${toolName}:`, toolError);
        responseLines.push(`${toolName} error: ${toolError instanceof Error ? toolError.message : 'Unknown error'}`);
      }
    }

    // Fallback if no content
    if (!responseLines.length) {
      if (data) {
        responseLines.push(JSON.stringify(data));
      } else {
        responseLines.push("No response received");
      }
    }

    const responseText = responseLines.join("\n");
    console.log("LLM Response:", responseText);
  } catch (error) {
    console.error("Error getting LLM response:", error);
  }
};
