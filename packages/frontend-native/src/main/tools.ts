import open from "open-things";
import robot from "@jitsi/robotjs";

export interface ToolResult {
  success: boolean;
  output: string;
}

export async function openTool(data: any): Promise<ToolResult> {
  const result = await open(data.thing);
  return { success: result.success, output: result.output };
}

export async function scrollTool(data: any): Promise<ToolResult> {
  return { success: true, output: "" };
}
