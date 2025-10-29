interface ScreenshotResult {
  success: boolean;
  image?: string; // base64 encoded image
  width?: number;
  height?: number;
  originalWidth?: number;
  originalHeight?: number;
  error?: string;
}

interface ToolResult {
  success: boolean;
  output: string;
}

interface OpenToolData {
  thing: string;
}

interface ScrollToolData {
  direction?: "up" | "down" | "left" | "right";
  distance?: number;
}

interface ClickToolData {
  x: number;
  y: number;
}

interface KeysToolData {
  list: string[];
}

interface ElectronAPI {
  sendMessage: (msg: string) => void;
  openTool: (data: OpenToolData) => Promise<ToolResult>;
  scrollTool: (data: ScrollToolData) => Promise<ToolResult>;
  clickTool: (data: ClickToolData) => Promise<ToolResult>;
  keysTool: (data: KeysToolData) => Promise<ToolResult>;
  toggleRecording: () => void;
  onToggleRecording: (callback: () => void) => (() => void);
  sendRecordingState: (isRecording: boolean) => void;
  takeScreenshot: () => Promise<ScreenshotResult>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};