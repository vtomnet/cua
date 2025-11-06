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

interface CurrentAppInfo {
  name: string;
  url?: string;
  title?: string;
}

interface JavaScriptExecutionResult {
  success: boolean;
  message?: string;
  error?: string;
}

interface ElectronAPI {
  sendMessage: (msg: string) => void;
  openTool: (data: OpenToolData) => Promise<ToolResult>;
  scrollTool: (data: ScrollToolData) => Promise<ToolResult>;
  clickTool: (data: ClickToolData) => Promise<ToolResult>;
  keysTool: (data: KeysToolData) => Promise<ToolResult>;
  toggleRecording: () => void;
  onToggleRecording: (callback: () => void) => (() => void);
  onCursorUpdate: (callback: (coordinates: { x: number; y: number }) => void) => (() => void);
  sendRecordingState: (isRecording: boolean) => void;
  takeScreenshot: () => Promise<ScreenshotResult>;
  saveAudioFile: (audioData: Float32Array, filename: string) => Promise<{ success: boolean; path?: string; error?: string }>;
  submitText: (text: string) => Promise<{ success: boolean }>;
  resizeControlWindow: (showSettings: boolean) => Promise<{ success: boolean }>;
  onProcessText: (callback: (text: string) => void) => (() => void);
  onRecordingStateUpdate: (callback: (isRecording: boolean) => void) => (() => void);
  getSetting: (key: string) => Promise<any>;
  setSetting: (key: string, value: any) => Promise<void>;
  getInitialRecordingState: () => Promise<boolean>;
  getCurrentApp: () => Promise<CurrentAppInfo>;
  executeJavaScript: (code: string) => Promise<JavaScriptExecutionResult>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};
