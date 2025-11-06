interface ScreenshotResult {
  image: string; // base64 encoded image
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
}

interface CurrentAppInfo {
  name: string;
  url?: string;
  title?: string;
}

interface ElectronAPI {
  sendMessage: (msg: string) => void;
  toggleRecording: () => void;
  onToggleRecording: (callback: () => void) => (() => void);
  onCursorUpdate: (callback: (coordinates: { x: number; y: number }) => void) => (() => void);
  sendRecordingState: (isRecording: boolean) => void;
  takeScreenshot: () => Promise<ScreenshotResult>;
  submitText: (text: string) => Promise<void>;
  resizeControlWindow: (showSettings: boolean) => Promise<void>;
  onProcessText: (callback: (text: string) => void) => (() => void);
  onRecordingStateUpdate: (callback: (isRecording: boolean) => void) => (() => void);
  getSetting: (key: string) => Promise<any>;
  setSetting: (key: string, value: any) => Promise<void>;
  getCurrentApp: () => Promise<CurrentAppInfo>;
  executeJavaScript: (code: string) => Promise<void>;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};
