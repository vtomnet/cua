import { contextBridge, ipcRenderer } from 'electron';

interface ScreenshotResult {
  success: boolean;
  image?: string;
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

contextBridge.exposeInMainWorld('electronAPI', {
  // Example IPC
  sendMessage: (msg: string) => ipcRenderer.send('message', msg),
  openTool: (data: OpenToolData): Promise<ToolResult> => ipcRenderer.invoke('open-tool', data),
  scrollTool: (data: ScrollToolData): Promise<ToolResult> => ipcRenderer.invoke('scroll-tool', data),
  clickTool: (data: ClickToolData): Promise<ToolResult> => ipcRenderer.invoke('click-tool', data),
  keysTool: (data: KeysToolData): Promise<ToolResult> => ipcRenderer.invoke('keys-tool', data),

  // Recording toggle from tray
  toggleRecording: () => ipcRenderer.send('toggle-recording'),
  onToggleRecording: (callback: () => void) => {
    ipcRenderer.on('toggle-recording', callback);
    return () => ipcRenderer.removeListener('toggle-recording', callback);
  },

  // Cursor position updates
  onCursorUpdate: (callback: (coordinates: { x: number; y: number }) => void) => {
    ipcRenderer.on('cursor-update', (_event, coordinates) => callback(coordinates));
    return () => ipcRenderer.removeListener('cursor-update', callback);
  },

  // Recording state communication
  sendRecordingState: (isRecording: boolean) => ipcRenderer.send('recording-state-changed', isRecording),

  // Screenshot functionality
  takeScreenshot: (): Promise<ScreenshotResult> => ipcRenderer.invoke('take-screenshot'),

  // Audio debugging - save audio files to disk
  saveAudioFile: (audioData: Float32Array, filename: string): Promise<{ success: boolean; path?: string; error?: string }> =>
    ipcRenderer.invoke('save-audio-file', audioData, filename),
});
