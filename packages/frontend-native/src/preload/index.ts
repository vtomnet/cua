import { contextBridge, ipcRenderer } from 'electron';

interface ScreenshotResult {
  image: string;
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

contextBridge.exposeInMainWorld('electronAPI', {
  // Example IPC
  sendMessage: (msg: string) => ipcRenderer.send('message', msg),

  // Recording toggle from tray
  toggleRecording: () => ipcRenderer.send('toggle-recording'),
  onToggleRecording: (callback: () => void) => {
    ipcRenderer.on('toggle-recording', callback);
    return () => ipcRenderer.removeListener('toggle-recording', callback);
  },

  // Cursor position updates
  onCursorUpdate: (callback: (coordinates: { x: number; y: number }) => void) => {
    const listener = (_event: any, coordinates: { x: number; y: number }) => callback(coordinates);
    ipcRenderer.on('cursor-update', listener);
    return () => ipcRenderer.removeListener('cursor-update', listener);
  },

  // Recording state communication
  sendRecordingState: (isRecording: boolean) => ipcRenderer.send('recording-state-changed', isRecording),

  // Screenshot functionality
  takeScreenshot: (): Promise<ScreenshotResult> => ipcRenderer.invoke('take-screenshot'),

  // Text submission from control panel
  submitText: (text: string): Promise<void> => ipcRenderer.invoke('submit-text', text),

  // Resize control window
  resizeControlWindow: (showSettings: boolean): Promise<void> => ipcRenderer.invoke('resize-control-window', showSettings),

  // Process text from control panel
  onProcessText: (callback: (text: string) => void) => {
    const listener = (_event: any, text: string) => callback(text);
    ipcRenderer.on('process-text', listener);
    return () => ipcRenderer.removeListener('process-text', listener);
  },

  // Recording state updates
  onRecordingStateUpdate: (callback: (isRecording: boolean) => void) => {
    const listener = (_event: any, isRecording: boolean) => callback(isRecording);
    ipcRenderer.on('recording-state-update', listener);
    return () => ipcRenderer.removeListener('recording-state-update', listener);
  },

  // Settings management
  getSetting: (key: string): Promise<any> => ipcRenderer.invoke('get-setting', key),
  setSetting: (key: string, value: any): Promise<void> => ipcRenderer.invoke('set-setting', key, value),

  // Get current application
  getCurrentApp: (): Promise<CurrentAppInfo> => ipcRenderer.invoke('get-current-app'),

  // Execute JavaScript code
  executeJavaScript: (code: string): Promise<void> => ipcRenderer.invoke('execute-javascript', code),
});
