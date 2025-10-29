interface ElectronAPI {
  sendMessage: (msg: string) => void;
  openTool: (data: any) => Promise<any>;
  scrollTool: (data: any) => Promise<any>;
  toggleRecording: () => void;
  onToggleRecording: (callback: () => void) => (() => void);
  sendRecordingState: (isRecording: boolean) => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

export {};