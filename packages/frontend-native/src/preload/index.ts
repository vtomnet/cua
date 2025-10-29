import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  // Example IPC
  sendMessage: (msg: string) => ipcRenderer.send('message', msg),
  openTool: (data: any) => ipcRenderer.invoke('open-tool', data),
  scrollTool: (data: any) => ipcRenderer.invoke('scroll-tool', data),

  // Recording toggle from tray
  toggleRecording: () => ipcRenderer.send('toggle-recording'),
  onToggleRecording: (callback: () => void) => {
    ipcRenderer.on('toggle-recording', callback);
    return () => ipcRenderer.removeListener('toggle-recording', callback);
  },

  // Recording state communication
  sendRecordingState: (isRecording: boolean) => ipcRenderer.send('recording-state-changed', isRecording),
});
