import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  // Example IPC
  sendMessage: (msg: string) => ipcRenderer.send('message', msg)
});
