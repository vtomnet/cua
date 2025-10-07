// packages/frontend-native/preload.ts
import { contextBridge } from 'electron';

// Expose safe APIs to the renderer
contextBridge.exposeInMainWorld('api', {
  ping: () => 'pong',
});
