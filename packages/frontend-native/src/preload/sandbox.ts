import { contextBridge, ipcRenderer } from 'electron';

const allowedMethods = new Set(['open', 'scroll', 'click', 'screenshot', 'keys']);

contextBridge.exposeInMainWorld('sandboxApi', {
    call(name: string, args?: any): Promise<any> {
        if (!allowedMethods.has(name)) {
            throw new Error(`Disallowed method: ${name}`);
        }
        return ipcRenderer.invoke('sandbox:call', { name, args });
    }
});
