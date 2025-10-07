import { app, BrowserWindow, ipcMain } from 'electron';
import { execFile } from "child_process";
import { promisify } from "util";
import robot from "@jitsi/robotjs";
import path from 'path';

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    frame: false,
    transparent: true,
    resizable: true,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),  // Built preload
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  win.once('ready-to-show', () => {
    win.maximize();
    win.setResizable(false);  // Keep previous non-resizable behavior after maximizing
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);  // HMR in dev
  } else {
    win.loadFile(path.join(__dirname, '../renderer/index.html'));
  }
}

ipcMain.handle("open-tool", async (event: any, data: any) => {
  console.log("Received from renderer:", data);

  const result = await openTool(data);
  return result;
});

const execFileAsync = promisify(execFile);

async function openTool(data: any): Promise<{ stdout: string, stderr: string }> {
  if (!data?.thing) throw new Error('missing data.thing');

  const { stdout, stderr } = (await execFileAsync('/usr/bin/open', [data.thing])) as {
    stdout: string;
    stderr: string;
  };

  console.log("stdout:", stdout);
  console.error("stderr:", stderr);
  return { stdout: stdout ?? '', stderr: stderr ?? '' };
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
