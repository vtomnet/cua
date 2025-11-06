import {
  app,
  BrowserWindow,
  Tray,
  Menu,
  ipcMain,
  nativeImage,
  IpcMainEvent,
  IpcMainInvokeEvent,
  screen,
  session,
} from 'electron';
import path from 'path';
import fs from 'fs';
import {
  openFn,
  scrollFn,
  clickFn,
  keysFn,
} from "./tools";
import { takeScreenshot, getCurrentApp } from "./capture";
import micTrayImg from "../assets/microphone-tray.png";
import micSlashTrayImg from "../assets/microphone-slash-tray.png";

let tray: Tray | null = null;
let isRecording = false;
let controlWindow: BrowserWindow | null = null;


// Settings storage
const settingsPath = path.join(app.getPath('userData'), 'settings.json');

interface Settings {
  recordOnLaunch?: boolean;
  [key: string]: any;
}

function loadSettings(): Settings {
  try {
    if (fs.existsSync(settingsPath)) {
      const data = fs.readFileSync(settingsPath, 'utf-8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error('Error loading settings:', error);
  }
  return { recordOnLaunch: true }; // Default settings
}

function saveSettings(settings: Settings): void {
  try {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf-8');
  } catch (error) {
    console.error('Error saving settings:', error);
  }
}

function updateTrayIcon() {
  if (!tray) return;

  const img = nativeImage.createFromDataURL(isRecording ? micTrayImg : micSlashTrayImg);
  if (process.platform === 'darwin') img.setTemplateImage(true);

  try {
    tray.setImage(img);
    tray.setToolTip(isRecording ? 'Click to stop recording' : 'Click to start recording');
  } catch (error) {
    console.error('Error updating tray icon:', error);
  }
}

function createTray() {
  // Start with microphone icon (not recording)
  const img = nativeImage.createFromDataURL(micTrayImg);
  if (process.platform === 'darwin') img.setTemplateImage(true);

  try {
    tray = new Tray(img);
    console.log('Tray created successfully');

    // Set initial icon state
    updateTrayIcon();

    // Handle left click to open control window
    tray.on('click', () => {
      console.log('Tray left-clicked - opening control window');
      createControlWindow();
    });

    // Handle right click to show context menu
    tray.on('right-click', () => {
      console.log('Tray right-clicked - showing context menu');
      const contextMenu = Menu.buildFromTemplate([
        {
          label: 'Open Devtools',
          click: () => {
            const windows = BrowserWindow.getAllWindows();
            if (windows.length > 0) {
              windows[0].webContents.openDevTools();
            }
          }
        },
        {
          type: 'separator'
        },
        {
          label: 'Quit',
          click: () => {
            app.quit();
          }
        }
      ]);

      tray?.popUpContextMenu(contextMenu);
    });

    // Make sure the tray icon is visible
    tray.setIgnoreDoubleClickEvents(false);
  } catch (error) {
    console.error('Error creating tray:', error);
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    frame: false,
    transparent: true,
    resizable: true,
    hasShadow: false,
    alwaysOnTop: true,
    focusable: false,
    show: false,
    skipTaskbar: true,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    win.loadFile(path.join(__dirname, '../renderer/index.html'));
  }

  win.once('ready-to-show', () => {
    win.maximize();
    win.setResizable(false);
    win.setIgnoreMouseEvents(true);
    win.showInactive();
  });

  // Prevent focus from Mission Control/App Exposé on macOS
  win.on('focus', () => {
    console.log('Window gained focus - immediately blurring');
    win.blur();
  });

  // Additional safeguard for macOS Mission Control
  win.on('show', () => {
    if (process.platform === 'darwin') {
      win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
      win.setIgnoreMouseEvents(true);
    }
  });
}

function createControlWindow() {
  // Don't create if already exists
  if (controlWindow && !controlWindow.isDestroyed()) {
    controlWindow.focus();
    return;
  }

  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

  const windowWidth = 400;
  const windowHeight = 180;

  // Position at bottom-center of screen
  const x = Math.floor((screenWidth - windowWidth) / 2);
  const y = screenHeight - windowHeight - 20; // 20px from bottom

  controlWindow = new BrowserWindow({
    width: windowWidth,
    height: windowHeight,
    x,
    y,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: false,
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  // Load control panel HTML
  if (process.env.VITE_DEV_SERVER_URL) {
    controlWindow.loadURL(`${process.env.VITE_DEV_SERVER_URL}/control.html`);
  } else {
    controlWindow.loadFile(path.join(__dirname, '../renderer/control.html'));
  }

  controlWindow.on('closed', () => {
    controlWindow = null;
  });
}

async function createSandbox() {
  const partition = 'sandbox:' + Math.random().toString(36).slice(2);
  const ses = session.fromPartition(partition, { cache: false });

  ses.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));
  ses.webRequest.onBeforeRequest({ urls: ['*://*/*'] }, (_d, cb) => cb({ cancel: true }));

  const win = new BrowserWindow({
    show: false,
    webPreferences: {
      partition,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      autoplayPolicy: 'document-user-activation-required',
      preload: path.join(__dirname, '../preload/sandbox.js'),
    }
  });

  win.webContents.on('will-navigate', (e) => e.preventDefault());
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));

  await win.loadURL('about:blank');
  return win;
}

// FIXME: security hole: prompt injections possible via app title / webpage title / webpage url. llama-guard?
const sandboxFns = {
  open: async (thing: string) => {
    await openFn(thing);
  },
  scroll: async (direction: string, distance: number = 70) => {
    await scrollFn(direction as "up" | "down" | "left" | "right", distance);
  },
  click: async (x: number, y: number) => {
    await clickFn(x, y);
  },
  screenshot: async () => {
    return "Screenshot taken"; // TODO
  },
  keys: async (list: string[]) => {
    await keysFn(list);
  },
};

ipcMain.handle('sandbox:call', async (_event, { name, args }) => {
  const fn = (sandboxFns as any)[name];
  if (!fn) {
    throw new Error(`Unknown sandbox function: ${name}`);
  }
  // Handle both single argument and multiple arguments
  if (Array.isArray(args)) {
    return await fn(...args);
  } else {
    return await fn(args);
  }
});

async function runSandboxed(code: string, timeoutMs = 60_000) {
  const win = await createSandbox();

  // Capture console messages from the sandbox
  win.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`[sandbox] [level:${level}, line:${line}] ${message}`);
  });

  const src = `
    (() => {
      'use strict';

      function open(thing) { return window.sandboxApi.call('open', thing); }
      function scroll(direction, distance = 70) { return window.sandboxApi.call('scroll', [direction, distance]); }
      function click(x, y) { return window.sandboxApi.call('click', [x, y]); }
      function screenshot() { return window.sandboxApi.call('screenshot', []); }
      function keys(list) { return window.sandboxApi.call('keys', [list]); }

      ${code}
    })();
  `.trim();

  const run = win.webContents.executeJavaScript(src, true);
  const result = await Promise.race([
    run,
    new Promise((_r, reject) => setTimeout(() => reject(new Error("Sandbox timed out")), timeoutMs)),
  ]).finally(() => {
    if (!win.isDestroyed()) win.destroy();
  });

  return result;
}

// JavaScript sandbox execution
ipcMain.handle("execute-javascript", async (_event: IpcMainInvokeEvent, code: string) => {
  console.log("Executing JavaScript code:", code);

  try {
    await runSandboxed(code);
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('JavaScript execution error:', errorMsg);
    throw new Error(`JavaScript execution failed: ${errorMsg}`);
  }
});

ipcMain.on("recording-state-changed", (_event: IpcMainEvent, recordingState: boolean) => {
  console.log("Recording state changed:", recordingState);
  isRecording = recordingState;
  updateTrayIcon();

  // Notify all windows of state change
  BrowserWindow.getAllWindows().forEach(win => {
    win.webContents.send('recording-state-update', recordingState);
  });
});

ipcMain.on("toggle-recording", () => {
  console.log("Toggle recording request received");
  const overlayWindow = BrowserWindow.getAllWindows().find(win => win !== controlWindow);
  if (overlayWindow) {
    overlayWindow.webContents.send('toggle-recording');
  }
});

ipcMain.handle("submit-text", async (_event: IpcMainInvokeEvent, text: string) => {
  console.log("Text submitted from control panel:", text);
  // Send the text to the overlay window to be processed by runAgent
  const windows = BrowserWindow.getAllWindows();
  const overlayWindow = windows.find(win => win !== controlWindow);
  if (overlayWindow) {
    overlayWindow.webContents.send('process-text', text);
  }
  // Success - no return value needed
});

ipcMain.handle("resize-control-window", async (_event: IpcMainInvokeEvent, showSettings: boolean) => {
  console.log("Resizing control window. Show settings:", showSettings);

  if (!controlWindow || controlWindow.isDestroyed()) {
    throw new Error("Control window is not available");
  }

  const windowWidth = 400;
  const normalHeight = 180;
  const expandedHeight = 450; // Larger height to accommodate settings
  const newHeight = showSettings ? expandedHeight : normalHeight;

  // Get current window bounds to preserve position
  const currentBounds = controlWindow.getBounds();

  // Only resize height, keep x and y position
  controlWindow.setBounds({
    x: currentBounds.x,
    y: currentBounds.y,
    width: windowWidth,
    height: newHeight
  });
});

ipcMain.handle("get-setting", async (_event: IpcMainInvokeEvent, key: string) => {
  const settings = loadSettings();
  return settings[key];
});

ipcMain.handle("set-setting", async (_event: IpcMainInvokeEvent, key: string, value: any) => {
  const settings = loadSettings();
  settings[key] = value;
  saveSettings(settings);
});

ipcMain.handle("get-initial-recording-state", async (_event: IpcMainInvokeEvent) => {
  const settings = loadSettings();
  return settings.recordOnLaunch !== false; // Default to true
});

ipcMain.handle("take-screenshot", takeScreenshot);

ipcMain.handle("get-current-app", getCurrentApp);

app.whenReady().then(() => {
  createTray();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
