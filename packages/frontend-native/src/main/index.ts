import { app, BrowserWindow, Tray, ipcMain, nativeImage } from 'electron';
import path from 'path';
import { openTool, scrollTool } from "./tools";

let tray: Tray | null = null;
let isRecording = false;

function updateTrayIcon() {
  if (!tray) return;

  const iconName = isRecording ? 'microphone-slash-tray.png' : 'microphone-tray.png';
  const iconPath = process.env.VITE_DEV_SERVER_URL
    ? path.join(process.cwd(), iconName)
    : path.join(__dirname, iconName);

  console.log('Updating tray icon to:', iconName);

  try {
    if (process.platform === 'darwin') {
      const imageObj = nativeImage.createFromPath(iconPath);
      imageObj.setTemplateImage(true);
      tray.setImage(imageObj);
    } else {
      tray.setImage(iconPath);
    }

    const tooltip = isRecording
      ? 'Xyzzy - Recording (click to stop)'
      : 'Xyzzy - Click to start recording';
    tray.setToolTip(tooltip);
  } catch (error) {
    console.error('Error updating tray icon:', error);
  }
}

function createTray() {
  // Start with microphone icon (not recording)
  const iconPath = process.env.VITE_DEV_SERVER_URL
    ? path.join(process.cwd(), 'microphone-tray.png')
    : path.join(__dirname, 'microphone-tray.png');

  console.log('Tray icon path:', iconPath);
  console.log('Icon exists:', require('fs').existsSync(iconPath));

  try {
    tray = new Tray(iconPath);
    console.log('Tray created successfully');

    // Set initial icon state
    updateTrayIcon();

    // Handle tray click to toggle recording
    tray.on('click', () => {
      console.log('Tray clicked - toggling recording');
      const windows = BrowserWindow.getAllWindows();
      if (windows.length > 0) {
        windows[0].webContents.send('toggle-recording');
      }
    });

    // No context menu - tray acts as pure toggle button

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
}

ipcMain.handle("open-tool", async (_event: any, data: any) => {
  console.log("Open tool received from renderer:", data);

  const result = await openTool(data);
  return result;
});

ipcMain.handle("scroll-tool", async (_event: any, data: any) => {
  console.log("Scroll tool received from renderer:", data);

  const result = await scrollTool(data);
  return result;
});

ipcMain.on("recording-state-changed", (_event: any, recordingState: boolean) => {
  console.log("Recording state changed:", recordingState);
  isRecording = recordingState;
  updateTrayIcon();
});

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
