import { app, BrowserWindow, Tray, Menu, ipcMain, nativeImage, IpcMainEvent, IpcMainInvokeEvent } from 'electron';
import path from 'path';
import robot from '@jitsi/robotjs';
import {
  openTool,
  scrollTool,
  clickTool,
  keysTool,
  OpenToolData,
  ScrollToolData,
  ClickToolData,
  KeysToolData
} from "./tools";
import micTrayImg from "../assets/microphone-tray.png";
import micSlashTrayImg from "../assets/microphone-slash-tray.png"

let tray: Tray | null = null;
let isRecording = false;

function updateTrayIcon() {
  if (!tray) return;

  const img = nativeImage.createFromDataURL(isRecording ? micTrayImg : micSlashTrayImg);
  if (process.platform === 'darwin') img.setTemplateImage(true);

  try {
    tray.setImage(img);

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
  const img = nativeImage.createFromDataURL(micTrayImg);
  if (process.platform === 'darwin') img.setTemplateImage(true);

  try {
    tray = new Tray(img);
    console.log('Tray created successfully');

    // Set initial icon state
    updateTrayIcon();

    // Handle left click to toggle recording
    tray.on('click', () => {
      console.log('Tray left-clicked - toggling recording');
      const windows = BrowserWindow.getAllWindows();
      if (windows.length > 0) {
        windows[0].webContents.send('toggle-recording');
      }
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

ipcMain.handle("open-tool", async (_event: IpcMainInvokeEvent, data: OpenToolData) => {
  console.log("Open tool received from renderer:", data);

  const result = await openTool(data);
  return result;
});

ipcMain.handle("scroll-tool", async (_event: IpcMainInvokeEvent, data: ScrollToolData) => {
  console.log("Scroll tool received from renderer:", data);

  const result = await scrollTool(data);
  return result;
});

ipcMain.handle("click-tool", async (_event: IpcMainInvokeEvent, data: ClickToolData) => {
  console.log("Click tool received from renderer:", data);

  const result = await clickTool(data);
  return result;
});

ipcMain.handle("keys-tool", async (_event: IpcMainInvokeEvent, data: KeysToolData) => {
  console.log("Keys tool received from renderer:", data);

  const result = await keysTool(data);
  return result;
});

ipcMain.on("recording-state-changed", (_event: IpcMainEvent, recordingState: boolean) => {
  console.log("Recording state changed:", recordingState);
  isRecording = recordingState;
  updateTrayIcon();
});

ipcMain.handle("take-screenshot", async (_event: IpcMainInvokeEvent) => {
  console.log("Taking screenshot via robotjs");

  try {
    // Take screenshot using robotjs
    const screenshot = robot.screen.capture();

    // Convert the screenshot to a native image for processing
    const buffer = Buffer.from(screenshot.image, 'binary');
    const originalImage = nativeImage.createFromBuffer(buffer);

    // Get original dimensions
    const originalSize = originalImage.getSize();
    console.log(`Original screenshot size: ${originalSize.width}x${originalSize.height}`);

    // Calculate new dimensions to fit within 1024px max dimension while maintaining aspect ratio
    const maxDimension = 1024;
    let newWidth = originalSize.width;
    let newHeight = originalSize.height;

    if (originalSize.width > maxDimension || originalSize.height > maxDimension) {
      const aspectRatio = originalSize.width / originalSize.height;

      if (originalSize.width > originalSize.height) {
        // Landscape: limit width
        newWidth = maxDimension;
        newHeight = Math.round(maxDimension / aspectRatio);
      } else {
        // Portrait: limit height
        newHeight = maxDimension;
        newWidth = Math.round(maxDimension * aspectRatio);
      }
    }

    console.log(`Resized screenshot to: ${newWidth}x${newHeight}`);

    // Resize the image
    const resizedImage = originalImage.resize({
      width: newWidth,
      height: newHeight,
      quality: 'good'
    });

    // Convert resized image to base64
    const resizedBuffer = resizedImage.toPNG();
    const base64Image = resizedBuffer.toString('base64');

    return {
      success: true,
      image: base64Image,
      width: newWidth,
      height: newHeight,
      originalWidth: originalSize.width,
      originalHeight: originalSize.height
    };
  } catch (error) {
    console.error("Error taking screenshot:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred"
    };
  }
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
