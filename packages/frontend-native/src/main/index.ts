import { app, BrowserWindow, Tray, Menu, ipcMain, nativeImage, IpcMainEvent, IpcMainInvokeEvent, desktopCapturer, screen } from 'electron';
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
let controlWindow: BrowserWindow | null = null;

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


ipcMain.handle("open-tool", async (_event: IpcMainInvokeEvent, data: OpenToolData) => {
  console.log("Open tool received from renderer:", data);

  const result = await openTool(data);
  return result;
});

ipcMain.handle("scroll-tool", async (event: IpcMainInvokeEvent, data: ScrollToolData) => {
  console.log("Scroll tool received from renderer:", data);

  // Calculate center coordinates for cursor positioning (same logic as in scrollTool)
  const screenSize = robot.getScreenSize();
  const centerX = Math.floor(screenSize.width / 2);
  const centerY = Math.floor(screenSize.height / 2);

  // Send cursor position update to renderer before executing the scroll
  const windows = BrowserWindow.getAllWindows();
  if (windows.length > 0) {
    windows[0].webContents.send('cursor-update', { x: centerX, y: centerY });
  }

  const result = await scrollTool(data);
  return result;
});

ipcMain.handle("click-tool", async (event: IpcMainInvokeEvent, data: ClickToolData) => {
  console.log("Click tool received from renderer:", data);

  // Send cursor position update to renderer before executing the click
  const windows = BrowserWindow.getAllWindows();
  if (windows.length > 0) {
    windows[0].webContents.send('cursor-update', { x: data.x, y: data.y });
  }

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
  return { success: true };
});

ipcMain.handle("resize-control-window", async (_event: IpcMainInvokeEvent, showSettings: boolean) => {
  console.log("Resizing control window. Show settings:", showSettings);

  if (!controlWindow || controlWindow.isDestroyed()) {
    return { success: false };
  }

  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

  const windowWidth = 400;
  const normalHeight = 180;
  const expandedHeight = 450; // Larger height to accommodate settings
  const newHeight = showSettings ? expandedHeight : normalHeight;

  // Reposition to keep it centered horizontally and at bottom
  const x = Math.floor((screenWidth - windowWidth) / 2);
  const y = screenHeight - newHeight - 20; // 20px from bottom

  controlWindow.setBounds({
    x,
    y,
    width: windowWidth,
    height: newHeight
  });

  return { success: true };
});

ipcMain.handle("take-screenshot", async (_event: IpcMainInvokeEvent) => {
  try {
    // Get primary display info
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: displayWidth, height: displayHeight } = primaryDisplay.size;
    const scaleFactor = primaryDisplay.scaleFactor;

    // Use desktopCapturer to get screen sources
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: {
        width: displayWidth * scaleFactor,
        height: displayHeight * scaleFactor
      }
    });

    if (sources.length === 0) {
      throw new Error('No screen sources found. Screen Recording permission may not be granted.');
    }

    // Use the first screen source (primary display)
    const primarySource = sources[0];
    const thumbnail = primarySource.thumbnail;
    const originalSize = thumbnail.getSize();

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

    // Resize the image
    const resizedImage = thumbnail.resize({
      width: newWidth,
      height: newHeight,
      quality: 'good'
    });

    // Convert to JPEG with quality compression (much smaller than PNG)
    // Quality 70 provides a good balance between file size and visual quality
    const resizedBuffer = resizedImage.toJPEG(70);
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
    console.error(`Error taking screenshot: ${error}\nDid you grant Screen Recording permissions?`, error);
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
