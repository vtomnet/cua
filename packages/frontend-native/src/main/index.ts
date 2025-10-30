import { app, BrowserWindow, Tray, Menu, ipcMain, nativeImage, IpcMainEvent, IpcMainInvokeEvent, desktopCapturer, screen } from 'electron';
import path from 'path';
import { writeFileSync, mkdirSync } from 'fs';
import { homedir } from 'os';
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
});

ipcMain.handle("take-screenshot", async (_event: IpcMainInvokeEvent) => {
  console.log("Taking screenshot via Electron desktopCapturer");

  try {
    // Get primary display info
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: displayWidth, height: displayHeight } = primaryDisplay.size;
    const scaleFactor = primaryDisplay.scaleFactor;

    console.log(`Display: ${displayWidth}x${displayHeight} @ ${scaleFactor}x scale`);

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

    console.log(`Captured screenshot: ${originalSize.width}x${originalSize.height}`);

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

    console.log(`Resizing screenshot to: ${newWidth}x${newHeight}`);

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

    // Log file sizes for debugging
    const originalPngSize = thumbnail.toPNG().length;
    const compressedSize = resizedBuffer.length;
    console.log(`Original PNG size: ${(originalPngSize / 1024).toFixed(2)} KB`);
    console.log(`Compressed JPEG size: ${(compressedSize / 1024).toFixed(2)} KB (${((compressedSize / originalPngSize) * 100).toFixed(1)}% of original)`);

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
    const errorMessage = error instanceof Error ? error.message : "Unknown error occurred";
    console.error("Make sure Screen Recording permission is granted in System Settings > Privacy & Security > Screen Recording");
    return {
      success: false,
      error: errorMessage
    };
  }
});

// Function to convert Float32Array to WAV format
function float32ArrayToWav(audioData: Float32Array, sampleRate: number = 16000): Buffer {
  const length = audioData.length;
  const arrayBuffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(arrayBuffer);

  // WAV file header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  // RIFF chunk descriptor
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * 2, true); // ChunkSize
  writeString(8, 'WAVE');

  // fmt sub-chunk
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, 1, true); // AudioFormat (PCM)
  view.setUint16(22, 1, true); // NumChannels (mono)
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, sampleRate * 2, true); // ByteRate
  view.setUint16(32, 2, true); // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample

  // data sub-chunk
  writeString(36, 'data');
  view.setUint32(40, length * 2, true); // Subchunk2Size

  // Convert float32 samples to int16
  let offset = 44;
  for (let i = 0; i < length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i])); // Clamp to [-1, 1]
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
    offset += 2;
  }

  return Buffer.from(arrayBuffer);
}

ipcMain.handle("save-audio-file", async (_event: IpcMainInvokeEvent, audioData: Float32Array, filename: string) => {
  try {
    // Create audio debug directory in user's home folder
    const audioDebugDir = path.join(homedir(), 'cua-audio-debug');

    try {
      mkdirSync(audioDebugDir, { recursive: true });
    } catch (error) {
      // Directory might already exist, that's okay
    }

    // Convert Float32Array from renderer to actual Float32Array
    const float32Array = new Float32Array(audioData);

    // Detect sample rate based on filename (resampled files are 24kHz, others are 16kHz)
    const sampleRate = filename.includes('resampled') ? 24000 : 16000;

    // Convert to WAV format
    const wavBuffer = float32ArrayToWav(float32Array, sampleRate);

    // Create full file path with timestamp if no extension provided
    const baseName = filename.endsWith('.wav') ? filename : `${filename}.wav`;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const finalFilename = baseName.replace('.wav', `_${timestamp}.wav`);
    const filePath = path.join(audioDebugDir, finalFilename);

    // Write the WAV file
    writeFileSync(filePath, wavBuffer);

    console.log(`Audio file saved: ${filePath} (${float32Array.length} samples, ${(float32Array.length / sampleRate).toFixed(2)}s, ${sampleRate}Hz)`);

    return {
      success: true,
      path: filePath
    };
  } catch (error) {
    console.error("Error saving audio file:", error);
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
