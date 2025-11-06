import {
  app,
  BrowserWindow,
  Tray,
  Menu,
  ipcMain,
  nativeImage,
  IpcMainEvent,
  IpcMainInvokeEvent,
  desktopCapturer,
  screen,
  session,
} from 'electron';
import path from 'path';
import fs from 'fs';
import { execSync } from 'child_process';
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
    const result = await openTool({ thing });
    return result.output;
  },
  scroll: async (direction: string, distance: number = 70) => {
    const result = await scrollTool({
      direction: direction as "up" | "down" | "left" | "right",
      distance
    });
    return result.output;
  },
  click: async (x: number, y: number) => {
    const result = await clickTool({ x, y });
    return result.output;
  },
  screenshot: async () => {
    // Screenshot is handled separately in the agent flow
    return "Screenshot taken";
  },
  keys: async (list: string[]) => {
    const result = await keysTool({ list });
    return result.output;
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
    console.log(`[sandbox] [level:${level}, line:${line}] ${message}`)
  });

  const src = `
    (() => {
      'use strict';

      function open(thing) { return window.sandboxApi.call('open', thing); }
      function scroll(direction, distance = 70) { return window.sandboxApi.call('scroll', [direction, distance]); }
      function click(x, y) { return window.sandboxApi.call('click', [x, y]); }
      function screenshot() { return window.sandboxApi.call('screenshot', []); }
      function keys(list) { return window.sandboxApi.call('keys', list); }

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
    return {
      success: true,
      message: "Code executed successfully"
    };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    console.error('JavaScript execution error:', errorMsg);
    return {
      success: false,
      error: errorMsg
    };
  }
});

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

interface CurrentAppInfo {
  name: string;
  url?: string;
  title?: string;
}

// Helper function to get browser info on macOS
async function getMacBrowserInfo(appName: string): Promise<CurrentAppInfo> {
  const result: CurrentAppInfo = { name: appName };

  try {
    // Map of common browser names to their AppleScript identifiers
    const browserMap: Record<string, string> = {
      'Google Chrome': 'Google Chrome',
      'Chrome': 'Google Chrome',
      'Safari': 'Safari',
      'Firefox': 'Firefox',
      'Microsoft Edge': 'Microsoft Edge',
      'Edge': 'Microsoft Edge',
      'Brave Browser': 'Brave Browser',
      'Brave': 'Brave Browser',
      'Opera': 'Opera',
      'Arc': 'Arc',
    };

    const browserIdentifier = browserMap[appName];
    if (!browserIdentifier) {
      return result; // Not a known browser
    }

    // Try to get URL and title based on browser type
    if (browserIdentifier === 'Safari') {
      const script = `
        tell application "Safari"
          if (count of windows) > 0 then
            set currentTab to current tab of front window
            set tabURL to URL of currentTab
            set tabTitle to name of currentTab
            return tabURL & "|||" & tabTitle
          end if
        end tell
      `;
      const output = execSync(`osascript -e '${script}'`, { encoding: 'utf-8' }).trim();
      const [url, title] = output.split('|||');
      if (url) result.url = url;
      if (title) result.title = title;
    } else if (browserIdentifier === 'Google Chrome' || browserIdentifier === 'Microsoft Edge' ||
               browserIdentifier === 'Brave Browser' || browserIdentifier === 'Opera' ||
               browserIdentifier === 'Arc') {
      // Chrome, Edge, Brave, Opera, and Arc use similar AppleScript syntax
      const script = `
        tell application "${browserIdentifier}"
          if (count of windows) > 0 then
            set currentTab to active tab of front window
            set tabURL to URL of currentTab
            set tabTitle to title of currentTab
            return tabURL & "|||" & tabTitle
          end if
        end tell
      `;
      const output = execSync(`osascript -e '${script}'`, { encoding: 'utf-8' }).trim();
      const [url, title] = output.split('|||');
      if (url) result.url = url;
      if (title) result.title = title;
    } else if (browserIdentifier === 'Firefox') {
      // Firefox doesn't have good AppleScript support, try alternative
      // We can try using the window title which often contains the page title
      try {
        const titleScript = 'tell application "System Events" to get title of front window of process "Firefox"';
        const windowTitle = execSync(`osascript -e '${titleScript}'`, { encoding: 'utf-8' }).trim();
        if (windowTitle && windowTitle !== 'Firefox') {
          result.title = windowTitle;
          // Firefox window titles typically end with " — Mozilla Firefox" or similar
          result.title = windowTitle.replace(/ [-—] Mozilla Firefox$/, '');
        }
      } catch (e) {
        console.log('Could not get Firefox title:', e);
      }
    }
  } catch (error) {
    console.log(`Could not get browser info for ${appName}:`, error);
  }

  return result;
}

// Helper function to get browser info on Windows
async function getWindowsBrowserInfo(appName: string): Promise<CurrentAppInfo> {
  const result: CurrentAppInfo = { name: appName };

  try {
    // Check if it's a known browser
    const browsers = ['chrome', 'msedge', 'firefox', 'brave', 'opera', 'iexplore'];
    const browserName = appName.toLowerCase();

    if (!browsers.some(b => browserName.includes(b))) {
      return result; // Not a browser
    }

    // Try to get window title which often contains URL info
    const script = `Add-Type @"
      using System;
      using System.Runtime.InteropServices;
      using System.Text;
      public class Win32 {
        [DllImport("user32.dll")]
        public static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")]
        public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
      }
"@
    $hwnd = [Win32]::GetForegroundWindow()
    $title = New-Object System.Text.StringBuilder 256
    [Win32]::GetWindowText($hwnd, $title, 256) | Out-Null
    $title.ToString()`;

    const windowTitle = execSync(`powershell -Command "${script}"`, { encoding: 'utf-8' }).trim();

    if (windowTitle) {
      result.title = windowTitle;
      // Clean up browser-specific suffixes
      result.title = windowTitle
        .replace(/ - Google Chrome$/, '')
        .replace(/ - Microsoft Edge$/, '')
        .replace(/ — Mozilla Firefox$/, '')
        .replace(/ - Opera$/, '')
        .replace(/ - Brave$/, '');
    }
  } catch (error) {
    console.log(`Could not get browser info for ${appName}:`, error);
  }

  return result;
}

ipcMain.handle("get-current-app", async (_event: IpcMainInvokeEvent): Promise<CurrentAppInfo> => {
  try {
    if (process.platform === 'darwin') {
      // macOS: Use AppleScript to get the frontmost application
      const script = 'tell application "System Events" to get name of first application process whose frontmost is true';
      const appName = execSync(`osascript -e '${script}'`, { encoding: 'utf-8' }).trim();

      // Try to get browser info if it's a browser
      return await getMacBrowserInfo(appName);
    } else if (process.platform === 'win32') {
      // Windows: Use PowerShell to get the foreground window
      const script = `Add-Type @"
        using System;
        using System.Runtime.InteropServices;
        using System.Text;
        public class Win32 {
          [DllImport("user32.dll")]
          public static extern IntPtr GetForegroundWindow();
          [DllImport("user32.dll")]
          public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
          [DllImport("user32.dll", SetLastError=true)]
          public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
        }
"@
      $hwnd = [Win32]::GetForegroundWindow()
      $processId = 0
      [Win32]::GetWindowThreadProcessId($hwnd, [ref]$processId) | Out-Null
      $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
      if ($process) { $process.ProcessName } else { "Unknown" }`;
      const appName = execSync(`powershell -Command "${script}"`, { encoding: 'utf-8' }).trim();

      // Try to get browser info if it's a browser
      return await getWindowsBrowserInfo(appName);
    } else if (process.platform === 'linux') {
      // Linux: Try using xdotool or wmctrl
      try {
        const windowId = execSync('xdotool getactivewindow', { encoding: 'utf-8' }).trim();
        const appName = execSync(`xdotool getwindowname ${windowId}`, { encoding: 'utf-8' }).trim();
        // For Linux, the window name often includes the title, so use it
        return { name: appName, title: appName };
      } catch {
        // Fallback to wmctrl
        const output = execSync('wmctrl -lx', { encoding: 'utf-8' });
        const lines = output.split('\n');
        const appName = lines[0]?.split(/\s+/)[2] || "Unknown";
        return { name: appName };
      }
    }
    return { name: "Unknown" };
  } catch (error) {
    console.error('Error getting current app:', error);
    return { name: "Unknown" };
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
