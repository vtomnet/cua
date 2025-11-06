import { screen, desktopCapturer } from "electron";
import { execSync } from 'child_process';

interface CurrentAppInfo {
  name: string;
  url?: string;
  title?: string;
}

export async function takeScreenshot() {
  try {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width: displayWidth, height: displayHeight } = primaryDisplay.size;
    const scaleFactor = primaryDisplay.scaleFactor;

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

    const primarySource = sources[0];
    const thumbnail = primarySource.thumbnail;
    const originalSize = thumbnail.getSize();

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

export async function getCurrentApp(): Promise<CurrentAppInfo> {
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
}
