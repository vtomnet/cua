import open from "open-things";
import robot from "@jitsi/robotjs";

export interface ToolResult {
  success: boolean;
  output: string;
}

export interface OpenToolData {
  thing: string;
}

export interface ScrollToolData {
  direction?: "up" | "down" | "left" | "right";
  distance?: number;
}

export interface ClickToolData {
  x: number;
  y: number;
}

export interface KeysToolData {
  list: string[];
}

export async function openTool(data: OpenToolData): Promise<ToolResult> {
  console.log(`opening ${data.thing}`);
  const result = await open(data.thing);
  return { success: result.success, output: result.output };
}

export async function scrollTool(data: ScrollToolData): Promise<ToolResult> {
  try {
    const { direction = "down", distance = 70 } = data;

    // Validate direction
    const validDirections = ["up", "down", "left", "right"];
    if (!validDirections.includes(direction)) {
      return {
        success: false,
        output: `Invalid direction: ${direction}. Must be one of: ${validDirections.join(", ")}`
      };
    }

    // Validate distance
    if (typeof distance !== 'number' || distance <= 0 || distance > 100) {
      return {
        success: false,
        output: "Invalid distance: must be a number between 1 and 100 (percentage of viewport)"
      };
    }

    console.log(`Scrolling ${direction} by ${distance}% of viewport`);

    // Get screen size to calculate viewport center
    const screenSize = robot.getScreenSize();
    const centerX = Math.floor(screenSize.width / 2);
    const centerY = Math.floor(screenSize.height / 2);

    // Move mouse to center of screen to focus the window under cursor
    robot.moveMouse(centerX, centerY);

    // Calculate scroll distance based on screen size and percentage
    // Using a reasonable viewport assumption (most windows are not full screen)
    const viewportHeight = screenSize.height * 0.8; // Assume 80% of screen height for viewport
    const viewportWidth = screenSize.width * 0.8;   // Assume 80% of screen width for viewport

    let scrollDistance: number = 0;
    let scrollX = 0;
    let scrollY = 0;

    switch (direction) {
      case "up":
        scrollDistance = Math.floor((viewportHeight * distance) / 100);
        scrollY = scrollDistance; // Positive values scroll up
        break;
      case "down":
        scrollDistance = Math.floor((viewportHeight * distance) / 100);
        scrollY = -scrollDistance; // Negative values scroll down
        break;
      case "left":
        scrollDistance = Math.floor((viewportWidth * distance) / 100);
        scrollX = scrollDistance; // Positive values scroll left
        break;
      case "right":
        scrollDistance = Math.floor((viewportWidth * distance) / 100);
        scrollX = -scrollDistance; // Negative values scroll right
        break;
    }

    // Perform the scroll using robotjs
    robot.scrollMouse(scrollX, scrollY);

    return {
      success: true,
      output: `Scrolled ${direction} by ${distance}% (${Math.abs(scrollDistance)}px) from center of screen`
    };
  } catch (error) {
    console.error("Error scrolling:", error);
    return {
      success: false,
      output: error instanceof Error ? error.message : "Unknown error occurred"
    };
  }
}

export async function clickTool(data: ClickToolData): Promise<ToolResult> {
  try {
    const { x, y } = data;

    // Validate coordinates
    if (typeof x !== 'number' || typeof y !== 'number') {
      return {
        success: false,
        output: "Invalid coordinates: x and y must be numbers"
      };
    }

    if (x < 0 || y < 0) {
      return {
        success: false,
        output: "Invalid coordinates: x and y must be positive"
      };
    }

    console.log(`Clicking at coordinates (${x}, ${y})`);

    // Move mouse to position and click
    robot.moveMouse(x, y);
    robot.mouseClick();

    return {
      success: true,
      output: `Clicked at coordinates (${x}, ${y})`
    };
  } catch (error) {
    console.error("Error clicking:", error);
    return {
      success: false,
      output: error instanceof Error ? error.message : "Unknown error occurred"
    };
  }
}

export async function keysTool(data: KeysToolData): Promise<ToolResult> {
  try {
    const { list } = data;

    // Validate input
    if (!Array.isArray(list)) {
      return {
        success: false,
        output: "Invalid input: list must be an array of strings"
      };
    }

    if (list.length === 0) {
      return {
        success: false,
        output: "Invalid input: list cannot be empty"
      };
    }

    const processedKeys: string[] = [];

    for (const keyItem of list) {
      if (typeof keyItem !== 'string') {
        return {
          success: false,
          output: "Invalid input: all items in list must be strings"
        };
      }

      processedKeys.push(keyItem);

      // Check if this is a key combination (e.g., "<cmd>+c")
      if (keyItem.includes('+') && keyItem.includes('<') && keyItem.includes('>')) {
        console.log(`Processing key combination: ${keyItem}`);

        // Parse key combination like "<cmd>+c"
        const parts = keyItem.split('+');
        const modifiers: string[] = [];
        let mainKey = '';

        for (const part of parts) {
          const trimmedPart = part.trim();
          if (trimmedPart.startsWith('<') && trimmedPart.endsWith('>')) {
            // This is a modifier key
            const modifierKey = trimmedPart.slice(1, -1); // Remove < >
            modifiers.push(modifierKey);
          } else {
            // This is the main key
            mainKey = trimmedPart;
          }
        }

        // Press modifiers down
        for (const modifier of modifiers) {
          robot.keyToggle(modifier, 'down');
        }

        // Press main key
        if (mainKey) {
          robot.keyTap(mainKey);
        }

        // Release modifiers
        for (const modifier of modifiers.reverse()) {
          robot.keyToggle(modifier, 'up');
        }

      } else if (keyItem.startsWith('<') && keyItem.endsWith('>')) {
        // Single modifier or special key like "<ctrl>"
        const key = keyItem.slice(1, -1); // Remove < >
        console.log(`Processing single key: ${key}`);
        robot.keyTap(key);

      } else {
        // Regular text or single character
        console.log(`Typing text: ${keyItem}`);
        robot.typeString(keyItem);
      }
    }

    return {
      success: true,
      output: `Processed ${list.length} key input(s): ${processedKeys.join(', ')}`
    };
  } catch (error) {
    console.error("Error processing keys:", error);
    return {
      success: false,
      output: error instanceof Error ? error.message : "Unknown error occurred"
    };
  }
}
