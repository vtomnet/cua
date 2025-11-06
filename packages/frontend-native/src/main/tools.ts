import open from "open-things";
import robot from "@jitsi/robotjs";

export async function openFn(thing: string) {
  console.log(`opening ${thing}`);
  await open(thing);
}

export async function scrollFn(direction: "up" | "down" | "left" | "right" = "up", distance = 70) {
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
}

export async function clickFn(x: number, y: number) {
  robot.moveMouse(x, y);
  await new Promise(resolve => setTimeout(resolve, 50));
  robot.mouseClick();
}

export async function keysFn(list: string[]) {
  const processedKeys: string[] = [];

  for (const keyItem of list) {
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
          let modifierKey = trimmedPart.slice(1, -1); // Remove < >
          if (modifierKey == "cmd") modifierKey = "command";
          else if (modifierKey == "ctrl") modifierKey = "control";
          modifiers.push(modifierKey);
        } else {
          // This is the main key
          mainKey = trimmedPart;
        }
      }

      robot.keyTap(mainKey, modifiers);
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
}
