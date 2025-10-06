import { BrowserWindow, BrowserView } from "electrobun/bun";
import { type CuaRPC } from "../shared/types";
import robot from "@jitsi/robotjs";
import { $ } from "bun";

const rpc = BrowserView.defineRPC<CuaRPC>({
  maxRequestTime: 5000,
  handlers: {
    requests: {
      // FIXME: doesn't open applications, e.g. "open terminal"
      doOpen: async ({ thing }: { thing: string }) => {
        console.log(`Opening ${thing}...`)
        await $`open ${thing}`;
        return;
      }
    }
  }
});

// Create the main application window
const mainWindow = new BrowserWindow({
  title: "Cua",
  url: "views://mainview/index.html",
  frame: {
    width: 800,
    height: 800,
    x: 200,
    y: 200,
  },
  rpc,
});
