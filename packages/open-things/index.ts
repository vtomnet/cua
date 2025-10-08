import { execFile } from "child_process";

// on macOS, use Launch Services API

// need to think more about this. Perhaps presence of `error` field
// is sufficient to replace `success` field. Is `output` needed at
// all? In case of failure, everything should go into `error`. In
// case of success, I'm not sure we expect any output. So `error`
// may be the only necessary field. Although, it may be useful to
// signal which application was opened; this aligns with Apple's
// LaunchServices interface. Do we get that from Windows and Linux
// as well?
interface OpenResult {
  success: boolean;
  output: string;
  error?: string;
}

export default async function open(things: string | Array<string>): Promise<OpenResult> {
  const thingList = Array.isArray(things) ? things : [things];

  return new Promise((resolve) => {
    execFile("/usr/bin/open", thingList, (error, stdout, stderr) => {
      if (error) {
        resolve({ success: false, output: stdout, error: stderr || error.message });
      } else {
        resolve({ success: true, output: stdout });
      }
    });
  });
}
