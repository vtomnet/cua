// Mediapipe, Silero VAD via ORT, maybe audio/camera recording routines. Seamlessly handles running fully locally, or offloading to server, via FSM.

import { hello } from "core";

export function foo() {
  return hello();
}

export { hello };
export * from './webrtc-audio';
