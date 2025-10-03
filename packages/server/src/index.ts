import express from "express";
import * as http from "http";
import WebSocket from "ws";
import nodeDataChannel from "node-datachannel";

// Types for WebSocket messages
interface OfferMessage {
  type: 'offer';
  offer: {
    type: string;
    sdp: string;
  };
}

interface CandidateMessage {
  type: 'candidate';
  candidate: {
    candidate: string;
    sdpMid: string | null;
    sdpMLineIndex: number | null;
  };
}

type WebSocketMessage = OfferMessage | CandidateMessage;

// Types for node-datachannel (based on actual library interface)
interface NodeDataChannelPeerConnection {
  onLocalCandidate(callback: (candidate: string, mid: string) => void): void;
  onStateChange(callback: (state: string) => void): void;
  onDataChannel(callback: (dc: NodeDataChannel) => void): void;
  setRemoteDescription(sdp: string, type: string): void;
  localDescription(): { type: string; sdp: string } | null;
  addRemoteCandidate(candidate: string, sdpMid?: string | null): void;
  close(): void;
}

interface NodeDataChannel {
  getLabel(): string;
  onMessage(callback: (msg: string | ArrayBuffer | Buffer) => void): void;
  onOpen(callback: () => void): void;
  onClosed(callback: () => void): void;
  sendMessage(data: string | ArrayBuffer | Buffer): void;
}

// WebRTC/WS/HTTP server, counterpart to frontend-{web,native}. Builds on 'core'.
// Can theoretically scale from small models for e.g. running on rpi, to large models if running on gpu

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static("public"));

const peers = new Map();

wss.on("connection", (ws: WebSocket) => {
  const clientId = Math.random().toString(36).substring(7);
  console.log("Client connected:", clientId);

  let pc: NodeDataChannelPeerConnection | null = null;

  ws.on("message", (data: WebSocket.RawData) => {
    try {
      const message = JSON.parse(data.toString()) as WebSocketMessage;

      switch (message.type) {
        case "offer":
          handleOffer(message.offer);
          break;
        case "candidate":
          handleCandidate(message.candidate);
          break;
      }
    } catch (err) {
      console.error("Error parsing message:", err);
      ws.send(JSON.stringify({ type: "error", error: (err as Error).message }));
    }
  });

  async function handleOffer(offer: { type: string; sdp: string }) {
    try {
      console.log("Received offer from:", clientId);

      pc = new nodeDataChannel.PeerConnection(clientId, {
        iceServers: ["stun:stun.l.google.com:19302"]
      });

      peers.set(clientId, pc);

      pc.onLocalCandidate((candidate: string, mid: string) => {
        ws.send(JSON.stringify({
          type: "candidate",
          candidate: candidate
        }));
      });

      pc.onStateChange((state: string) => {
        console.log(`Peer connection state (${clientId}):`, state);
      });

      // Handle data channel for audio
      pc.onDataChannel((dc: NodeDataChannel) => {
        console.log('Data channel opened:', dc.getLabel());

        dc.onMessage((msg: string | ArrayBuffer | Buffer) => {
          // msg is a Buffer containing audio data
          const length = msg instanceof ArrayBuffer ? msg.byteLength : (msg as Buffer).length;
          console.log(`Received audio chunk: ${length} bytes`);

          // Process audio data here
          // You can:
          // - Save to file
          // - Stream to speech-to-text service
          // - Process with audio analysis
          // - Broadcast to other clients
        });

        dc.onOpen(() => {
          console.log('Data channel is open');
          dc.sendMessage('Audio channel ready');
        });

        dc.onClosed(() => {
          console.log('Data channel closed');
        });
      });

      // Set remote description
      pc.setRemoteDescription(offer.sdp, offer.type);

      // Create and send answer
      const answerSdp = pc.localDescription();
      if (answerSdp) {
        ws.send(JSON.stringify({
          type: 'answer',
          answer: {
            type: answerSdp.type,
            sdp: answerSdp.sdp
          }
        }));
      }
    } catch (err) {
      console.error("Error handling offer:", err);
      ws.send(JSON.stringify({ type: "error", error: (err as Error).message }));
    }
  }

  function handleCandidate(candidate: { candidate: string; sdpMid: string | null; sdpMLineIndex: number | null }) {
    if (pc) {
      try {
        pc.addRemoteCandidate(candidate.candidate, candidate.sdpMid || undefined);
      } catch (err) {
        console.error('Error adding ICE candidate:', err);
      }
    }
  }

  ws.on('close', () => {
    console.log('Client disconnected:', clientId);
    if (pc) {
      pc.close();
      peers.delete(clientId);
    }
  });

  ws.on('error', (err: Error) => {
    console.error('WebSocket error:', err);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
