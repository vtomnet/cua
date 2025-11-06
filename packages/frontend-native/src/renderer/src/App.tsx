import { useEffect, useState } from "react";
import Cursor from "./components/Cursor";
import ErrorMessage from "./components/ErrorMessage";
import { useRecorder } from "./record";
import { transcribe, disconnectTranscription } from "./transcribe";
import { runAgent, AgentCancelledError } from "./llm";

const App = (): JSX.Element => {
  const [isRecording, setIsRecording] = useState(false);

  // Load initial recording state from settings
  useEffect(() => {
    const loadInitialState = async () => {
      if (window.electronAPI?.getInitialRecordingState) {
        try {
          const shouldRecord = await window.electronAPI.getInitialRecordingState();
          setIsRecording(shouldRecord);

          // Send initial state to main process
          if (window.electronAPI?.sendRecordingState) {
            window.electronAPI.sendRecordingState(shouldRecord);
          }
        } catch (error) {
          console.error('Error loading initial recording state:', error);
        }
      }
    };

    loadInitialState();
  }, []);

  // Handle toggle recording from main process
  useEffect(() => {
    if (!window.electronAPI?.onToggleRecording) return;

    const cleanup = window.electronAPI.onToggleRecording(() => {
      setIsRecording(prev => {
        const newState = !prev;
        console.log(`Recording ${newState ? 'enabled' : 'disabled'}`);

        // Notify main process of the state change
        if (window.electronAPI?.sendRecordingState) {
          window.electronAPI.sendRecordingState(newState);
        }

        return newState;
      });
    });

    return cleanup;
  }, []);

  const handleSpeechEnd = async (audioData: Float32Array) => {
    try {
      console.log(`Transcribing ${audioData.length} samples (${(audioData.length / 16000).toFixed(2)}s)`);

      const transcript = await transcribe(audioData);
      console.log("Transcript:", transcript);

      if (transcript.trim()) {
        try {
          const appInfo = await window.electronAPI?.getCurrentApp() || { name: "Unknown" };
          const info = {
            date: new Date().toISOString(),
            currentApp: appInfo.name,
            ...(appInfo.url && { url: appInfo.url }),
            ...(appInfo.title && { title: appInfo.title })
          };
          console.log("Context info:", info);
          await runAgent(transcript, info);
        } catch (error) {
          if (error instanceof AgentCancelledError) {
            console.warn("Agent execution was cancelled by a new invocation");
          } else {
            throw error; // Re-throw other errors to outer catch
          }
        }
      }
    } catch (error) {
      console.error("Transcription failed:", error);
    }
  };

  const handleSpeechStart = () => {
    console.log("User started speaking");
  };

  const handleError = (error: Error) => {
    console.error("Recording error:", error);
  };

  // Initialize recorder with callbacks
  const { status, error, isSpeaking } = useRecorder(isRecording, {
    onSpeechEnd: handleSpeechEnd,
    onSpeechStart: handleSpeechStart,
    onError: handleError,
  });

  // Cleanup transcription connection on unmount
  useEffect(() => {
    return () => {
      disconnectTranscription();
    };
  }, []);

  // Handle text processing from control panel
  useEffect(() => {
    if (!window.electronAPI?.onProcessText) return;

    const cleanup = window.electronAPI.onProcessText(async (text: string) => {
      console.log("Processing text from control panel:", text);
      if (text.trim()) {
        try {
          const appInfo = await window.electronAPI?.getCurrentApp() || { name: "Unknown" };
          const info = {
            date: new Date().toISOString(),
            currentApp: appInfo.name,
            ...(appInfo.url && { url: appInfo.url }),
            ...(appInfo.title && { title: appInfo.title })
          };
          console.log("Context info:", info);
          await runAgent(text, info);
        } catch (error) {
          if (error instanceof AgentCancelledError) {
            console.warn("Agent execution was cancelled by a new invocation");
          } else {
            console.error("Agent execution failed:", error);
          }
        }
      }
    });

    return cleanup;
  }, []);

  return (
    <main>
      <Cursor status={status}/>
      <ErrorMessage error={error}/>
    </main>
  )
};

export default App;
