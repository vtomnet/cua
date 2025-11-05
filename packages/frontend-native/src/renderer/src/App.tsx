import { useEffect } from "react";
import Cursor from "./components/Cursor";
import ErrorMessage from "./components/ErrorMessage";
import { useRecorder } from "./record";
import { transcribe, disconnectTranscription } from "./transcribe";
import { runAgent, AgentCancelledError } from "./llm";

const App = (): JSX.Element => {
  const handleSpeechEnd = async (audioData: Float32Array) => {
    try {
      console.log(`Transcribing ${audioData.length} samples (${(audioData.length / 16000).toFixed(2)}s)`);

      const transcript = await transcribe(audioData);
      console.log("Transcript:", transcript);

      if (transcript.trim()) {
        try {
          await runAgent(transcript);
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
  const { status, error, isSpeaking } = useRecorder({
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
          await runAgent(text);
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
