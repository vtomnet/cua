import { useEffect } from "react";
import "./app.css";
import Cursor from "./components/Cursor";
import ErrorMessage from "./components/ErrorMessage";
import { useRecorder } from "./record";
import { transcribe, disconnectTranscription } from "./transcribe";
import { runAgent } from "./llm";

const App = (): JSX.Element => {
  // Define callbacks for speech events
  const handleSpeechEnd = async (audioData: Float32Array) => {
    try {
      console.log(`Transcribing ${audioData.length} samples (${(audioData.length / 16000).toFixed(2)}s)`);

      const transcript = await transcribe(audioData);
      console.log("Transcript:", transcript);

      if (transcript.trim()) {
        await runAgent(transcript);
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

  return (
    <main>
      <Cursor status={status}/>
      <ErrorMessage error={error}/>
    </main>
  )
};

export default App;
