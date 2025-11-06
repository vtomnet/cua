import { useState, useEffect } from "react";
import Settings from "./Settings";

const ControlPanel = (): JSX.Element => {
  const [inputText, setInputText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRecording, setIsRecording] = useState(true); // Will be updated based on settings
  const [showSettings, setShowSettings] = useState(false);

  // Load initial recording state from settings
  useEffect(() => {
    const loadInitialState = async () => {
      if (window.electronAPI?.getSetting) {
        try {
          const shouldRecord = await window.electronAPI.getSetting('recordOnLaunch');
          // Default to true if not set
          setIsRecording(shouldRecord !== false);
        } catch (error) {
          console.error('Error loading initial recording state:', error);
        }
      }
    };

    loadInitialState();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isSubmitting) return;

    setIsSubmitting(true);
    try {
      await window.electronAPI.submitText(inputText);
      setInputText("");
    } catch (error) {
      console.error("Failed to submit text:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleToggleRecording = () => {
    // Send toggle recording to the overlay window
    // The actual state will be updated via the recording-state-update listener
    window.electronAPI.toggleRecording();
  };

  const handleToggleSettings = async () => {
    const newShowSettings = !showSettings;
    setShowSettings(newShowSettings);

    try {
      // Resize window based on settings visibility
      await window.electronAPI.resizeControlWindow(newShowSettings);
    } catch (error) {
      console.error("Failed to resize window:", error);
    }
  };

  // Listen for recording state updates from main process
  useEffect(() => {
    if (!window.electronAPI?.onRecordingStateUpdate) return;

    const cleanup = window.electronAPI.onRecordingStateUpdate((recordingState: boolean) => {
      console.log("Control Panel received recording state update:", recordingState);
      setIsRecording(recordingState);
    });

    return cleanup;
  }, []);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      padding: "16px",
      gap: "12px",
      fontFamily: "system-ui, -apple-system, sans-serif",
      height: "100vh",
      boxSizing: "border-box"
    }}>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: "8px" }}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type your message..."
          disabled={isSubmitting}
          autoFocus
          style={{
            flex: 1,
            padding: "8px 12px",
            borderRadius: "6px",
            border: "1px solid #ccc",
            fontSize: "14px",
            outline: "none"
          }}
          onFocus={(e) => e.target.style.borderColor = "#007AFF"}
          onBlur={(e) => e.target.style.borderColor = "#ccc"}
        />
        <button
          type="submit"
          disabled={!inputText.trim() || isSubmitting}
          style={{
            padding: "8px 16px",
            borderRadius: "6px",
            border: "none",
            backgroundColor: inputText.trim() && !isSubmitting ? "#007AFF" : "#ccc",
            color: "white",
            fontSize: "14px",
            cursor: inputText.trim() && !isSubmitting ? "pointer" : "not-allowed",
            fontWeight: "500"
          }}
        >
          Send
        </button>
      </form>

      <div style={{ display: "flex", gap: "8px", justifyContent: "space-between" }}>
        <button
          onClick={handleToggleRecording}
          style={{
            flex: 1,
            padding: "12px",
            borderRadius: "6px",
            border: "none",
            backgroundColor: isRecording ? "#FF3B30" : "#34C759",
            color: "white",
            fontSize: "14px",
            cursor: "pointer",
            fontWeight: "500",
            transition: "opacity 0.2s"
          }}
          onMouseEnter={(e) => e.currentTarget.style.opacity = "0.8"}
          onMouseLeave={(e) => e.currentTarget.style.opacity = "1"}
        >
          {isRecording ? "⏹ Stop Recording" : "🎤 Start Recording"}
        </button>

        <button
          onClick={handleToggleSettings}
          style={{
            padding: "12px 20px",
            borderRadius: "6px",
            border: "1px solid #ccc",
            backgroundColor: showSettings ? "#f5f5f5" : "white",
            fontSize: "14px",
            cursor: "pointer",
            fontWeight: "500",
            transition: "background-color 0.2s"
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = "#f5f5f5"}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = showSettings ? "#f5f5f5" : "white"}
        >
          {showSettings ? "⚙️ Close Settings" : "⚙️ Settings"}
        </button>
      </div>

      {!showSettings && (
        <div style={{
          fontSize: "12px",
          color: "#666",
          textAlign: "center",
          marginTop: "auto"
        }}>
          {isRecording ? "Recording active" : "Ready"}
        </div>
      )}

      {showSettings && (
        <div style={{
          flex: 1,
          overflow: "auto",
          marginTop: "8px",
          borderTop: "1px solid #e0e0e0",
          paddingTop: "12px"
        }}>
          <Settings />
        </div>
      )}
    </div>
  );
};

export default ControlPanel;

