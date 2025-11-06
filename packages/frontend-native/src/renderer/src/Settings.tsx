import { useState, useEffect } from "react";

const Settings = (): JSX.Element => {
  const [recordOnLaunch, setRecordOnLaunch] = useState<boolean | null>(null);

  // Load initial setting
  useEffect(() => {
    const loadSettings = async () => {
      if (window.electronAPI?.getSetting) {
        const value = await window.electronAPI.getSetting('recordOnLaunch');
        setRecordOnLaunch(value !== undefined ? value : true);
      }
    };
    loadSettings();
  }, []);

  // Save setting when changed
  const handleToggleRecordOnLaunch = async () => {
    if (recordOnLaunch === null) return; // Don't allow toggle until loaded

    const newValue = !recordOnLaunch;
    setRecordOnLaunch(newValue);

    if (window.electronAPI?.setSetting) {
      await window.electronAPI.setSetting('recordOnLaunch', newValue);
    }
  };

  // Don't render until setting is loaded
  if (recordOnLaunch === null) {
    return (
      <div style={{
        display: "flex",
        flexDirection: "column",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}>
        <h2 style={{
          fontSize: "18px",
          fontWeight: "600",
          marginBottom: "16px",
          color: "#333"
        }}>
          Settings
        </h2>
        <div style={{ fontSize: "14px", color: "#666" }}>Loading...</div>
      </div>
    );
  }

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      fontFamily: "system-ui, -apple-system, sans-serif",
    }}>
      <h2 style={{
        fontSize: "18px",
        fontWeight: "600",
        marginBottom: "16px",
        color: "#333"
      }}>
        Settings
      </h2>

      <div style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px"
      }}>
        {/* Record on Launch Setting */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px",
          backgroundColor: "#f9f9f9",
          borderRadius: "6px",
          border: "1px solid #e0e0e0"
        }}>
          <div style={{ flex: 1 }}>
            <div style={{
              fontSize: "14px",
              fontWeight: "500",
              color: "#333",
              marginBottom: "4px"
            }}>
              Record on Launch
            </div>
            <div style={{
              fontSize: "12px",
              color: "#666"
            }}>
              Start recording microphone automatically when the app launches
            </div>
          </div>

          <label style={{
            position: "relative",
            display: "inline-block",
            width: "50px",
            height: "26px",
            marginLeft: "12px",
            cursor: "pointer"
          }}>
            <input
              type="checkbox"
              checked={recordOnLaunch}
              onChange={handleToggleRecordOnLaunch}
              style={{
                opacity: 0,
                width: 0,
                height: 0
              }}
            />
            <span style={{
              position: "absolute",
              cursor: "pointer",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: recordOnLaunch ? "#34C759" : "#ccc",
              transition: "0.3s",
              borderRadius: "26px"
            }}>
              <span style={{
                position: "absolute",
                content: "",
                height: "20px",
                width: "20px",
                left: recordOnLaunch ? "27px" : "3px",
                bottom: "3px",
                backgroundColor: "white",
                transition: "0.3s",
                borderRadius: "50%"
              }} />
            </span>
          </label>
        </div>
      </div>
    </div>
  );
};

export default Settings;

