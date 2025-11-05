const Settings = (): JSX.Element => {
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
      <p style={{
        fontSize: "14px",
        color: "#666",
        marginBottom: "12px"
      }}>
        Settings will be added here
      </p>
      {/* Placeholder for future settings */}
      <div style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px"
      }}>
        {/* Add settings options here in the future */}
      </div>
    </div>
  );
};

export default Settings;

