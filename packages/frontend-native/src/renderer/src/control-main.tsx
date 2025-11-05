import React from "react";
import ReactDOM from "react-dom/client";
import ControlPanel from "./ControlPanel";
import "./app.css";

ReactDOM.createRoot(document.getElementById("app") as HTMLElement).render(
  <React.StrictMode>
    <ControlPanel />
  </React.StrictMode>,
);

