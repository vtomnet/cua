import { useEffect, useMemo, useRef, useState } from "react";
import "./SiriEdgeVisualizer.css";

type SiriEdgeVisualizerProps = {
  analyser: AnalyserNode | null;
  isRecording: boolean;
};

const SiriEdgeVisualizer = ({ analyser, isRecording }: SiriEdgeVisualizerProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(analyser);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const isRecordingRef = useRef(isRecording);
  const animationFrameRef = useRef<number | null>(null);
  const [audioLevel, setAudioLevel] = useState(0);

  useEffect(() => {
    analyserRef.current = analyser;
    if (analyser) {
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
    } else {
      dataArrayRef.current = null;
    }
  }, [analyser]);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  // Audio analysis loop
  useEffect(() => {
    const updateAudioLevel = () => {
      const analyserNode = analyserRef.current;
      const dataArray = dataArrayRef.current;

      if (analyserNode && dataArray && isRecordingRef.current) {
        analyserNode.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const average = sum / dataArray.length / 256;
        setAudioLevel(average);
      } else {
        setAudioLevel(0);
      }

      animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
    };

    if (isRecording) {
      updateAudioLevel();
    } else {
      setAudioLevel(0);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isRecording]);

  // Calculate wave positions based on audio level
  const waveStyle = useMemo(() => {
    const scale = 1 + audioLevel * 0.5;
    const intensity = audioLevel * 100;
    return {
      transform: `scale(${scale})`,
      filter: `brightness(${1 + audioLevel * 0.3})`,
      opacity: 0.7 + audioLevel * 0.3,
      '--wave-intensity': `${intensity}%`,
    } as React.CSSProperties & { '--wave-intensity': string };
  }, [audioLevel]);

  const isActive = isRecording || audioLevel > 0.01;

  return (
    <div
      ref={containerRef}
      className={`siri-edge-container ${isActive ? 'active' : ''}`}
    >
      {/* Base glow layer */}
      <div className="siri-glow-base" />

      {/* Multiple blur layers for depth */}
      <div className="siri-glow-layer blur-8" />
      <div className="siri-glow-layer blur-12" />
      <div className="siri-glow-layer blur-16" />
      <div className="siri-glow-layer blur-20" />

      {/* Specular highlight */}
      <div className="siri-specular" />

      {/* Audio-reactive wave bulges */}
      <div className="siri-wave-container" style={waveStyle}>
        <div className="siri-wave wave-1" />
        <div className="siri-wave wave-2" />
        <div className="siri-wave wave-3" />
      </div>

      {/* Status indicator */}
      <div className="siri-status">
        {isRecording ? "Listening..." : ""}
      </div>
    </div>
  );
};

export default SiriEdgeVisualizer;