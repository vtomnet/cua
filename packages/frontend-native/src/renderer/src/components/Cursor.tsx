import { useEffect, useRef } from "react";

type CursorProps = {
  analyser: AnalyserNode | null;
  status: "idle" | "loading" | "recording" | "error";
  currentResponse: string | null;
};

const Cursor = ({ analyser, status, currentResponse }: CursorProps): JSX.Element => {
  const analyserRef = useRef<AnalyserNode | null>(analyser);
  const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const isRecordingRef = useRef(status === "recording");
  const smoothedVolumeRef = useRef(0);
  const animationFrameRef = useRef<number>();
  const glowRef = useRef<HTMLDivElement>(null);

  const getGlowColor = () => {
    switch (status) {
      case "loading":
        return "bg-yellow-400";
      case "recording":
        return "bg-blue-400";
      case "error":
        return "bg-red-600";
      case "idle":
      default:
        return "bg-gray-400";
    }
  };

  useEffect(() => {
    analyserRef.current = analyser;
    if (analyser) {
      dataArrayRef.current = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
    } else {
      dataArrayRef.current = null;
      smoothedVolumeRef.current = 0;
    }
  }, [analyser]);

  useEffect(() => {
    isRecordingRef.current = status === "recording";
  }, [status]);

  useEffect(() => {
    const animate = () => {
      const analyserNode = analyserRef.current;
      const dataArray = dataArrayRef.current;
      let volume = 0;

      if (analyserNode && dataArray) {
        analyserNode.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i += 1) {
          sum += dataArray[i];
        }
        volume = sum / dataArray.length / 256;
      }

      const target = isRecordingRef.current ? volume : volume * 0.4;
      const smoothed = smoothedVolumeRef.current + (target - smoothedVolumeRef.current) * 0.2;
      smoothedVolumeRef.current = smoothed;

      const elapsed = Date.now() * 0.001;
      let totalIntensity = 0.3;

      // Different animation patterns based on status
      switch (status) {
        case "loading":
          // Faster pulsing for loading
          totalIntensity = 0.4 + Math.sin(elapsed * 2) * 0.4;
          break;
        case "error":
          // Intense, urgent pulsing for errors
          totalIntensity = 0.6 + Math.sin(elapsed * 3) * 0.3;
          break;
        case "recording":
          // Normal speech-reactive animation when recording
          const speechPulse = smoothed * 2.0;
          totalIntensity = 0.5 + speechPulse;
          break;
        case "idle":
        default:
          // Gentle breathing pattern when idle
          const breathPulse = Math.sin(elapsed * 0.8) * 0.3;
          totalIntensity = 0.3 + breathPulse;
          break;
      }

      if (glowRef.current) {
        const blurAmount = 8 + totalIntensity * 12;
        const opacity = Math.max(0.2, Math.min(0.8, totalIntensity));
        glowRef.current.style.filter = `blur(${blurAmount}px)`;
        glowRef.current.style.opacity = opacity.toString();
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [status]);

  return (
    <div className="relative h-full w-full bg-transparent flex items-center justify-center">
      {/* Status glow background */}
      <div
        ref={glowRef}
        className={`absolute w-16 h-16 ${getGlowColor()} rounded-full`}
        style={{
          filter: 'blur(12px)',
          opacity: 0.4,
        }}
      />

      {/* Cursor SVG */}
      <div className="relative z-10">
        <svg
          width="48px"
          height="48px"
          viewBox="0 0 32 32"
          version="1.1"
          xmlns="http://www.w3.org/2000/svg"
          xmlnsXlink="http://www.w3.org/1999/xlink"
          style={{ transform: 'scaleX(-1)' }}
        >
          <title>Cursor</title>
          <g stroke="none" fill="none" fillRule="evenodd">
            <path
              d="M16.501,13.8601001 L24.884,22.2611001 C25.937,23.3171001 25.19,25.1191001 23.699,25.1191001 L22.475,25.119 L23.6908,28.0067001 C23.9038,28.5127001 23.9068,29.0727001 23.6998,29.5817001 C23.4918,30.0917001 23.0978,30.4897001 22.5898,30.7027001 C22.3338,30.8097001 22.0658,30.8637001 21.7918,30.8637001 C20.9608,30.8637001 20.2158,30.3687001 19.8938,29.6027001 L18.616,26.565 L17.784,27.3031001 C16.703,28.2591001 15,27.4921001 15,26.0481001 L15,14.4811001 C15,13.6971001 15.947,13.3051001 16.501,13.8601001 Z"
              fill="#FFFFFF"
            />
            <path
              d="M15.9995,15.1292001 C15.9995,14.9982001 16.1585,14.9322001 16.2505,15.0252001 L24.1585,22.9502001 C24.5895,23.3822001 24.2835,24.1192001 23.6735,24.1192001 L20.9695,24.1176804 L22.7691,28.3936001 C22.9961,28.9336001 22.7421,29.5546001 22.2031,29.7806001 C21.6621,30.0076001 21.0421,29.7546001 20.8161,29.2156001 L18.9985,24.8916804 L17.1385,26.5392001 C16.7225,26.9072001 16.0806176,26.6507019 16.0065415,26.1273654 L15.9995,26.0262001 Z"
              fill="#000000"
            />
          </g>
        </svg>

        {/* Floating response message - positioned at southeast corner of cursor */}
        {currentResponse && (
          <div className="absolute top-8 -right-2 max-w-sm max-h-32 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg border border-gray-200 p-3 text-sm text-gray-800 pointer-events-none animate-in fade-in duration-300 overflow-y-auto">
            <div className="whitespace-pre-line break-words">
              {currentResponse}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Cursor;
