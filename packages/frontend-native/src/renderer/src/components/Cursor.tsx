import { useEffect, useRef, useState } from "react";
import CursorSVG from "../../../assets/cursor.svg?react";

// type CursorProps = {
//   analyser: AnalyserNode | null;
//   status: "idle" | "loading" | "recording" | "error";
//   currentResponse: string | null;
//   position: { x: number; y: number } | null;
// };

type CursorProps = {
  status: "idle" | "loading" | "recording" | "error";
}

// const Cursor = ({ analyser, status, currentResponse, position }: CursorProps): JSX.Element => {
const Cursor = ({ status }: CursorProps): JSX.Element => {
  // const analyserRef = useRef<AnalyserNode | null>(analyser);
  // const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  // const isRecordingRef = useRef(status === "recording");
  // const smoothedVolumeRef = useRef(0);
  // const animationFrameRef = useRef<number>();
  // const glowRef = useRef<HTMLDivElement>(null);

  // // Cursor position state for smooth animation
  // const [animatedPosition, setAnimatedPosition] = useState<{ x: number; y: number } | null>(null);
  // const targetPositionRef = useRef<{ x: number; y: number } | null>(null);
  // const animatedPositionRef = useRef<{ x: number; y: number } | null>(null);

  // const getGlowColor = () => {
  //   switch (status) {
  //     case "loading":
  //       return "bg-yellow-400";
  //     case "recording":
  //       return "bg-blue-400";
  //     case "error":
  //       return "bg-red-600";
  //     case "idle":
  //     default:
  //       return "bg-gray-400";
  //   }
  // };

  // useEffect(() => {
  //   analyserRef.current = analyser;
  //   if (analyser) {
  //     dataArrayRef.current = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
  //   } else {
  //     dataArrayRef.current = null;
  //     smoothedVolumeRef.current = 0;
  //   }
  // }, [analyser]);

  // useEffect(() => {
  //   isRecordingRef.current = status === "recording";
  // }, [status]);

  // // Update target position when position prop changes
  // useEffect(() => {
  //   if (position) {
  //     targetPositionRef.current = position;

  //     // If this is the first position, set it immediately without animation
  //     if (!animatedPositionRef.current) {
  //       animatedPositionRef.current = { ...position };
  //       setAnimatedPosition({ ...position });
  //     }
  //   }
  // }, [position]);

  // useEffect(() => {
  //   const animate = () => {
  //     const analyserNode = analyserRef.current;
  //     const dataArray = dataArrayRef.current;
  //     let volume = 0;

  //     // Animate cursor position
  //     const targetPos = targetPositionRef.current;
  //     const currentPos = animatedPositionRef.current;

  //     if (targetPos && currentPos) {
  //       const lerp = 0.15; // Smoothing factor (0 = no movement, 1 = instant)
  //       const newX = currentPos.x + (targetPos.x - currentPos.x) * lerp;
  //       const newY = currentPos.y + (targetPos.y - currentPos.y) * lerp;

  //       // Update position if there's meaningful movement
  //       const distanceThreshold = 0.1;
  //       if (Math.abs(newX - currentPos.x) > distanceThreshold || Math.abs(newY - currentPos.y) > distanceThreshold) {
  //         animatedPositionRef.current = { x: newX, y: newY };
  //         setAnimatedPosition({ x: newX, y: newY });
  //       }
  //     }

  //     if (analyserNode && dataArray) {
  //       analyserNode.getByteFrequencyData(dataArray);
  //       let sum = 0;
  //       for (let i = 0; i < dataArray.length; i += 1) {
  //         sum += dataArray[i];
  //       }
  //       volume = sum / dataArray.length / 256;
  //     }

  //     const target = isRecordingRef.current ? volume : volume * 0.4;
  //     const smoothed = smoothedVolumeRef.current + (target - smoothedVolumeRef.current) * 0.2;
  //     smoothedVolumeRef.current = smoothed;

  //     const elapsed = Date.now() * 0.001;
  //     let totalIntensity = 0.3;

  //     // Different animation patterns based on status
  //     switch (status) {
  //       case "loading":
  //         // Faster pulsing for loading
  //         totalIntensity = 0.4 + Math.sin(elapsed * 2) * 0.4;
  //         break;
  //       case "error":
  //         // Intense, urgent pulsing for errors
  //         totalIntensity = 0.6 + Math.sin(elapsed * 3) * 0.3;
  //         break;
  //       case "recording":
  //         // Normal speech-reactive animation when recording
  //         const speechPulse = smoothed * 2.0;
  //         totalIntensity = 0.5 + speechPulse;
  //         break;
  //       case "idle":
  //       default:
  //         // Static appearance when idle - no pulse
  //         totalIntensity = 0.3;
  //         break;
  //     }

  //     if (glowRef.current) {
  //       const blurAmount = 8 + totalIntensity * 12;
  //       const opacity = Math.max(0.2, Math.min(0.8, totalIntensity));
  //       glowRef.current.style.filter = `blur(${blurAmount}px)`;
  //       glowRef.current.style.opacity = opacity.toString();
  //     }

  //     animationFrameRef.current = requestAnimationFrame(animate);
  //   };

  //   animate();

  //   return () => {
  //     if (animationFrameRef.current) {
  //       cancelAnimationFrame(animationFrameRef.current);
  //     }
  //   };
  // }, [status]);

  // // Calculate cursor position styles
  // const cursorStyle = animatedPosition
  //   ? {
  //       position: 'absolute' as const,
  //       left: `${animatedPosition.x}px`,
  //       top: `${animatedPosition.y}px`,
  //       transform: 'translate(-24px, -24px)', // Center the cursor (48px / 2 = 24px)
  //       transition: 'none', // Disable CSS transitions since we're handling animation manually
  //     }
  //   : {
  //       position: 'relative' as const,
  //       transform: 'none',
  //     };

  // return (
  //   <div className="relative h-full w-full bg-transparent flex items-center justify-center">
  //     {/* Cursor container - either positioned absolutely or centered */}
  //     <div style={cursorStyle}>
  //       {/* Status glow background */}
  //       <div
  //         ref={glowRef}
  //         className={`absolute w-16 h-16 ${getGlowColor()} rounded-full`}
  //         style={{
  //           filter: 'blur(12px)',
  //           opacity: 0.4,
  //           transform: 'translate(-4.5px, 3px)',
  //         }}
  //       />

  //       {/* Cursor SVG */}
  //       <div className="relative z-10">
  //         <CursorSVG width={48} height={48} style={{ transform: 'scaleX(-1)' }} />

  //         {/* Floating response message - positioned at southeast corner of cursor */}
  //         {currentResponse && (
  //           <div className="absolute top-8 -right-2 max-w-sm max-h-32 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg border border-gray-200 p-3 text-sm text-gray-800 pointer-events-none animate-in fade-in duration-300 overflow-y-auto">
  //             <div className="whitespace-pre-line break-words">
  //               {currentResponse}
  //             </div>
  //           </div>
  //         )}
  //       </div>
  //     </div>
  //   </div>
  // );

  return (
    <div>Cursor</div>
  )
};

export default Cursor;
