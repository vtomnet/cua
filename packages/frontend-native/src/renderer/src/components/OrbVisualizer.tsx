import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

type OrbVisualizerProps = {
  analyser: AnalyserNode | null;
  isRecording: boolean;
};

type OrbProps = {
  analyser: AnalyserNode | null;
  isRecording: boolean;
};

const Orb = ({ analyser, isRecording }: OrbProps) => {
  const meshRef = useRef<
    THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>
  >(null);
  const analyserRef = useRef<AnalyserNode | null>(analyser);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const isRecordingRef = useRef(isRecording);
  const smoothedVolumeRef = useRef(0);
  const { gl } = useThree();
  const initialExposure = useMemo(() => gl.toneMappingExposure, [gl]);

  useEffect(() => {
    analyserRef.current = analyser;
    if (analyser) {
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
    } else {
      dataArrayRef.current = null;
      smoothedVolumeRef.current = 0;
    }
  }, [analyser]);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  useEffect(() => {
    return () => {
      gl.toneMappingExposure = initialExposure;
    };
  }, [gl, initialExposure]);

  useFrame((state) => {
    const mesh = meshRef.current;
    if (!mesh) {
      return;
    }

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
    const smoothed = THREE.MathUtils.lerp(
      smoothedVolumeRef.current,
      target,
      0.2,
    );
    smoothedVolumeRef.current = smoothed;

    const elapsed = state.clock.getElapsedTime();
    mesh.scale.setScalar(1 + smoothed * 1.5);
    mesh.position.y = Math.sin(elapsed * 1.5) * 0.05;
    mesh.rotation.y += 0.002 + smoothed * 0.01;

    const material = mesh.material;
    if (material instanceof THREE.MeshStandardMaterial) {
      material.emissiveIntensity = 0.5 + smoothed * 2.0;
      material.color.setHSL(0.55 + smoothed * 0.2, 1.0, 0.5);
    }

    const breathPulse = Math.sin(elapsed * 0.5) * 0.02;
    const bloomStrength = THREE.MathUtils.clamp(
      1.5 + smoothed * 1.5 + breathPulse,
      1.0,
      4.0,
    );

    gl.toneMappingExposure = THREE.MathUtils.lerp(
      gl.toneMappingExposure,
      0.9 + bloomStrength * 0.2,
      0.1,
    );
  });

  return (
    <mesh ref={meshRef} position={[0, 0, 0]}>
      <sphereGeometry args={[1, 128, 128]} />
      <meshStandardMaterial
        color="#00ffff"
        emissive="#0099ff"
        roughness={0.4}
        metalness={0.9}
        emissiveIntensity={0.5}
      />
    </mesh>
  );
};

const OrbVisualizer = ({ analyser, isRecording }: OrbVisualizerProps): JSX.Element => {
  const devicePixelRatio =
    typeof window !== "undefined" && window.devicePixelRatio
      ? Math.min(window.devicePixelRatio, 2)
      : 1;

  return (
    <div className="relative h-full w-full bg-transparent">
      <Canvas
        className="absolute inset-0"
        camera={{ position: [0, 0, 3], fov: 75, near: 0.1, far: 100 }}
        dpr={devicePixelRatio}
        gl={{ alpha: true, antialias: true }}
        onCreated={({ gl }) => {
          gl.setClearColor(new THREE.Color(0x000000), 0);
        }}
      >
        <ambientLight intensity={0.2} />
        <pointLight position={[2, 2, 2]} intensity={1.2} />
        <Orb analyser={analyser} isRecording={isRecording} />
      </Canvas>
      <div className="pointer-events-none absolute inset-x-0 bottom-4 text-center text-xs uppercase tracking-[0.35em] text-cyan-100/80">
        {isRecording ? "Listening" : "Visualizer Idle"}
      </div>
    </div>
  );
};

export default OrbVisualizer;
