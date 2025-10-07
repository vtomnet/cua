import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

type OrbVisualizerProps = {
  analyser: AnalyserNode | null;
  isRecording: boolean;
};

const OrbVisualizer = ({ analyser, isRecording }: OrbVisualizerProps): JSX.Element => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const orbRef = useRef<
    THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial> | null
  >(null);
  const animationFrameRef = useRef<number>();
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const clockRef = useRef(new THREE.Clock());
  const smoothedVolumeRef = useRef(0);
  const isRecordingRef = useRef(isRecording);

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
    const mount = mountRef.current;
    if (!mount) {
      return;
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.setClearColor(0x000000, 0);
    renderer.setClearAlpha(0);
    rendererRef.current = renderer;
    renderer.domElement.style.backgroundColor = "transparent";
    renderer.domElement.style.background = "transparent";
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = null;
    const camera = new THREE.PerspectiveCamera(
      75,
      mount.clientWidth / mount.clientHeight,
      0.1,
      100,
    );
    camera.position.z = 3;

    const geometry = new THREE.SphereGeometry(1, 128, 128);
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x00ffff),
      emissive: new THREE.Color(0x0099ff),
      roughness: 0.4,
      metalness: 0.9,
      emissiveIntensity: 0.5,
    });
    const orb = new THREE.Mesh(geometry, material);
    scene.add(orb);
    orbRef.current = orb;

    const pointLight = new THREE.PointLight(0xffffff, 1.2);
    pointLight.position.set(2, 2, 2);
    scene.add(pointLight);

    const ambient = new THREE.AmbientLight(0x202020);
    scene.add(ambient);

    const composer = new EffectComposer(renderer);
    composer.renderTarget1.texture.format = THREE.RGBAFormat;
    composer.renderTarget2.texture.format = THREE.RGBAFormat;

    const renderPass = new RenderPass(scene, camera);
    renderPass.clear = true;
    renderPass.clearAlpha = 0;

    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(mount.clientWidth, mount.clientHeight),
      1.5,
      0.4,
      0.85,
    );

    composer.addPass(renderPass);
    composer.addPass(bloomPass);

    const blendMaterial = bloomPass.blendMaterial;
    blendMaterial.blending = THREE.CustomBlending;
    blendMaterial.blendSrc = THREE.OneFactor;
    blendMaterial.blendDst = THREE.OneFactor;
    blendMaterial.blendEquation = THREE.AddEquation;
    blendMaterial.blendSrcAlpha = THREE.ZeroFactor;
    blendMaterial.blendDstAlpha = THREE.OneFactor;
    blendMaterial.blendEquationAlpha = THREE.AddEquation;

    const transparentOutputPass = new ShaderPass(
      new THREE.ShaderMaterial({
        uniforms: {
          baseTexture: { value: null },
        },
        vertexShader: `
          varying vec2 vUv;
          void main() {
            vUv = uv;
            gl_Position = vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform sampler2D baseTexture;
          varying vec2 vUv;
          void main() {
            vec4 color = texture2D(baseTexture, vUv);
            float intensity = max(max(color.r, color.g), color.b);
            float alpha = step(0.02, intensity);
            gl_FragColor = vec4(color.rgb, alpha);
          }
        `,
        transparent: true,
        depthWrite: false,
      }),
      "baseTexture",
    );

    composer.addPass(transparentOutputPass);
    composerRef.current = composer;

    clockRef.current.start();

    const handleResize = () => {
      const width = mount.clientWidth;
      const height = mount.clientHeight;
      if (height === 0 || width === 0) {
        return;
      }
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      composer.setSize(width, height);
      bloomPass.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);

    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      const orbMesh = orbRef.current;
      const currentComposer = composerRef.current;
      if (!orbMesh || !currentComposer) {
        return;
      }

      const delta = clockRef.current.getDelta();
      const elapsed = clockRef.current.getElapsedTime();

      let volume = 0;
      const analyserNode = analyserRef.current;
      const dataArray = dataArrayRef.current;
      if (analyserNode && dataArray) {
        analyserNode.getByteFrequencyData(dataArray);
        const sum = dataArray.reduce((acc, value) => acc + value, 0);
        volume = sum / dataArray.length / 256;
      }

      const target = isRecordingRef.current ? volume : volume * 0.4;
      const smoothed = THREE.MathUtils.lerp(
        smoothedVolumeRef.current,
        target,
        0.2,
      );
      smoothedVolumeRef.current = smoothed;

      const materialRef = orbMesh.material;
      const scale = 1 + smoothed * 1.5;
      orbMesh.scale.setScalar(scale);
      orbMesh.position.y = Math.sin(elapsed * 1.5) * 0.05;
      orbMesh.rotation.y += 0.002 + smoothed * 0.01;

      materialRef.emissiveIntensity = 0.5 + smoothed * 2.0;
      materialRef.color.setHSL(0.55 + smoothed * 0.2, 1.0, 0.5);

      const breathPulse = Math.sin(elapsed * 0.5) * 0.02;
      const bloomStrength = 1.5 + smoothed * 1.5 + breathPulse;
      bloomPass.strength = THREE.MathUtils.clamp(bloomStrength, 1.0, 4.0);

      currentComposer.render(delta);
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      window.removeEventListener("resize", handleResize);

      transparentOutputPass.dispose();
      bloomPass.dispose();
      composer.dispose();
      renderer.dispose();

      geometry.dispose();
      material.dispose();

      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }

      rendererRef.current = null;
      composerRef.current = null;
      orbRef.current = null;
    };
  }, []);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  return (
    <div className="relative h-full w-full bg-transparent">
      <div ref={mountRef} className="absolute inset-0 bg-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-4 text-center text-xs uppercase tracking-[0.35em] text-cyan-100/80">
        {isRecording ? "Listening" : "Visualizer Idle"}
      </div>
    </div>
  );
};

export default OrbVisualizer;
