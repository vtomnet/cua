import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web';
import cvReadyPromise from "@techstark/opencv-js";

// Disable the proxy worker because bundlers rewrite the worker entry to our app bundle,
// which crashes inside the worker when it touches DOM globals.
ort.env.wasm.proxy = false;
ort.env.wasm.numThreads = 1;

let cv = null;

// Camera calibration constants from c.py
const CAMERA_MATRIX = [
    [1.49454593e3, 0.0, 9.55794289e2],
    [0.0, 1.49048883e3, 5.18040731e2],
    [0.0, 0.0, 1.0]
];

const CAMERA_DISTORTION = [0.08599595, -0.37972518, -0.0059906, -0.00468435, 0.45227431];

// Eye tracking constants
const EYE_ROI_SIZE = [60, 36];
const FOCAL_LENGTH_NORM = 960;
const DISTANCE_NORM = 600;
const PREPROCESS_WIDTH = 36;
const PREPROCESS_HEIGHT = 60;

// 3D face coordinates from c.py
const GENERIC_3D_FACE_COORDINATES = [
    [-45.0967681126441, -21.3128582097374, 21.3128582097374, 45.0967681126441, -26.2995769055718, 26.2995769055718],
    [-0.483773045049757, 0.483773045049757, 0.483773045049757, -0.483773045049757, 68.5950352778326, 68.5950352778326],
    [2.39702984214363, -2.39702984214363, -2.39702984214363, 2.39702984214363, -9.86076131526265e-32, -9.86076131526265e-32]
];

class MixedNormalizer {
    constructor() {
        this.focalLengthNorm = FOCAL_LENGTH_NORM;
        this.distanceNorm = DISTANCE_NORM;
        this.eyeRoiSize = EYE_ROI_SIZE;
        this.cameraMatrix = CAMERA_MATRIX;
        this.cameraDistortion = CAMERA_DISTORTION;
        this.generic3dFaceCoordinates = GENERIC_3D_FACE_COORDINATES;
        this.faceLandmarker = null;
        this.isInitialized = false;
    }

    async initialize() {
        try {
            cv = await cvReadyPromise;

            // Wait for OpenCV.js to be ready
            if (!cv || !cv.solvePnP) {
                throw new Error("OpenCV.js not loaded or solvePnP not available.");
            }

            // Initialize MediaPipe
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            );

            this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "/face_landmarker.task" // Place the model in public folder
                },
                outputFaceBlendshapes: false,
                outputFacialTransformationMatrixes: false,
                numFaces: 1,
                minFaceDetectionConfidence: 0.5,
                minFacePresenceConfidence: 0.5,
                minTrackingConfidence: 0.5,
                runningMode: "VIDEO"
            });

            this.isInitialized = true;
            return true;
        } catch (error) {
            throw new Error(`Failed to initialize MediaPipe or OpenCV.js: ${error.message}`);
        }
    }

    // Matrix operations (keep manual for small matrices)
    matrixMultiply(a, b) {
        const rows = a.length;
        const cols = b[0].length;
        const result = Array(rows).fill().map(() => Array(cols).fill(0));

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                for (let k = 0; k < a[0].length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    estimateHeadPose(face2d) {
        const objectPointsArray = this.generic3dFaceCoordinates[0].map((_, i) => [
            this.generic3dFaceCoordinates[0][i],
            this.generic3dFaceCoordinates[1][i],
            this.generic3dFaceCoordinates[2][i]
        ]);

        let objMat = cv.matFromArray(6, 1, cv.CV_32FC3, objectPointsArray.flat());
        let imgMat = cv.matFromArray(6, 1, cv.CV_32FC2, face2d.flat());
        let camMat = cv.matFromArray(3, 3, cv.CV_32FC1, this.cameraMatrix.flat());
        let distMat = cv.matFromArray(1, 5, cv.CV_32FC1, this.cameraDistortion);
        let rvec = new cv.Mat(3, 1, cv.CV_32FC1);
        let tvec = new cv.Mat(3, 1, cv.CV_32FC1);

        // First pass with EPNP
        cv.solvePnP(objMat, imgMat, camMat, distMat, rvec, tvec, false, cv.SOLVEPNP_EPNP);

        // Refine with iterative
        cv.solvePnP(objMat, imgMat, camMat, distMat, rvec, tvec, true, cv.SOLVEPNP_ITERATIVE);

        const rotVec = [rvec.data32F[0], rvec.data32F[1], rvec.data32F[2]];
        const transVec = [tvec.data32F[0], tvec.data32F[1], tvec.data32F[2]];

        // Clean up
        objMat.delete();
        imgMat.delete();
        camMat.delete();
        distMat.delete();
        rvec.delete();
        tvec.delete();

        return { rotVec, transVec };
    }

    // Rodrigues using OpenCV.js
    rodrigues(rotVec) {
        let rvecMat = cv.matFromArray(3, 1, cv.CV_32FC1, rotVec);
        let rotMat = new cv.Mat();
        cv.Rodrigues(rvecMat, rotMat);

        const R = [
            [rotMat.data32F[0], rotMat.data32F[1], rotMat.data32F[2]],
            [rotMat.data32F[3], rotMat.data32F[4], rotMat.data32F[5]],
            [rotMat.data32F[6], rotMat.data32F[7], rotMat.data32F[8]]
        ];

        rvecMat.delete();
        rotMat.delete();

        return R;
    }

    retrieveEyes(rotVec, transVec) {
        const headRotation = this.rodrigues(rotVec);
        const headTranslation = [[transVec[0]], [transVec[1]], [transVec[2]]];

        // Transform 3D face coordinates
        const faceLandmarks3d = this.matrixMultiply(headRotation, this.generic3dFaceCoordinates);

        // Add translation
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < faceLandmarks3d[i].length; j++) {
                faceLandmarks3d[i][j] += headTranslation[i][0];
            }
        }

        // Calculate eye centers
        const rightEye = [
            0.5 * (faceLandmarks3d[0][0] + faceLandmarks3d[0][1]),
            0.5 * (faceLandmarks3d[1][0] + faceLandmarks3d[1][1]),
            0.5 * (faceLandmarks3d[2][0] + faceLandmarks3d[2][1])
        ];

        const leftEye = [
            0.5 * (faceLandmarks3d[0][2] + faceLandmarks3d[0][3]),
            0.5 * (faceLandmarks3d[1][2] + faceLandmarks3d[1][3]),
            0.5 * (faceLandmarks3d[2][2] + faceLandmarks3d[2][3])
        ];

        return { eyes: [rightEye, leftEye], headRotation };
    }

    normalizeEye(eye, headRotation, tempCanvas) {
        const distance = Math.sqrt(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]);
        const zScale = this.distanceNorm / distance;

        const cameraNorm = [
            [this.focalLengthNorm, 0, this.eyeRoiSize[0] / 2],
            [0, this.focalLengthNorm, this.eyeRoiSize[1] / 2],
            [0, 0, 1]
        ];

        const scalingMatrix = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, zScale]
        ];

        const forward = eye.map(coord => coord / distance);

        // Cross product: forward × headRotation[:,0] (headRotation[0] is first column)
        const down = [
            forward[1] * headRotation[2][0] - forward[2] * headRotation[1][0],
            forward[2] * headRotation[0][0] - forward[0] * headRotation[2][0],
            forward[0] * headRotation[1][0] - forward[1] * headRotation[0][0]
        ];
        const downNorm = Math.sqrt(down.reduce((sum, val) => sum + val * val, 0));
        down.forEach((val, i) => down[i] /= downNorm);

        // Cross product: down × forward
        const right = [
            down[1] * forward[2] - down[2] * forward[1],
            down[2] * forward[0] - down[0] * forward[2],
            down[0] * forward[1] - down[1] * forward[0]
        ];
        const rightNorm = Math.sqrt(right.reduce((sum, val) => sum + val * val, 0));
        right.forEach((val, i) => right[i] /= rightNorm);

        const rotationMatrix = [right, down, forward];

        // Use OpenCV.js for inverse
        let camMat = cv.matFromArray(3, 3, cv.CV_64FC1, this.cameraMatrix.flat());
        let invMat = new cv.Mat();
        cv.invert(camMat, invMat, cv.DECOMP_LU);
        const cameraInv = [
            [invMat.data64F[0], invMat.data64F[1], invMat.data64F[2]],
            [invMat.data64F[3], invMat.data64F[4], invMat.data64F[5]],
            [invMat.data64F[6], invMat.data64F[7], invMat.data64F[8]]
        ];
        invMat.delete();
        camMat.delete();

        // Transformation matrix
        const temp1 = this.matrixMultiply(rotationMatrix, cameraInv);
        const temp2 = this.matrixMultiply(scalingMatrix, temp1);
        const transformationMatrix = this.matrixMultiply(cameraNorm, temp2);

        // Use OpenCV.js for warpPerspective on grayscale
        let srcMat = cv.imread(tempCanvas);
        let grayMat = new cv.Mat();
        cv.cvtColor(srcMat, grayMat, cv.COLOR_RGBA2GRAY);
        let M = cv.matFromArray(3, 3, cv.CV_64FC1, transformationMatrix.flat());
        let dsize = new cv.Size(this.eyeRoiSize[0], this.eyeRoiSize[1]);
        let warpedMat = new cv.Mat();
        cv.warpPerspective(grayMat, warpedMat, M, dsize, cv.INTER_LINEAR);

        // Histogram equalization
        cv.equalizeHist(warpedMat, warpedMat);

        // Output to canvas (convert back to RGBA for consistency)
        let outputCanvas = document.createElement('canvas');
        outputCanvas.width = this.eyeRoiSize[0];
        outputCanvas.height = this.eyeRoiSize[1];
        cv.imshow(outputCanvas, warpedMat);

        // Clean up
        srcMat.delete();
        grayMat.delete();
        M.delete();
        warpedMat.delete();

        return outputCanvas;
    }

    getFaceLandmarks(canvas) {
        if (!this.isInitialized || !this.faceLandmarker) {
            throw new Error("MediaPipe not initialized");
        }

        const results = this.faceLandmarker.detectForVideo(canvas, performance.now());

        if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
            throw new Error("No face landmarks detected");
        }

        const landmarks = results.faceLandmarks[0];
        const width = canvas.width;
        const height = canvas.height;

        // Convert normalized coordinates to pixel coordinates
        return landmarks.map(landmark => [
            landmark.x * width,
            landmark.y * height
        ]);
    }

    normalizeFrame(canvas) {
        // Create temp canvas for undistortion
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0);

        // Undistort using OpenCV.js
        let src = cv.imread(tempCanvas);
        let dst = new cv.Mat();
        let camMat = cv.matFromArray(3, 3, cv.CV_64FC1, this.cameraMatrix.flat());
        let distMat = cv.matFromArray(1, 5, cv.CV_64FC1, this.cameraDistortion);
        cv.undistort(src, dst, camMat, distMat);
        cv.imshow(tempCanvas, dst);

        src.delete();
        dst.delete();
        camMat.delete();
        distMat.delete();

        const landmarks = this.getFaceLandmarks(tempCanvas);

        // Map MediaPipe landmarks to key facial points (from c.py)
        const face2d = [
            landmarks[33],   // Right eye inner corner
            landmarks[133],  // Right eye outer corner
            landmarks[362],  // Left eye inner corner
            landmarks[263],  // Left eye outer corner
            landmarks[61],   // Right mouth corner
            landmarks[291],  // Left mouth corner
        ];

        const { rotVec, transVec } = this.estimateHeadPose(face2d);
        const { eyes, headRotation } = this.retrieveEyes(rotVec, transVec);

        const rightEyeNormalized = this.normalizeEye(eyes[0], headRotation, tempCanvas);
        const leftEyeNormalized = this.normalizeEye(eyes[1], headRotation, tempCanvas);

        return [rightEyeNormalized, leftEyeNormalized];
    }
}

class RageNetPredictor {
    constructor() {
        this.session = null;
        this.isInitialized = false;
        this.inputNames = null;
        this.outputName = null;
        this.preprocessCanvas = document.createElement('canvas');
        this.preprocessCanvas.width = PREPROCESS_WIDTH;
        this.preprocessCanvas.height = PREPROCESS_HEIGHT;
        this.preprocessCtx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true });
        this.rightBuffer = new Float32Array(PREPROCESS_WIDTH * PREPROCESS_HEIGHT);
        this.leftBuffer = new Float32Array(PREPROCESS_WIDTH * PREPROCESS_HEIGHT);
        this.rightTensor = new ort.Tensor('float32', this.rightBuffer, [1, PREPROCESS_HEIGHT, PREPROCESS_WIDTH, 1]);
        this.leftTensor = new ort.Tensor('float32', this.leftBuffer, [1, PREPROCESS_HEIGHT, PREPROCESS_WIDTH, 1]);
    }

    async initialize() {
        try {
            this.session = await ort.InferenceSession.create('https://x.vtom.net/rn_sw_attention.onnx', {
                executionProviders: ["webgpu", "wasm"],
            });

            this.inputNames = this.session.inputNames;
            this.outputName = this.session.outputNames[0];

            console.log("ONNX model inputs:", this.inputNames);
            console.log("ONNX model outputs:", this.session.outputNames);

            this.isInitialized = true;
            return true;
        } catch (error) {
            throw new Error(`Failed to initialize ONNX model: ${error.message}`);
        }
    }

    prepareEye(sourceCanvas, targetBuffer) {
        this.preprocessCtx.drawImage(
            sourceCanvas,
            0,
            0,
            PREPROCESS_WIDTH,
            PREPROCESS_HEIGHT
        );

        const data = this.preprocessCtx
            .getImageData(0, 0, PREPROCESS_WIDTH, PREPROCESS_HEIGHT)
            .data;

        for (let srcIndex = 0, dstIndex = 0; srcIndex < data.length; srcIndex += 4, dstIndex++) {
            const gray = data[srcIndex];
            const normalized = gray / 255.0;
            // Preserve legacy double-normalization used during model training
            targetBuffer[dstIndex] = (normalized - 128.0) / 128.0;
        }
    }

    async predict(rightEyeCanvas, leftEyeCanvas) {
        if (!this.isInitialized) {
            throw new Error("ONNX model not initialized");
        }

        this.prepareEye(rightEyeCanvas, this.rightBuffer);
        this.prepareEye(leftEyeCanvas, this.leftBuffer);

        const feeds = {
            [this.inputNames[0]]: this.rightTensor,
            [this.inputNames[1]]: this.leftTensor
        };

        const results = await this.session.run(feeds);
        const prediction = results[this.outputName];

        console.log('[Model] Gaze predictor output:', {
            tensorName: this.outputName,
            x: prediction.data[0],
            y: prediction.data[1],
            raw: prediction.data
        });

        return {
            x: prediction.data[0],
            y: prediction.data[1]
        };
    }
}

class EyeTracker {
  constructor() {
    this.normalizer = new MixedNormalizer();
    this.predictor = new RageNetPredictor();
    this.isRunning = false;
    this.isInitialized = false;
    this.video = null;
    this.canvas = null;
    this.ctx = null;
    this.gazeCanvas = null;
    this.gazeCtx = null;
    this.overlayElements = null;
    this.animationId = null;
    this.frameCount = 0;
    this.lastFpsUpdate = 0;
    this.viewportWidth = window.innerWidth;
    this.viewportHeight = window.innerHeight;
    this.resizeHandler = null;
    this.rightEyeDebugCanvas = null;
    this.leftEyeDebugCanvas = null;
    this.rightEyeDebugCtx = null;
    this.leftEyeDebugCtx = null;
  }

  async initialize() {
    try {
      // Initialize components
      await this.normalizer.initialize();
      await this.predictor.initialize();

      // Get DOM elements
      this.video = document.getElementById('webcam');
      this.gazeCanvas = document.getElementById('gazeCanvas');
      this.gazeCtx = this.gazeCanvas.getContext('2d');
      this.overlayElements = {
        gazeX: document.getElementById('gazeX'),
        gazeY: document.getElementById('gazeY'),
        fps: document.getElementById('fps')
      };
      this.rightEyeDebugCanvas = document.getElementById('rightEyeDebug');
      this.leftEyeDebugCanvas = document.getElementById('leftEyeDebug');

      if (this.rightEyeDebugCanvas) {
        this.rightEyeDebugCanvas.width = EYE_ROI_SIZE[0];
        this.rightEyeDebugCanvas.height = EYE_ROI_SIZE[1];
        this.rightEyeDebugCtx = this.rightEyeDebugCanvas.getContext('2d', { willReadFrequently: true });
        this.rightEyeDebugCtx.imageSmoothingEnabled = false;
      }

      if (this.leftEyeDebugCanvas) {
        this.leftEyeDebugCanvas.width = EYE_ROI_SIZE[0];
        this.leftEyeDebugCanvas.height = EYE_ROI_SIZE[1];
        this.leftEyeDebugCtx = this.leftEyeDebugCanvas.getContext('2d', { willReadFrequently: true });
        this.leftEyeDebugCtx.imageSmoothingEnabled = false;
      }

      // Set up gaze canvas
      if (!this.resizeHandler) {
        this.resizeHandler = () => this.resizeCanvas();
        window.addEventListener('resize', this.resizeHandler);
      }
      this.resizeCanvas();

      // Create hidden canvas for video processing
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d');

      this.updateOverlay({ gazeX: null, gazeY: null, fps: null });
      this.isInitialized = true;
      return true;
    } catch (error) {
      throw new Error(`Failed to initialize eye tracker: ${error.message}`);
    }
  }

  async startWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 1920,
            height: 1080,
            frameRate: 30
          }
        });
        this.video.srcObject = stream;

        return new Promise((resolve) => {
          this.video.onloadedmetadata = () => {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            resolve();
          };
        });
      } catch (error) {
        throw new Error(`Failed to access webcam: ${error.message}`);
      }
  }

  stopWebcam() {
    if (this.video && this.video.srcObject) {
      const tracks = this.video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      this.video.srcObject = null;
    }
  }

  resizeCanvas() {
    if (!this.gazeCanvas) {
      return;
    }
    this.viewportWidth = window.innerWidth;
    this.viewportHeight = window.innerHeight;
    this.gazeCanvas.width = this.viewportWidth;
    this.gazeCanvas.height = this.viewportHeight;
  }

  updateOverlay({ gazeX, gazeY, fps } = {}) {
    if (!this.overlayElements) {
      return;
    }

    if (gazeX !== undefined) {
      this.overlayElements.gazeX.textContent = gazeX === null ? '-' : gazeX.toFixed(2);
    }

    if (gazeY !== undefined) {
      this.overlayElements.gazeY.textContent = gazeY === null ? '-' : gazeY.toFixed(2);
    }

    if (fps !== undefined) {
      this.overlayElements.fps.textContent = fps === null ? '-' : fps.toFixed(1);
    }
  }

  async processFrame() {
    if (!this.isRunning) return;

    try {
      // Draw video frame to hidden canvas
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

      const normalizedEyes = this.normalizer.normalizeFrame(this.canvas);
      if (!normalizedEyes) {
        this.updateOverlay({ gazeX: null, gazeY: null });
        this.animationId = requestAnimationFrame(() => this.processFrame());
        return;
      }

      const [rightEyeCanvas, leftEyeCanvas] = normalizedEyes;

      if (this.rightEyeDebugCtx && rightEyeCanvas) {
        this.rightEyeDebugCtx.drawImage(
          rightEyeCanvas,
          0,
          0,
          this.rightEyeDebugCanvas.width,
          this.rightEyeDebugCanvas.height
        );
      }

      if (this.leftEyeDebugCtx && leftEyeCanvas) {
        this.leftEyeDebugCtx.drawImage(
          leftEyeCanvas,
          0,
          0,
          this.leftEyeDebugCanvas.width,
          this.leftEyeDebugCanvas.height
        );
      }

      // Predict gaze
      const prediction = await this.predictor.predict(rightEyeCanvas, leftEyeCanvas);
      const gazeX = prediction.x * this.viewportWidth;
      const gazeY = prediction.y * this.viewportHeight;

      // Clamp to screen bounds
      const clampedX = Math.max(0, Math.min(gazeX, this.viewportWidth - 1));
      const clampedY = Math.max(0, Math.min(gazeY, this.viewportHeight - 1));

      // Draw gaze point
      this.gazeCtx.clearRect(0, 0, this.gazeCanvas.width, this.gazeCanvas.height);
      this.gazeCtx.fillStyle = 'red';
      this.gazeCtx.beginPath();
      this.gazeCtx.arc(clampedX, clampedY, 15, 0, 2 * Math.PI);
      this.gazeCtx.fill();

      this.updateOverlay({ gazeX: clampedX, gazeY: clampedY });

      // Update stats
      this.frameCount++;
      const now = performance.now();
      if (now - this.lastFpsUpdate > 1000) {
        const fps = this.frameCount / ((now - this.lastFpsUpdate) / 1000);
        this.updateOverlay({ fps });
        this.frameCount = 0;
        this.lastFpsUpdate = now;
      }
    } catch (error) {
        console.error('Tracking error:', error);
    }

    this.animationId = requestAnimationFrame(() => this.processFrame());
  }

  async start() {
    if (this.isRunning) return;

    try {
      this.resizeCanvas();
      await this.startWebcam();
      this.isRunning = true;
      this.lastFpsUpdate = performance.now();
      this.frameCount = 0;
      this.updateOverlay({ gazeX: null, gazeY: null, fps: 0 });
      this.processFrame();

    } catch (error) {
      console.error('Failed to start:', error);
      this.stop();
      throw error;
    }
  }

  stop() {
    this.isRunning = false;

    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    this.stopWebcam();

    // Clear displays
    if (this.gazeCtx) {
      this.gazeCtx.clearRect(0, 0, this.gazeCanvas.width, this.gazeCanvas.height);
    }

    if (this.rightEyeDebugCtx && this.rightEyeDebugCanvas) {
      this.rightEyeDebugCtx.clearRect(0, 0, this.rightEyeDebugCanvas.width, this.rightEyeDebugCanvas.height);
    }

    if (this.leftEyeDebugCtx && this.leftEyeDebugCanvas) {
      this.leftEyeDebugCtx.clearRect(0, 0, this.leftEyeDebugCanvas.width, this.leftEyeDebugCanvas.height);
    }

    this.updateOverlay({ gazeX: null, gazeY: null, fps: null });
  }
}

// Application initialization
let eyeTracker = new EyeTracker();

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const fullscreenBtn = document.getElementById('fullscreenBtn');

startBtn.addEventListener('click', async () => {
    if (!eyeTracker) return;

    startBtn.disabled = true;

    try {
        if (!eyeTracker.isInitialized) {
            await eyeTracker.initialize();
        }
        await eyeTracker.start();
        stopBtn.disabled = false;
    } catch (error) {
        console.error(error);
        startBtn.disabled = false;
    }
});

stopBtn.addEventListener('click', () => {
    if (!eyeTracker) return;

    stopBtn.disabled = true;
    eyeTracker.stop();
    startBtn.disabled = false;
});

const updateFullscreenButton = () => {
    if (!fullscreenBtn) return;
    fullscreenBtn.textContent = document.fullscreenElement ? 'Exit Fullscreen' : 'Enter Fullscreen';
};

if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', async () => {
        try {
            if (!document.fullscreenElement) {
                await document.documentElement.requestFullscreen();
            } else {
                await document.exitFullscreen();
            }
        } catch (error) {
            console.error('Fullscreen toggle failed:', error);
        } finally {
            updateFullscreenButton();
        }
    });

    document.addEventListener('fullscreenchange', updateFullscreenButton);
    updateFullscreenButton();
}
