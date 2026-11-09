import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { HandLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import {
  computeCoverScale,
  containerPxToNormalizedVideo,
  computeWorldUnitsPerMM,
  computeModelMMPerUnit,
  computeCalibration,
  computeHandChirality,
  liveScaleFactor,
  liveRollDelta,
  shortestAngleDelta,
  sampleBrightness,
  MIN_BRIGHTNESS,
} from './calibration.js';
import './style.css';

// Stable "rotate vector a onto vector b". THREE's setFromUnitVectors is
// unstable when a and b are nearly opposite (~180 deg); in that case we
// rotate 180 deg around an explicit axis perpendicular to a.
function robustQuatFromTo(a, b) {
  const q = new THREE.Quaternion();
  const d = a.dot(b);
  if (d < -0.9999) {
    let perp = new THREE.Vector3(1, 0, 0).cross(a);
    if (perp.lengthSq() < 1e-6) perp = new THREE.Vector3(0, 0, 1).cross(a);
    perp.normalize();
    return q.setFromAxisAngle(perp, Math.PI);
  }
  return q.setFromUnitVectors(a, b);
}

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const statusEl = document.getElementById('status');

const calibrationOverlay = document.getElementById('calibration-overlay');
const cardOutline = document.getElementById('card-outline');
const brightnessStatusEl = document.getElementById('brightness-status');
const startBtn = document.getElementById('calibrate-start-btn');
const countdownEl = document.getElementById('calibrate-countdown');
const recalibrateBtn = document.getElementById('recalibrate-btn');

// Device detection: on mobile, the back camera is used to look at your
// own wrist (no mirroring needed -- it shows the real world, like a
// window, not a reflection). On desktop, the built-in webcam faces YOU,
// so it needs mirroring to feel natural, like a mirror. Pose Landmarker
// also depends on this: it needs your elbow visible, which is realistic
// on a laptop (camera far from your wrist) but usually impossible on a
// phone (camera held close, right at the wrist) -- so it's desktop-only.
function isMobileDevice() {
  return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.matchMedia('(pointer: coarse)').matches;
}

const IS_MOBILE = isMobileDevice();
const CAMERA_FACING = IS_MOBILE ? 'environment' : 'user';

const PLACEMENT_DEPTH = -1;

// The MediaPipe wrist landmark sits right at the wrist crease -- a real
// watch is worn a bit further up the forearm, away from the hand. This
// is a first guess, not a measurement; expect to tune it visually.
// A 45 mm watch sits with its near edge a few mm clear of the wrist bone,
// so its centre is roughly 26 mm up the forearm from the wrist landmark.
const WRIST_TO_WATCH_OFFSET_MM = 26;

// Realistic Timex-style model built in Blender (crystal measured at 39 mm
// from the reference photo using the credit-card-sized authenticity card).
// The previous placeholder is still at /models/watch_reference.glb (crystal 36 mm).
const MODEL_PATH = '/models/watch_realistic.glb';
const REAL_CRYSTAL_DIAMETER_MM = 39;
const REFERENCE_MESH_NAME = 'Crystal';

const ROLL_DIRECTION = 1;
const ROLL_SMOOTHING = 0.25;

// Pose Landmarker (and therefore live pitch/yaw + occlusion) is
// desktop-only for now -- see the comment above. On mobile, the watch
// holds a fixed/neutral tilt; only roll (twisting) still tracks live.
// This is a real, known capability gap on mobile, not a hidden one.
const PITCH_YAW_ENABLED = !IS_MOBILE;
const ARM_AXIS_FLIP = -1;
const POSE_THROTTLE = 8;
const POSE_EASE_RATE = 0.35;
const POSE_WRIST_MATCH_THRESHOLD = 0.15;
const POSE_ELBOW = { left: 13, right: 14 };
const POSE_WRIST = { left: 15, right: 16 };

const MODEL_ARM_AXIS = new THREE.Vector3(0, 1, 0);

const OCCLUDER_ENABLED = PITCH_YAW_ENABLED;
// Sized for the realistic watch's wrist-shaped strap loop: the "arm" cylinder
// is ~52 mm across, its axis sits 28 mm behind the dial centre (the caseback is
// at 5 mm, a typical wrist is ~22 mm deep), so the strap shows around the sides
// of the arm and is hidden where it passes behind it.
const OCCLUDER_RADIUS_MM = 26;
const OCCLUDER_CENTER_DEPTH_MM = 28;
const OCCLUDER_LENGTH_MM = 160;
const OCCLUDER_SEGMENTS = 16;

// Hide the watch once its dial turns away from the camera (cos of angle).
const FACING_AWAY_THRESHOLD = 0.05;

// Time to get the card against the wrist at arm's length before auto-capture.
// (Keep in sync with the button label in index.html.)
const CALIBRATION_COUNTDOWN_SECONDS = 8;

let modelMMPerUnit = null;

const debugTransform = {
  rotX: 0,
  rotY: 0,
  rotZ: -Math.PI / 2, // confirmed correct across multiple sessions -- baked in as the default
};

let handLandmarker, poseLandmarker;
let scene, camera, renderer, watchModel, occluderMesh;
let lastVideoTime = -1;
let frameCount = 0;
let detectCount = 0;

let lastLandmarks = null;
let calibration = null;
let smoothedScale = 1;
let smoothedRollDelta = 0;

let committedDir = new THREE.Vector3(0, 1, 0);
let poseTargetDir = new THREE.Vector3(0, 1, 0);
let poseEverFound = false;

let lightOk = false;
let countdownTimer = null;

const brightnessCanvas = document.createElement('canvas');
brightnessCanvas.width = 32;
brightnessCanvas.height = 32;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: CAMERA_FACING, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;

  if (CAMERA_FACING === 'user') {
    video.classList.add('mirrored');
    canvas.classList.add('mirrored');
  }

  return new Promise((resolve) => {
    video.onloadedmetadata = () => resolve(video);
  });
}

async function setupTrackers() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'CPU',
    },
    runningMode: 'VIDEO',
    numHands: 1,
  });
  statusEl.dataset.delegate = 'CPU (forced)';

  if (PITCH_YAW_ENABLED) {
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
  }
}

function dumpSceneNames(root) {
  const names = [];
  root.traverse(function (obj) {
    names.push(obj.type + ': "' + obj.name + '"');
  });
  console.log('Loaded model scene graph:\n' + names.join('\n'));
}

function loadWatchModel() {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(
      MODEL_PATH,
      (gltf) => {
        const model = gltf.scene;
        dumpSceneNames(model);

        const referenceObj = model.getObjectByName(REFERENCE_MESH_NAME);
        if (!referenceObj) {
          reject(new Error('Reference object "' + REFERENCE_MESH_NAME + '" not found -- see console for actual scene names.'));
          return;
        }

        const box = new THREE.Box3().setFromObject(referenceObj);
        const size = new THREE.Vector3();
        box.getSize(size);
        const measuredDiameterUnits = Math.max(size.x, size.y);

        if (!isFinite(measuredDiameterUnits) || measuredDiameterUnits <= 0) {
          reject(new Error('Reference object "' + REFERENCE_MESH_NAME + '" found but has no measurable geometry.'));
          return;
        }

        modelMMPerUnit = computeModelMMPerUnit(REAL_CRYSTAL_DIAMETER_MM, measuredDiameterUnits);
        console.log(
          'Derived scale: ' + measuredDiameterUnits.toFixed(5) + ' model units = ' + REAL_CRYSTAL_DIAMETER_MM + 'mm -> ' + modelMMPerUnit.toFixed(4) + ' mm/unit'
        );

        model.visible = false;
        scene.add(model);
        resolve(model);
      },
      undefined,
      (err) => reject(err)
    );
  });
}

function setupOccluder() {
  if (!OCCLUDER_ENABLED) return;
  const geometry = new THREE.CylinderGeometry(1, 1, 1, OCCLUDER_SEGMENTS);
  const material = new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthWrite: true,
    depthTest: true,
  });
  occluderMesh = new THREE.Mesh(geometry, material);
  occluderMesh.renderOrder = -1;
  occluderMesh.visible = false;
  scene.add(occluderMesh);
}

function setupThree() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
  camera.position.z = 2;

  renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
  keyLight.position.set(1, 1, 2);
  scene.add(keyLight);
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  // Metals (case, bezel, buckle) only look like metal if there is something
  // to reflect. The old code built the environment from an EMPTY scene, which
  // is pure black, so steel rendered black. RoomEnvironment is a small built-in
  // studio-like scene that gives polished steel soft highlights.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 1.0;

  window.addEventListener('resize', onResize);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function normalizedToWorld(nx, ny, depth) {
  if (depth === undefined) depth = PLACEMENT_DEPTH;
  const vector = new THREE.Vector3((nx - 0.5) * 2, -(ny - 0.5) * 2, 0.5);
  vector.unproject(camera);
  const dir = vector.sub(camera.position).normalize();
  const distance = (depth - camera.position.z) / dir.z;
  return camera.position.clone().add(dir.multiplyScalar(distance));
}

function computePoseArmDirection(poseLandmarks, handWrist) {
  if (!poseLandmarks || poseLandmarks.length === 0) return null;
  const pose = poseLandmarks[0];

  const distLeft = Math.hypot(pose[POSE_WRIST.left].x - handWrist.x, pose[POSE_WRIST.left].y - handWrist.y);
  const distRight = Math.hypot(pose[POSE_WRIST.right].x - handWrist.x, pose[POSE_WRIST.right].y - handWrist.y);

  const side = distLeft <= distRight ? 'left' : 'right';
  const bestDist = Math.min(distLeft, distRight);

  if (bestDist > POSE_WRIST_MATCH_THRESHOLD) return null;

  const elbow = pose[POSE_ELBOW[side]];
  const wrist = pose[POSE_WRIST[side]];

  const elbowWorld = normalizedToWorld(elbow.x, elbow.y);
  const wristWorld = normalizedToWorld(wrist.x, wrist.y);
  const dir = wristWorld.sub(elbowWorld);
  dir.z = 0;
  return dir.lengthSq() > 0.00001 ? dir.normalize() : null;
}

function setupDebugControls() {
  const panel = document.createElement('div');
  panel.id = 'debug-panel';

  const rows = [
    { label: 'Rot X', id: 'rotx', initial: 0 },
    { label: 'Rot Y', id: 'roty', initial: 0 },
    { label: 'Rot Z', id: 'rotz', initial: -90 },
  ];

  rows.forEach(function (row) {
    const rowDiv = document.createElement('div');
    rowDiv.className = 'debug-row';

    const label = document.createElement('label');
    label.textContent = row.label + ': ';
    const span = document.createElement('span');
    span.id = 'dbg-' + row.id + '-val';
    span.textContent = row.initial + 'deg';
    label.appendChild(span);

    const minusBtn = document.createElement('button');
    minusBtn.textContent = '-';
    minusBtn.dataset.action = row.id + '-down';

    const plusBtn = document.createElement('button');
    plusBtn.textContent = '+';
    plusBtn.dataset.action = row.id + '-up';

    rowDiv.appendChild(label);
    rowDiv.appendChild(minusBtn);
    rowDiv.appendChild(plusBtn);
    panel.appendChild(rowDiv);
  });

  document.getElementById('app').appendChild(panel);

  const rotXValEl = document.getElementById('dbg-rotx-val');
  const rotYValEl = document.getElementById('dbg-roty-val');
  const rotZValEl = document.getElementById('dbg-rotz-val');

  function deg(rad) {
    return Math.round((rad * 180) / Math.PI);
  }

  panel.addEventListener('click', function (e) {
    const action = e.target.dataset.action;
    if (!action) return;

    const ROT_STEP = Math.PI / 2;

    switch (action) {
      case 'rotx-up': debugTransform.rotX += ROT_STEP; break;
      case 'rotx-down': debugTransform.rotX -= ROT_STEP; break;
      case 'roty-up': debugTransform.rotY += ROT_STEP; break;
      case 'roty-down': debugTransform.rotY -= ROT_STEP; break;
      case 'rotz-up': debugTransform.rotZ += ROT_STEP; break;
      case 'rotz-down': debugTransform.rotZ -= ROT_STEP; break;
    }

    rotXValEl.textContent = deg(debugTransform.rotX) + 'deg';
    rotYValEl.textContent = deg(debugTransform.rotY) + 'deg';
    rotZValEl.textContent = deg(debugTransform.rotZ) + 'deg';
  });
}

function checkBrightnessLoop() {
  if (video.readyState >= 2) {
    const brightness = sampleBrightness(video, brightnessCanvas);
    lightOk = brightness >= MIN_BRIGHTNESS;
    brightnessStatusEl.textContent = lightOk
      ? 'Lighting OK'
      : 'Too dark (' + Math.round(brightness) + '/' + MIN_BRIGHTNESS + ') -- move somewhere brighter';
  }
  setTimeout(checkBrightnessLoop, 400);
}

function startCalibrationCountdown() {
  startBtn.hidden = true;
  countdownEl.hidden = false;
  countdownEl.onclick = null;

  let secondsLeft = CALIBRATION_COUNTDOWN_SECONDS;
  countdownEl.textContent = 'Positioning card... ' + secondsLeft;

  countdownTimer = setInterval(function () {
    secondsLeft -= 1;
    if (secondsLeft <= 0) {
      clearInterval(countdownTimer);
      countdownTimer = null;
      attemptAutoCapture();
    } else {
      countdownEl.textContent = 'Positioning card... ' + secondsLeft;
    }
  }, 1000);
}

function attemptAutoCapture() {
  if (lightOk && lastLandmarks) {
    captureCalibration();
  } else {
    const reason = !lastLandmarks ? 'hand not detected' : 'lighting too dark';
    countdownEl.textContent = "Couldn't capture (" + reason + ') -- tap to retry';
    countdownEl.onclick = resetCalibrationUI;
  }
}

function resetCalibrationUI() {
  countdownEl.hidden = true;
  countdownEl.onclick = null;
  startBtn.hidden = false;
}

function captureCalibration() {
  const outlineRect = cardOutline.getBoundingClientRect();
  const coverScale = computeCoverScale(video.videoWidth, video.videoHeight, window.innerWidth, window.innerHeight);

  const midY = outlineRect.top + outlineRect.height / 2;
  const leftNorm = containerPxToNormalizedVideo(
    outlineRect.left, midY, video.videoWidth, video.videoHeight, window.innerWidth, window.innerHeight, coverScale
  );
  const rightNorm = containerPxToNormalizedVideo(
    outlineRect.right, midY, video.videoWidth, video.videoHeight, window.innerWidth, window.innerHeight, coverScale
  );
  const leftWorld = normalizedToWorld(leftNorm.x, leftNorm.y);
  const rightWorld = normalizedToWorld(rightNorm.x, rightNorm.y);
  const cardWorldWidth = leftWorld.distanceTo(rightWorld);
  const worldUnitsPerMM = computeWorldUnitsPerMM(cardWorldWidth);

  calibration = computeCalibration({
    landmarks: lastLandmarks,
    videoNativeW: video.videoWidth,
    videoNativeH: video.videoHeight,
    worldUnitsPerMM: worldUnitsPerMM,
  });

  smoothedScale = 1;
  smoothedRollDelta = 0;
  calibrationOverlay.hidden = true;
  recalibrateBtn.hidden = false;
}

function startRecalibration() {
  calibration = null;
  watchModel.visible = false;
  if (occluderMesh) occluderMesh.visible = false;
  calibrationOverlay.hidden = false;
  recalibrateBtn.hidden = true;
  resetCalibrationUI();
}

startBtn.addEventListener('click', startCalibrationCountdown);
recalibrateBtn.addEventListener('click', startRecalibration);

function updateDiagnostics(handFound, tiltDeg, poseStatus) {
  const detectRate = frameCount > 0 ? Math.round((detectCount / frameCount) * 100) : 0;
  const calText = calibration
    ? 'scale: ' + smoothedScale.toFixed(2) + 'x | roll: ' + Math.round((smoothedRollDelta * 180) / Math.PI) + 'deg | tilt: ' + (tiltDeg !== null ? tiltDeg + 'deg' : '--') + ' | pose: ' + poseStatus
    : 'not calibrated';
  statusEl.textContent = (handFound ? 'Hand detected' : 'No hand') + ' | ' + (IS_MOBILE ? 'mobile' : 'desktop') + ' | detect rate: ' + detectRate + '% | ' + calText;
}

function renderLoop() {
  requestAnimationFrame(renderLoop);

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    frameCount++;

    const result = handLandmarker.detectForVideo(video, performance.now());
    const found = result.landmarks && result.landmarks.length > 0;
    lastLandmarks = found ? result.landmarks[0] : null;

    let tiltDegDisplay = null;
    let poseStatus = 'off (mobile)';

    if (PITCH_YAW_ENABLED) {
      if (found && frameCount % POSE_THROTTLE === 0) {
        const poseResult = poseLandmarker.detectForVideo(video, performance.now());
        const newDir = computePoseArmDirection(poseResult.landmarks, lastLandmarks[0]);
        if (newDir) {
          poseTargetDir.copy(newDir);
          poseEverFound = true;
        }
      }
      poseStatus = poseEverFound ? 'tracking' : 'searching...';
    }

    if (found && calibration) {
      detectCount++;

      const rawScale = liveScaleFactor(calibration, lastLandmarks, video.videoWidth, video.videoHeight);
      smoothedScale = smoothedScale + (rawScale - smoothedScale) * 0.2;

      const rawRollDelta = liveRollDelta(calibration, lastLandmarks, video.videoWidth, video.videoHeight);
      smoothedRollDelta += shortestAngleDelta(rawRollDelta, smoothedRollDelta) * ROLL_SMOOTHING;
      smoothedRollDelta = shortestAngleDelta(smoothedRollDelta, 0);

      // If the hand's own geometric chirality has flipped since
      // calibration (switched hands, or flipped palm/back-of-hand
      // toward the camera), the roll direction needs to flip too --
      // otherwise a real twist reads backwards on the "wrong" hand.
      const currentChirality = computeHandChirality(lastLandmarks, video.videoWidth, video.videoHeight);
      const chiralitySign = currentChirality === calibration.chiralityAtCalibration ? 1 : -1;

      const wrist = lastLandmarks[0];
      const wristPoint = normalizedToWorld(wrist.x, wrist.y);

      // Direction from the palm (middle-knuckle) through the wrist,
      // continued -- this points up the forearm, away from the fingers.
      // Same two landmarks calibration.js already uses for scale
      // reference, so this works identically on desktop and mobile.
      const middleMcp = lastLandmarks[9];
      const middleMcpWorld = normalizedToWorld(middleMcp.x, middleMcp.y);
      const backOffsetDir = wristPoint.clone().sub(middleMcpWorld).normalize();
      // Scale the offset by the live distance factor too (smoothedScale),
      // exactly like the model size below. Before, only the model shrank
      // when the hand moved away, while the offset stayed full-size, which
      // pushed the watch too far down the forearm.
      const backOffsetDistance = calibration.worldUnitsPerMM * smoothedScale * WRIST_TO_WATCH_OFFSET_MM;
      const worldPos = wristPoint.clone().addScaledVector(backOffsetDir, backOffsetDistance);

      watchModel.position.copy(worldPos);

      const finalScale = calibration.worldUnitsPerMM * modelMMPerUnit * smoothedScale;
      watchModel.scale.setScalar(finalScale);

      const baseQuat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(debugTransform.rotX, debugTransform.rotY, debugTransform.rotZ)
      );
      const rollQuat = new THREE.Quaternion().setFromAxisAngle(
        MODEL_ARM_AXIS,
        smoothedRollDelta * ROLL_DIRECTION * chiralitySign
      );

      let pitchYawQuat = new THREE.Quaternion();

      if (PITCH_YAW_ENABLED && poseEverFound) {
        const eased = committedDir.clone().lerp(poseTargetDir, POSE_EASE_RATE);
        if (eased.lengthSq() > 1e-6) committedDir.copy(eased.normalize());
        else committedDir.copy(poseTargetDir);

        const targetDir = committedDir.clone().multiplyScalar(ARM_AXIS_FLIP).normalize();
        pitchYawQuat = robustQuatFromTo(MODEL_ARM_AXIS, targetDir);
        tiltDegDisplay = Math.round((MODEL_ARM_AXIS.angleTo(targetDir) * 180) / Math.PI);
      }

      const finalQuat = pitchYawQuat.clone().multiply(rollQuat).multiply(baseQuat);
      watchModel.quaternion.copy(finalQuat);

      // Which way is the dial pointing? Camera looks down -Z, so a dial
      // normal with +Z faces the camera.
      const worldFaceNormal = new THREE.Vector3(0, 0, 1).applyQuaternion(finalQuat);
      const facingCamera = worldFaceNormal.z > FACING_AWAY_THRESHOLD;

      // When the occluder (invisible arm) is active it already hides exactly
      // the parts of the watch that are behind your wrist, so the watch stays
      // "visible" and you still see the strap wrap around the arm as you twist.
      // Only when there is no occluder (mobile) do we fall back to hiding the
      // whole watch once the dial faces away.
      const occluderActive = OCCLUDER_ENABLED && occluderMesh && PITCH_YAW_ENABLED && poseEverFound;

      if (OCCLUDER_ENABLED && occluderMesh) {
        if (occluderActive) {
          const radiusWorld = calibration.worldUnitsPerMM * smoothedScale * OCCLUDER_RADIUS_MM;
          const lengthWorld = calibration.worldUnitsPerMM * smoothedScale * OCCLUDER_LENGTH_MM;
          const depthWorld = calibration.worldUnitsPerMM * smoothedScale * OCCLUDER_CENTER_DEPTH_MM;

          const occluderCenter = worldPos.clone().addScaledVector(worldFaceNormal, -depthWorld);

          occluderMesh.scale.set(radiusWorld, lengthWorld, radiusWorld);
          occluderMesh.position.copy(occluderCenter);
          occluderMesh.quaternion.copy(pitchYawQuat);
          occluderMesh.visible = true;
        } else {
          occluderMesh.visible = false;
        }
      }

      watchModel.visible = occluderActive ? true : facingCamera;
    } else {
      if (found) detectCount++;
      watchModel.visible = false;
      if (occluderMesh) occluderMesh.visible = false;
    }

    updateDiagnostics(found, tiltDegDisplay, poseStatus);
  }

  renderer.render(scene, camera);
}

async function main() {
  try {
    statusEl.textContent = 'Requesting camera...';
    await setupCamera();
    statusEl.textContent = 'Loading trackers...';
    await setupTrackers();
    setupThree();
    setupOccluder();
    statusEl.textContent = 'Loading watch model...';
    watchModel = await loadWatchModel();
    setupDebugControls();
    statusEl.textContent = 'Ready';
    checkBrightnessLoop();
    renderLoop();
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
    console.error(err);
  }
}

main();
