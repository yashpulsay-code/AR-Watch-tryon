export const CARD_WIDTH_MM = 85.60;
export const MIN_BRIGHTNESS = 70;

const REF_LANDMARK_A = 0;
const REF_LANDMARK_B = 9;

const ROLL_LANDMARK_A = 5;
const ROLL_LANDMARK_B = 17;

export function computeCoverScale(videoNativeW, videoNativeH, containerW, containerH) {
  return Math.max(containerW / videoNativeW, containerH / videoNativeH);
}

export function containerPxToNormalizedVideo(screenX, screenY, videoNativeW, videoNativeH, containerW, containerH, coverScale) {
  const displayedVideoW = videoNativeW * coverScale;
  const displayedVideoH = videoNativeH * coverScale;
  const offsetX = (displayedVideoW - containerW) / 2;
  const offsetY = (displayedVideoH - containerH) / 2;
  const videoDisplayX = screenX + offsetX;
  const videoDisplayY = screenY + offsetY;
  return {
    x: videoDisplayX / displayedVideoW,
    y: videoDisplayY / displayedVideoH,
  };
}

export function handReferenceNormalizedLength(landmarks, videoNativeW, videoNativeH) {
  const a = landmarks[REF_LANDMARK_A];
  const b = landmarks[REF_LANDMARK_B];
  const dx = (b.x - a.x) * videoNativeW;
  const dy = (b.y - a.y) * videoNativeH;
  const pixelDist = Math.sqrt(dx * dx + dy * dy);
  return pixelDist / videoNativeW;
}

export function computeHandRollAngle(landmarks, videoNativeW, videoNativeH) {
  const a = landmarks[ROLL_LANDMARK_A];
  const b = landmarks[ROLL_LANDMARK_B];
  const dx = (b.x - a.x) * videoNativeW;
  const dy = (b.y - a.y) * videoNativeH;
  return Math.atan2(dy, dx);
}

export function shortestAngleDelta(a, b) {
  let diff = (a - b) % (2 * Math.PI);
  if (diff > Math.PI) diff -= 2 * Math.PI;
  if (diff < -Math.PI) diff += 2 * Math.PI;
  return diff;
}

// A self-consistent "handedness" signal derived purely from the hand's
// own geometry -- NOT MediaPipe's built-in handedness label, which has
// camera-mirroring conventions that are easy to get backwards. This is
// the sign of the cross product between wrist->indexMCP and
// wrist->pinkyMCP. Left and right hands are mirror images of each
// other, so this sign reliably flips between them (and also flips if
// the same hand shows its palm vs the back of the hand to the camera).
// We only ever compare this against its OWN value at calibration time,
// so we never need to know what "Left" or "Right" actually means here --
// only whether it's the SAME as when we calibrated.
export function computeHandChirality(landmarks, videoNativeW, videoNativeH) {
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const pinkyMcp = landmarks[17];

  const v1x = (indexMcp.x - wrist.x) * videoNativeW;
  const v1y = (indexMcp.y - wrist.y) * videoNativeH;
  const v2x = (pinkyMcp.x - wrist.x) * videoNativeW;
  const v2y = (pinkyMcp.y - wrist.y) * videoNativeH;

  const cross = v1x * v2y - v1y * v2x;
  return cross >= 0 ? 1 : -1;
}

export function sampleBrightness(video, sampleCanvas) {
  const ctx = sampleCanvas.getContext('2d');
  const w = sampleCanvas.width;
  const h = sampleCanvas.height;
  ctx.drawImage(video, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);

  let total = 0;
  const pixelCount = data.length / 4;
  for (let i = 0; i < data.length; i += 4) {
    total += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return total / pixelCount;
}

export function computeWorldUnitsPerMM(cardWorldWidthUnits) {
  return cardWorldWidthUnits / CARD_WIDTH_MM;
}

export function computeModelMMPerUnit(realWorldDiameterMM, measuredModelDiameterUnits) {
  return realWorldDiameterMM / measuredModelDiameterUnits;
}

export function computeCalibration({ landmarks, videoNativeW, videoNativeH, worldUnitsPerMM }) {
  const handRefNormalizedLength = handReferenceNormalizedLength(landmarks, videoNativeW, videoNativeH);
  const rollAngleAtCalibration = computeHandRollAngle(landmarks, videoNativeW, videoNativeH);
  const chiralityAtCalibration = computeHandChirality(landmarks, videoNativeW, videoNativeH);

  return {
    worldUnitsPerMM: worldUnitsPerMM,
    handRefNormalizedLengthAtCalibration: handRefNormalizedLength,
    rollAngleAtCalibration: rollAngleAtCalibration,
    chiralityAtCalibration: chiralityAtCalibration,
  };
}

export function liveScaleFactor(calibration, currentLandmarks, videoNativeW, videoNativeH) {
  const currentLength = handReferenceNormalizedLength(currentLandmarks, videoNativeW, videoNativeH);
  return currentLength / calibration.handRefNormalizedLengthAtCalibration;
}

export function liveRollDelta(calibration, currentLandmarks, videoNativeW, videoNativeH) {
  const currentAngle = computeHandRollAngle(currentLandmarks, videoNativeW, videoNativeH);
  return shortestAngleDelta(currentAngle, calibration.rollAngleAtCalibration);
}
