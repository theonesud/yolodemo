/**
 * Post-processing for the two NMS-free YOLO26 heads.
 *
 * Both models emit a fixed 300 detection rows in *letterboxed input* pixels:
 *
 *   segmentation row: [x1, y1, x2, y2, conf, cls, ...32 mask coefficients]
 *   pose row:         [x1, y1, x2, y2, conf, cls, ...17 * (x, y, conf)]
 *
 * Everything here is plain typed-array math so it stays fast on phones.
 */

import { INPUT_SIZE, clamp, toOriginal, type LetterboxMeta } from "./preprocess";
import { COCO_CLASSES, classColor } from "./coco";

export const MAX_DETECTIONS = 300;
export const MASK_ROWS = 38; // 6 + 32
export const POSE_ROW = 57; // 6 + 17 * 3
export const MASK_THRESHOLD = 0.5;

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface MaskPatch {
  canvas: HTMLCanvasElement;
  x: number;
  y: number;
}

export interface BaseDetection {
  box: [number, number, number, number]; // original image pixels
  score: number;
  classId: number;
  className: string;
}

export interface SegDetection extends BaseDetection {
  task: "seg";
  mask: MaskPatch | null;
}

export interface PoseDetection extends BaseDetection {
  task: "pose";
  keypoints: Keypoint[];
}

export type Detection = SegDetection | PoseDetection;

export interface DecodeOptions {
  confidence: number;
  /** Original output-space box rect (input px) is derived from the box. */
  buildMasks?: boolean;
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function clampBox(
  box: [number, number, number, number],
  w: number,
  h: number,
): [number, number, number, number] {
  const x1 = clamp(box[0], 0, w);
  const y1 = clamp(box[1], 0, h);
  const x2 = clamp(box[2], 0, w);
  const y2 = clamp(box[3], 0, h);
  return [x1, y1, Math.max(x1, x2), Math.max(y1, y2)];
}

/** Reconstruct a single instance mask from 32 coefficients x prototypes. */
function buildMaskPatch(
  coeffs: Float32Array,
  protos: Float32Array,
  nm: number,
  protoSize: number,
  inputBox: [number, number, number, number],
  origBox: [number, number, number, number],
  color: string,
): MaskPatch | null {
  const s = protoSize / INPUT_SIZE;

  const px1 = clamp(Math.floor(inputBox[0] * s), 0, protoSize - 1);
  const py1 = clamp(Math.floor(inputBox[1] * s), 0, protoSize - 1);
  const px2 = clamp(Math.ceil(inputBox[2] * s), 1, protoSize);
  const py2 = clamp(Math.ceil(inputBox[3] * s), 1, protoSize);

  const cw = px2 - px1;
  const ch = py2 - py1;
  if (cw <= 0 || ch <= 0) return null;

  const plane = protoSize * protoSize;
  const crop = new Float32Array(cw * ch);

  for (let y = 0; y < ch; y++) {
    const rowBase = (py1 + y) * protoSize + px1;
    for (let x = 0; x < cw; x++) {
      const pix = rowBase + x;
      let acc = 0;
      for (let c = 0; c < nm; c++) acc += coeffs[c] * protos[c * plane + pix];
      crop[y * cw + x] = 1 / (1 + Math.exp(-acc)); // sigmoid
    }
  }

  const bw = Math.max(1, Math.round(origBox[2] - origBox[0]));
  const bh = Math.max(1, Math.round(origBox[3] - origBox[1]));

  const canvas = document.createElement("canvas");
  canvas.width = bw;
  canvas.height = bh;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const image = ctx.createImageData(bw, bh);
  const [r, g, b] = hexToRgb(color);

  for (let j = 0; j < bh; j++) {
    const v = ((j + 0.5) / bh) * ch - 0.5;
    const y0 = clamp(Math.floor(v), 0, ch - 1);
    const y1 = Math.min(ch - 1, y0 + 1);
    const fy = clamp(v - y0, 0, 1);

    for (let i = 0; i < bw; i++) {
      const u = ((i + 0.5) / bw) * cw - 0.5;
      const x0 = clamp(Math.floor(u), 0, cw - 1);
      const x1 = Math.min(cw - 1, x0 + 1);
      const fx = clamp(u - x0, 0, 1);

      const top = crop[y0 * cw + x0] * (1 - fx) + crop[y0 * cw + x1] * fx;
      const bot = crop[y1 * cw + x0] * (1 - fx) + crop[y1 * cw + x1] * fx;
      const value = top * (1 - fy) + bot * fy;

      if (value > MASK_THRESHOLD) {
        const o = (j * bw + i) * 4;
        image.data[o] = r;
        image.data[o + 1] = g;
        image.data[o + 2] = b;
        image.data[o + 3] = 235;
      }
    }
  }

  ctx.putImageData(image, 0, 0);
  return { canvas, x: origBox[0], y: origBox[1] };
}

export function decodeSegmentation(
  out0: Float32Array,
  protos: Float32Array,
  meta: LetterboxMeta,
  confidence: number,
  nm: number,
  protoSize: number,
  buildMasks: boolean,
): SegDetection[] {
  const detections: SegDetection[] = [];

  for (let i = 0; i < MAX_DETECTIONS; i++) {
    const base = i * MASK_ROWS;
    const score = out0[base + 4];
    if (score < confidence) continue;

    const classId = Math.round(out0[base + 5]);
    const className = COCO_CLASSES[classId] ?? `class ${classId}`;

    const inputBox: [number, number, number, number] = [
      out0[base],
      out0[base + 1],
      out0[base + 2],
      out0[base + 3],
    ];

    const origBox = clampBox(
      [
        toOriginal(inputBox[0], meta.padX, meta.scale),
        toOriginal(inputBox[1], meta.padY, meta.scale),
        toOriginal(inputBox[2], meta.padX, meta.scale),
        toOriginal(inputBox[3], meta.padY, meta.scale),
      ],
      meta.origW,
      meta.origH,
    );

    const color = classColor(classId);
    let mask: MaskPatch | null = null;
    if (buildMasks) {
      const coeffs = out0.subarray(base + 6, base + 6 + nm);
      mask = buildMaskPatch(coeffs, protos, nm, protoSize, inputBox, origBox, color);
    }

    detections.push({
      task: "seg",
      box: origBox,
      score,
      classId,
      className,
      mask,
    });
  }

  detections.sort((a, b) => b.score - a.score);
  return detections;
}

export function decodePose(
  out0: Float32Array,
  meta: LetterboxMeta,
  confidence: number,
): PoseDetection[] {
  const detections: PoseDetection[] = [];
  const kptCount = (POSE_ROW - 6) / 3;

  for (let i = 0; i < MAX_DETECTIONS; i++) {
    const base = i * POSE_ROW;
    const score = out0[base + 4];
    if (score < confidence) continue;

    const classId = Math.round(out0[base + 5]);
    const className = classId === 0 ? "person" : `class ${classId}`;

    const origBox = clampBox(
      [
        toOriginal(out0[base], meta.padX, meta.scale),
        toOriginal(out0[base + 1], meta.padY, meta.scale),
        toOriginal(out0[base + 2], meta.padX, meta.scale),
        toOriginal(out0[base + 3], meta.padY, meta.scale),
      ],
      meta.origW,
      meta.origH,
    );

    const keypoints: Keypoint[] = [];
    for (let k = 0; k < kptCount; k++) {
      const o = base + 6 + k * 3;
      keypoints.push({
        x: toOriginal(out0[o], meta.padX, meta.scale),
        y: toOriginal(out0[o + 1], meta.padY, meta.scale),
        score: out0[o + 2],
      });
    }

    detections.push({ task: "pose", box: origBox, score, classId, className, keypoints });
  }

  detections.sort((a, b) => b.score - a.score);
  return detections;
}
