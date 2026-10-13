/**
 * Canvas overlay drawing for segmentation masks / boxes and pose skeletons.
 */

import { SKELETON, classColor } from "./coco";
import type { Detection, PoseDetection, SegDetection } from "./postprocess";

export interface DrawOptions {
  maskOpacity: number;
  keypointThreshold: number;
}

function labelFont(w: number): string {
  const size = Math.max(12, Math.round(w / 55));
  return `600 ${size}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;
}

function lineWidth(w: number): number {
  return Math.max(2, Math.round(w / 420));
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  w: number,
) {
  const size = Math.max(12, Math.round(w / 55));
  ctx.font = labelFont(w);
  const padX = size * 0.4;
  const padY = size * 0.25;
  const metrics = ctx.measureText(text);
  const boxW = metrics.width + padX * 2;
  const boxH = size + padY * 2;
  const top = Math.max(0, y - boxH);

  ctx.fillStyle = color;
  ctx.fillRect(x, top, boxW, boxH);
  ctx.fillStyle = "#0b0f14";
  ctx.textBaseline = "top";
  ctx.fillText(text, x + padX, top + padY);
}

function drawBox(
  ctx: CanvasRenderingContext2D,
  det: Detection,
  w: number,
  color: string,
) {
  const [x1, y1, x2, y2] = det.box;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth(w);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  drawLabel(ctx, `${det.className} ${(det.score * 100).toFixed(0)}%`, x1, y1, color, w);
}

/** Composites every mask onto one offscreen layer, then draws it once. */
function drawMasks(
  ctx: CanvasRenderingContext2D,
  dets: SegDetection[],
  opacity: number,
  w: number,
  h: number,
  scratch: HTMLCanvasElement,
) {
  const masks = dets.filter((d) => d.mask);
  if (!masks.length) return;

  if (scratch.width !== w || scratch.height !== h) {
    scratch.width = w;
    scratch.height = h;
  }
  const sctx = scratch.getContext("2d");
  if (!sctx) return;
  sctx.clearRect(0, 0, w, h);
  for (const det of masks) {
    const patch = det.mask!;
    sctx.drawImage(patch.canvas, Math.round(patch.x), Math.round(patch.y));
  }

  const prev = ctx.globalAlpha;
  ctx.globalAlpha = opacity;
  ctx.drawImage(scratch, 0, 0);
  ctx.globalAlpha = prev;
}

function drawPoseSkeletons(
  ctx: CanvasRenderingContext2D,
  dets: PoseDetection[],
  threshold: number,
  w: number,
) {
  const lw = lineWidth(w);
  const r = Math.max(3, Math.round(w / 260));

  for (const det of dets) {
    const color = classColor(det.classId);
    const kpts = det.keypoints;

    // bones
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.lineCap = "round";
    for (const [a, b] of SKELETON) {
      const ka = kpts[a];
      const kb = kpts[b];
      if (!ka || !kb) continue;
      if (ka.score < threshold || kb.score < threshold) continue;
      ctx.beginPath();
      ctx.moveTo(ka.x, ka.y);
      ctx.lineTo(kb.x, kb.y);
      ctx.stroke();
    }

    // joints
    for (const kp of kpts) {
      if (kp.score < threshold) continue;
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, r, 0, Math.PI * 2);
      ctx.fillStyle = "#f8fafc";
      ctx.fill();
      ctx.lineWidth = Math.max(1, lw - 1);
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }
}

export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  dets: Detection[],
  w: number,
  h: number,
  opts: DrawOptions,
  maskScratch: HTMLCanvasElement,
) {
  ctx.lineJoin = "round";
  ctx.textBaseline = "alphabetic";

  const segDets = dets.filter((d): d is SegDetection => d.task === "seg");
  const poseDets = dets.filter((d): d is PoseDetection => d.task === "pose");

  if (segDets.length) {
    drawMasks(ctx, segDets, opts.maskOpacity, w, h, maskScratch);
  }

  if (poseDets.length) {
    drawPoseSkeletons(ctx, poseDets, opts.keypointThreshold, w);
  }

  for (const det of dets) {
    drawBox(ctx, det, w, classColor(det.classId));
  }
}
