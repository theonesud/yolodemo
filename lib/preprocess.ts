/**
 * Image pre-processing: "letterbox" an arbitrary frame into the fixed
 * 640x640 model input (aspect ratio preserved, grey padding), then convert it
 * to a normalised NCHW Float32 tensor.
 */

export const INPUT_SIZE = 640;
export const PAD_VALUE = 114;

export interface LetterboxMeta {
  /** input/original scale factor. */
  scale: number;
  /** Padding offsets inside the 640x640 input, in input pixels. */
  padX: number;
  padY: number;
  origW: number;
  origH: number;
}

export interface Preprocessed {
  data: Float32Array; // 1x3x640x640 NCHW, 0..1
  meta: LetterboxMeta;
}

let scratch: HTMLCanvasElement | null = null;

function getScratch(): HTMLCanvasElement {
  if (!scratch) {
    scratch = document.createElement("canvas");
    scratch.width = INPUT_SIZE;
    scratch.height = INPUT_SIZE;
  }
  return scratch;
}

export function letterbox(
  source: CanvasImageSource,
  origW: number,
  origH: number,
): Preprocessed {
  const canvas = getScratch();
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2D canvas context unavailable");

  ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);

  const scale = Math.min(INPUT_SIZE / origW, INPUT_SIZE / origH);
  const drawW = Math.round(origW * scale);
  const drawH = Math.round(origH * scale);
  const padX = Math.floor((INPUT_SIZE - drawW) / 2);
  const padY = Math.floor((INPUT_SIZE - drawH) / 2);

  ctx.drawImage(source, 0, 0, origW, origH, padX, padY, drawW, drawH);

  const rgba = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const area = INPUT_SIZE * INPUT_SIZE;
  const out = new Float32Array(3 * area);

  for (let i = 0, p = 0; i < area; i++, p += 4) {
    out[i] = rgba[p] / 255;
    out[area + i] = rgba[p + 1] / 255;
    out[2 * area + i] = rgba[p + 2] / 255;
  }

  return {
    data: out,
    meta: { scale, padX, padY, origW, origH },
  };
}

/** Convert an input-space coordinate back to original image space. */
export function toOriginal(v: number, pad: number, scale: number): number {
  return (v - pad) / scale;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
