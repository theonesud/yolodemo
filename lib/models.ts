/**
 * Model registry + on-device download/caching.
 *
 * We use the *nano* (smallest) YOLO26 weights, exported to ONNX by the
 * Ultralytics community as NMS-free end-to-end graphs (`nms=False`), which
 * keeps the browser post-processing tiny: no non-max suppression required.
 *
 *   - yolo26n-seg.onnx   -> output0 [1,300,38]  (xyxy,conf,cls + 32 mask coeffs)
 *                           output1 [1,32,160,160] (mask prototypes)
 *   - yolo26n-pose.onnx  -> output0 [1,300,57]  (xyxy,conf,cls + 17 * (x,y,conf))
 *
 * Files are fetched once from Hugging Face and stored in the browser Cache
 * Storage API, so every later visit is instant and works fully offline.
 */

export type Task = "seg" | "pose";

export interface ModelSpec {
  task: Task;
  id: string;
  label: string;
  /** Approx. download size in bytes (used for the progress UI). */
  size: number;
  url: string;
  inputSize: number;
  /** Number of mask prototypes (seg only). */
  maskCoeffs?: number;
  protoSize?: number;
}

const HF = "https://huggingface.co/zwh20081/yolo26-onnx/resolve/main";

export const MODELS: Record<Task, ModelSpec> = {
  seg: {
    task: "seg",
    id: "yolo26n-seg",
    label: "YOLO26n · segmentation",
    size: 11_218_161,
    url: `${HF}/yolo26n-seg.onnx`,
    inputSize: 640,
    maskCoeffs: 32,
    protoSize: 160,
  },
  pose: {
    task: "pose",
    id: "yolo26n-pose",
    label: "YOLO26n · pose",
    size: 12_125_279,
    url: `${HF}/yolo26n-pose.onnx`,
    inputSize: 640,
  },
};

const CACHE_NAME = "yolo26-models-v1";

export interface DownloadProgress {
  loaded: number;
  total: number;
  /** True when the file was already on the device. */
  cached: boolean;
}

async function openCache(): Promise<Cache | null> {
  if (typeof caches === "undefined") return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

export async function isModelCached(spec: ModelSpec): Promise<boolean> {
  const cache = await openCache();
  if (!cache) return false;
  return Boolean(await cache.match(spec.url));
}

export async function clearModelCache(): Promise<void> {
  if (typeof caches === "undefined") return;
  await caches.delete(CACHE_NAME);
}

/**
 * Returns the model bytes, downloading from the network on first use and
 * storing them in Cache Storage for offline reuse afterwards.
 */
export async function fetchModelBytes(
  spec: ModelSpec,
  onProgress?: (p: DownloadProgress) => void,
): Promise<ArrayBuffer> {
  const cache = await openCache();

  const hit = cache ? await cache.match(spec.url) : undefined;
  if (hit) {
    const buf = await hit.arrayBuffer();
    onProgress?.({ loaded: buf.byteLength, total: buf.byteLength, cached: true });
    return buf;
  }

  const res = await fetch(spec.url, { mode: "cors", credentials: "omit" });
  if (!res.ok) {
    throw new Error(
      `Download failed (${res.status}) for ${spec.id}. Check your connection.`,
    );
  }

  const total = Number(res.headers.get("content-length")) || spec.size;
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  if (res.body) {
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.({ loaded, total, cached: false });
    }
  } else {
    const buf = new Uint8Array(await res.arrayBuffer());
    chunks.push(buf);
    loaded = buf.byteLength;
    onProgress?.({ loaded, total, cached: false });
  }

  const blob = new Blob(chunks as BlobPart[], {
    type: "application/octet-stream",
  });
  const arrayBuffer = await blob.arrayBuffer();

  if (cache) {
    try {
      await cache.put(
        spec.url,
        new Response(blob, { headers: { "content-type": "application/octet-stream" } }),
      );
    } catch {
      /* cache may be full or unavailable; inference still works this session */
    }
  }

  return arrayBuffer;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
