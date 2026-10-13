/**
 * Inference engine wrapper around ONNX Runtime Web.
 *
 * Important: the ONNX Runtime runtime is loaded at *runtime* from
 * `/ort/ort.webgpu.bundle.min.mjs` (a plain module script) instead of being
 * imported through the bundler. Bundled ORT builds mis-handle their worker
 * bootstrap and can wedge the WebGPU execution provider; loading the exact
 * upstream artifact avoids that entirely and keeps the app offline-capable.
 *
 * Backend: WebGPU when available, otherwise multi-threaded WASM. Both are
 * validated with a warm-up run; a failing WebGPU session falls back to WASM.
 */

import type { InferenceSession } from "onnxruntime-web";
import {
  MODELS,
  fetchModelBytes,
  type DownloadProgress,
  type ModelSpec,
  type Task,
} from "./models";
import { INPUT_SIZE, letterbox, type LetterboxMeta } from "./preprocess";
import { decodePose, decodeSegmentation, type Detection } from "./postprocess";

type Ort = typeof import("onnxruntime-web");

export type Backend = "webgpu" | "wasm";
export type BackendPref = "auto" | Backend;

export interface RunResult {
  detections: Detection[];
  meta: LetterboxMeta;
  inferMs: number;
  postMs: number;
}

const ORT_URL = "/ort/ort.webgpu.bundle.min.mjs";

declare global {
  interface Window {
    __ORT__?: Ort;
  }
}

let ortPromise: Promise<Ort> | null = null;

function loadOrt(): Promise<Ort> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("ONNX Runtime can only load in the browser"));
  }
  if (window.__ORT__) return Promise.resolve(window.__ORT__);
  if (ortPromise) return ortPromise;

  ortPromise = new Promise<Ort>((resolve, reject) => {
    const onReady = () => {
      if (window.__ORT__) resolve(window.__ORT__);
      else reject(new Error("ONNX Runtime failed to initialise"));
    };
    window.addEventListener("ort-ready", onReady, { once: true });

    const script = document.createElement("script");
    script.type = "module";
    script.textContent = [
      `import * as ort from "${ORT_URL}";`,
      `ort.env.wasm.wasmPaths = "/ort/";`,
      `ort.env.logLevel = "error";`,
      `window.__ORT__ = ort;`,
      `window.dispatchEvent(new Event("ort-ready"));`,
    ].join("");
    script.onerror = () =>
      reject(new Error("Could not load the ONNX Runtime runtime files."));
    document.head.appendChild(script);
  });

  return ortPromise;
}

export function isWebGpuAvailable(): boolean {
  return (
    typeof navigator !== "undefined" &&
    "gpu" in navigator &&
    Boolean((navigator as Navigator & { gpu?: unknown }).gpu)
  );
}

export function readBackendPref(): BackendPref {
  if (typeof window === "undefined") return "auto";
  const v = new URLSearchParams(window.location.search).get("ep");
  return v === "wasm" || v === "webgpu" ? v : "auto";
}

function providersFor(pref: BackendPref): string[] {
  if (pref === "wasm") return ["wasm"];
  if (pref === "webgpu") return ["webgpu", "wasm"];
  return isWebGpuAvailable() ? ["webgpu", "wasm"] : ["wasm"];
}

/**
 * ONNX Runtime Web's WebGPU execution provider can only register one device
 * at a time. Creating two sessions concurrently throws
 * "another WebGPU EP inference session is being created." — the window is
 * wider on slower machines, which is why it only bites some laptops. We funnel
 * every session creation through a single promise chain so they run one at a
 * time (downloads still happen in parallel).
 */
let createQueue: Promise<unknown> = Promise.resolve();

function serializeSessionCreate<T>(run: () => Promise<T>): Promise<T> {
  const result = createQueue.then(run, run);
  // Keep the chain alive even when a create rejects.
  createQueue = result.then(
    () => undefined,
    () => undefined,
  );
  return result;
}

async function createSession(
  ort: Ort,
  bytes: ArrayBuffer,
  pref: BackendPref,
): Promise<{ session: InferenceSession; backend: Backend }> {
  const providers = providersFor(pref);
  const session = await serializeSessionCreate(() =>
    ort.InferenceSession.create(bytes, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
    }),
  );
  return { session, backend: providers[0] === "webgpu" ? "webgpu" : "wasm" };
}

export class YoloEngine {
  readonly task: Task;
  readonly spec: ModelSpec;
  backend: Backend;
  /** Wall-clock ms of the first (shader-compiling) run, for the UI. */
  warmupMs = 0;

  private session: InferenceSession;
  private bytes: ArrayBuffer;
  private pref: BackendPref;
  private inputName: string;
  private out0Name: string;
  private protoName?: string;
  private raw: {
    out0: Float32Array;
    protos?: Float32Array;
    meta: LetterboxMeta;
  } | null = null;

  private constructor(
    task: Task,
    spec: ModelSpec,
    session: InferenceSession,
    bytes: ArrayBuffer,
    backend: Backend,
    pref: BackendPref,
  ) {
    this.task = task;
    this.spec = spec;
    this.session = session;
    this.bytes = bytes;
    this.backend = backend;
    this.pref = pref;
    this.inputName = session.inputNames[0];
    this.out0Name = session.outputNames[0];
    this.protoName = session.outputNames[1];
  }

  static async load(
    task: Task,
    pref: BackendPref = "auto",
    onProgress?: (p: DownloadProgress) => void,
  ): Promise<YoloEngine> {
    const spec = MODELS[task];
    const [bytes, ort] = await Promise.all([
      fetchModelBytes(spec, onProgress),
      loadOrt(),
    ]);
    let created = await createSession(ort, bytes, pref);

    // Warm up so shader compilation happens now, and so a broken GPU is
    // detected before it reaches the user.
    const engine = new YoloEngine(
      task,
      spec,
      created.session,
      bytes,
      created.backend,
      pref,
    );
    const ok = await engine.warmUp();
    if (!ok && created.backend === "webgpu") {
      console.warn("[yolo] WebGPU warm-up failed, switching to WASM");
      try {
        created.session.release?.();
      } catch {
        /* ignore */
      }
      created = await createSession(ort, bytes, "wasm");
      const wasmEngine = new YoloEngine(
        task,
        spec,
        created.session,
        bytes,
        "wasm",
        "wasm",
      );
      await wasmEngine.warmUp();
      return wasmEngine;
    }

    return engine;
  }

  private async warmUp(): Promise<boolean> {
    const ort = await loadOrt();
    const input = new ort.Tensor(
      "float32",
      new Float32Array(3 * INPUT_SIZE * INPUT_SIZE),
      [1, 3, INPUT_SIZE, INPUT_SIZE],
    );
    const started = performance.now();
    try {
      await withTimeout(
        this.session.run({ [this.inputName]: input }),
        20000,
      );
      this.warmupMs = Math.round(performance.now() - started);
      return true;
    } catch (err) {
      console.warn("[yolo] warm-up failed", err);
      return false;
    }
  }

  private decode(confidence: number): Detection[] {
    if (!this.raw) return [];
    const { out0, protos, meta } = this.raw;
    if (this.task === "seg") {
      if (!protos) return [];
      return decodeSegmentation(
        out0,
        protos,
        meta,
        confidence,
        this.spec.maskCoeffs ?? 32,
        this.spec.protoSize ?? 160,
        true,
      );
    }
    return decodePose(out0, meta, confidence);
  }

  /** Runs the model on one frame and returns decoded detections. */
  async run(
    source: CanvasImageSource,
    width: number,
    height: number,
    confidence: number,
  ): Promise<RunResult> {
    const { data, meta } = letterbox(source, width, height);
    const ort = await loadOrt();
    const input = new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const t0 = performance.now();
    let results: Record<string, { data: unknown }>;
    try {
      results = (await this.session.run({
        [this.inputName]: input,
      })) as unknown as Record<string, { data: unknown }>;
    } catch (err) {
      if (this.backend === "webgpu") {
        console.warn("[yolo] WebGPU inference failed, retrying on WASM", err);
        await this.rebuild("wasm");
        return this.run(source, width, height, confidence);
      }
      throw err;
    }
    const inferMs = performance.now() - t0;

    const out0 = results[this.out0Name].data as Float32Array;
    const protos = this.protoName
      ? (results[this.protoName].data as Float32Array)
      : undefined;
    this.raw = { out0, protos, meta };

    const t1 = performance.now();
    const detections = this.decode(confidence);
    const postMs = performance.now() - t1;

    return { detections, meta, inferMs, postMs };
  }

  /** Re-decodes cached raw outputs (e.g. after a threshold change). */
  reprocess(confidence: number): Detection[] {
    return this.decode(confidence);
  }

  private async rebuild(backend: Backend) {
    const ort = await loadOrt();
    try {
      this.session.release?.();
    } catch {
      /* ignore */
    }
    this.session = await serializeSessionCreate(() =>
      ort.InferenceSession.create(this.bytes, {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
      }),
    );
    this.backend = backend;
    this.pref = backend;
    this.raw = null;
  }

  dispose() {
    try {
      this.session.release?.();
    } catch {
      /* ignore */
    }
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("timed out")), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}
