"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  MODELS,
  formatBytes,
  isModelCached,
  clearModelCache,
  type Task,
} from "@/lib/models";
import {
  YoloEngine,
  readBackendPref,
  type Backend,
  type BackendPref,
} from "@/lib/engine";
import { drawOverlay } from "@/lib/draw";
import { classColor } from "@/lib/coco";
import type { Detection } from "@/lib/postprocess";

const TASKS: Task[] = ["seg", "pose", "detect"];

const MODES: { task: Task; icon: string; label: string; blurb: string }[] = [
  { task: "seg", icon: "◧", label: "Segmentation", blurb: "boxes + pixel masks" },
  { task: "pose", icon: "⌇", label: "Pose estimation", blurb: "17 keypoints + skeleton" },
  { task: "detect", icon: "⚠", label: "Defect inspection", blurb: "PCB AOI · 6 defect types" },
];

// Per-mode starting confidence. Defect inspection begins stricter (0.50) so
// only confident fault verdicts surface by default; the COCO modes keep 0.35.
const DEFAULT_CONF: Record<Task, number> = { seg: 0.35, pose: 0.35, detect: 0.5 };

type SampleMode = Task;

const SAMPLES: {
  id: string;
  label: string;
  src: string;
  hint: string;
  modes: SampleMode[];
}[] = [
  { id: "bus", label: "Street scene", src: "/samples/bus.jpg", hint: "seg · pose", modes: ["seg", "pose"] },
  { id: "zidane", label: "Footballers", src: "/samples/zidane.jpg", hint: "pose", modes: ["pose"] },
  { id: "cats", label: "Cats & remotes", src: "/samples/cats.jpg", hint: "segmentation", modes: ["seg"] },
  { id: "pcb-missing-hole", label: "Missing hole", src: "/samples/pcb/missing_hole.jpg", hint: "PCB defect", modes: ["detect"] },
  { id: "pcb-mouse-bite", label: "Mouse bite", src: "/samples/pcb/mouse_bite.jpg", hint: "PCB defect", modes: ["detect"] },
  { id: "pcb-open-circuit", label: "Open circuit", src: "/samples/pcb/open_circuit.jpg", hint: "PCB defect", modes: ["detect"] },
  { id: "pcb-short", label: "Short", src: "/samples/pcb/short.jpg", hint: "PCB defect", modes: ["detect"] },
  { id: "pcb-spur", label: "Spur", src: "/samples/pcb/spur.jpg", hint: "PCB defect", modes: ["detect"] },
  { id: "pcb-spurious-copper", label: "Spurious copper", src: "/samples/pcb/spurious_copper.jpg", hint: "PCB defect", modes: ["detect"] },
];

const DEFAULT_SAMPLE = SAMPLES[0];

interface LoadState {
  state: "idle" | "working" | "ready" | "error";
  pct: number;
  cached: boolean;
  message: string;
}

const idleState = (): LoadState => ({
  state: "idle",
  pct: 0,
  cached: false,
  message: "Loads when selected",
});

interface Stats {
  inferMs: number;
  postMs: number;
  fps: number | null;
}

export default function YoloStudio() {
  const [mode, setMode] = useState<Task>("seg");
  const [backendPref, setBackendPref] = useState<BackendPref>("auto");
  const [sourceType, setSourceType] = useState<"still" | "camera">("still");
  const [sampleId, setSampleId] = useState<string | null>(DEFAULT_SAMPLE.id);
  const [conf, setConf] = useState(DEFAULT_CONF.seg);
  const [maskOpacity, setMaskOpacity] = useState(0.45);
  const [kptThreshold, setKptThreshold] = useState(0.5);

  const [loads, setLoads] = useState<Record<Task, LoadState>>({
    seg: idleState(),
    pose: idleState(),
    detect: idleState(),
  });
  const [backends, setBackends] = useState<Record<Task, Backend | null>>({
    seg: null,
    pose: null,
    detect: null,
  });
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState<Stats>({ inferMs: 0, postMs: 0, fps: null });
  const [imageSize, setImageSize] = useState({ w: 0, h: 0 });
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [engineVersion, setEngineVersion] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [dragging, setDragging] = useState(false);

  // ---- refs (mutable mirrors used by the animation loop) -------------------
  const ui = useRef({ mode, conf, maskOpacity, kptThreshold });
  // Remembers the confidence chosen per mode so switching modes restores it.
  const confByTask = useRef<Record<Task, number>>({ ...DEFAULT_CONF });
  const sourceTypeRef = useRef<"still" | "camera">("still");
  const detsRef = useRef<Detection[]>([]);
  const engines = useRef<Map<Task, YoloEngine>>(new Map());
  const pending = useRef<Map<Task, Promise<YoloEngine>>>(new Map());
  const generation = useRef(0);
  const busyRef = useRef(false);
  const rafRef = useRef<number | null>(null);
  const frameTimes = useRef<number[]>([]);
  const lastListUpdate = useRef(0);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const maskScratch = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const lastStill = useRef({
    src: DEFAULT_SAMPLE.src,
    key: `sample:${DEFAULT_SAMPLE.id}`,
  });

  // ---- model loading -------------------------------------------------------
  const ensureEngine = useCallback(
    async (task: Task): Promise<YoloEngine> => {
      const existing = engines.current.get(task);
      if (existing) return existing;
      const inFlight = pending.current.get(task);
      if (inFlight) return inFlight;

      const gen = generation.current;
      const spec = MODELS[task];
      const setLoad = (next: LoadState) =>
        setLoads((prev) => ({ ...prev, [task]: next }));

      const run = (async () => {
        try {
          setLoad({ state: "working", pct: 0, cached: false, message: "Checking cache…" });
          const cached = await isModelCached(spec);

          const engine = await YoloEngine.load(
            task,
            backendPref,
            (p) => {
              const pct = p.total ? Math.round((p.loaded / p.total) * 100) : 0;
              setLoad({
                state: "working",
                pct,
                cached: p.cached,
                message: p.cached
                  ? "Loading from this device…"
                  : `${formatBytes(p.loaded)} / ${formatBytes(p.total)}`,
              });
            },
          );

          if (gen !== generation.current) {
            engine.dispose();
            throw new Error("cancelled");
          }

          engines.current.set(task, engine);
          setBackends((b) => ({ ...b, [task]: engine.backend }));
          setLoad({
            state: "ready",
            pct: 100,
            cached: true,
            message: `${engine.backend.toUpperCase()} · warm-up ${engine.warmupMs} ms`,
          });
          setEngineVersion((v) => v + 1);
          return engine;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg !== "cancelled") {
            setLoad({ state: "error", pct: 0, cached: false, message: msg });
          }
          throw err;
        } finally {
          pending.current.delete(task);
        }
      })();

      pending.current.set(task, run);
      return run;
    },
    [backendPref],
  );

  // ---- painting ------------------------------------------------------------
  const paintFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (sourceTypeRef.current === "camera") {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.videoWidth) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
    } else {
      const img = imageRef.current;
      if (img && img.complete && img.naturalWidth) {
        if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
        }
        ctx.drawImage(img, 0, 0);
      }
    }

    let scratch = maskScratch.current;
    if (!scratch) {
      scratch = document.createElement("canvas");
      maskScratch.current = scratch;
    }

    drawOverlay(
      ctx,
      detsRef.current,
      canvas.width,
      canvas.height,
      {
        maskOpacity: ui.current.maskOpacity,
        keypointThreshold: ui.current.kptThreshold,
      },
      scratch,
    );
  }, []);

  // ---- single-image analysis ----------------------------------------------
  const analyzeStill = useCallback(async () => {
    const img = imageRef.current;
    if (!img || !img.complete || !img.naturalWidth) return;
    const task = ui.current.mode;
    try {
      const engine = await ensureEngine(task);
      if (imageRef.current !== img || ui.current.mode !== task) return;
      busyRef.current = true;
      setAnalyzing(true);
      const r = await engine.run(
        img,
        img.naturalWidth,
        img.naturalHeight,
        ui.current.conf,
      );
      detsRef.current = r.detections;
      setDetections(r.detections);
      setStats({ inferMs: r.inferMs, postMs: r.postMs, fps: null });
      paintFrame();
    } catch {
      /* the model card already shows the error */
    } finally {
      busyRef.current = false;
      setAnalyzing(false);
    }
  }, [ensureEngine, paintFrame]);

  const reprocess = useCallback(() => {
    if (sourceTypeRef.current === "camera") return;
    const engine = engines.current.get(ui.current.mode);
    if (!engine || !imageRef.current) return;
    const dets = engine.reprocess(ui.current.conf);
    detsRef.current = dets;
    setDetections(dets);
    paintFrame();
  }, [paintFrame]);

  // ---- camera loop ---------------------------------------------------------
  const step = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    paintFrame();
    if (busyRef.current) return;
    const engine = engines.current.get(ui.current.mode);
    if (!engine) return;

    busyRef.current = true;
    const started = performance.now();
    engine
      .run(video, video.videoWidth, video.videoHeight, ui.current.conf)
      .then((r) => {
        const now = performance.now();
        frameTimes.current.push(now - started);
        if (frameTimes.current.length > 24) frameTimes.current.shift();
        detsRef.current = r.detections;
        if (now - lastListUpdate.current > 200) {
          lastListUpdate.current = now;
          const avg =
            frameTimes.current.reduce((a, b) => a + b, 0) / frameTimes.current.length;
          setDetections(r.detections);
          setStats({ inferMs: r.inferMs, postMs: r.postMs, fps: 1000 / avg });
        }
      })
      .catch((err) => console.warn("[yolo] frame failed", err))
      .finally(() => {
        busyRef.current = false;
      });
  }, [paintFrame]);

  const startLoop = useCallback(() => {
    if (rafRef.current != null) return;
    const tick = () => {
      rafRef.current = requestAnimationFrame(tick);
      step();
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [step]);

  const stopLoop = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  // ---- camera lifecycle ----------------------------------------------------
  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("Video element unavailable");
      video.srcObject = stream;
      await video.play();
      sourceTypeRef.current = "camera";
      setSourceType("camera");
      detsRef.current = [];
      setDetections([]);
    } catch (err) {
      setCameraError(
        err instanceof Error
          ? err.message
          : "Camera access was denied or is unavailable.",
      );
    }
  }, []);

  const stopCamera = useCallback(() => {
    stopLoop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
    sourceTypeRef.current = "still";
    setSourceType("still");
    detsRef.current = [];
    setDetections([]);
    requestAnimationFrame(() => {
      paintFrame();
      void analyzeStill();
    });
  }, [analyzeStill, paintFrame, stopLoop]);

  // ---- image sources -------------------------------------------------------
  const loadStill = useCallback(
    (src: string, key: string) => {
      lastStill.current = { src, key };
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        imageRef.current = img;
        setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
        detsRef.current = [];
        setDetections([]);
        paintFrame();
        void analyzeStill();
      };
      img.onerror = () => setCameraError(`Could not load ${src}`);
      img.src = src;
    },
    [analyzeStill, paintFrame],
  );

  const onFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      stopLoop();
      sourceTypeRef.current = "still";
      setSourceType("still");
      setSampleId(null);
      loadStill(url, `upload:${file.name}`);
    },
    [loadStill, stopLoop],
  );

  // ---- effects -------------------------------------------------------------
  // honour ?ep=wasm / ?ep=webgpu on first load
  useEffect(() => {
    const pref = readBackendPref();
    if (pref !== "auto") setBackendPref(pref);
  }, []);

  // start on the default sample
  useEffect(() => {
    loadStill(DEFAULT_SAMPLE.src, `sample:${DEFAULT_SAMPLE.id}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // load only the active model up front; the other one is fetched lazily the
  // first time its mode is selected so we never keep two WebGPU sessions
  // resident in GPU memory at once (re-runs if the backend changes)
  useEffect(() => {
    void ensureEngine(mode).catch(() => {});
  }, [ensureEngine, mode]);

  // keep the mutable mirror in sync with rendered state
  useEffect(() => {
    ui.current = { mode, conf, maskOpacity, kptThreshold };
  }, [mode, conf, maskOpacity, kptThreshold]);

  // re-run a still image whenever the mode or ready engine changes
  useEffect(() => {
    if (sourceType !== "still") return;
    void analyzeStill();
  }, [mode, engineVersion, sourceType, analyzeStill]);

  // re-decode on confidence change (no new inference for still images)
  useEffect(() => {
    reprocess();
  }, [conf, reprocess]);

  // repaint on visual-only changes
  useEffect(() => {
    if (sourceTypeRef.current === "still") paintFrame();
  }, [maskOpacity, kptThreshold, paintFrame]);

  // camera loop follows source/mode/engine
  useEffect(() => {
    if (sourceType !== "camera") {
      stopLoop();
      return;
    }
    let cancelled = false;
    void ensureEngine(mode)
      .then(() => {
        if (!cancelled) startLoop();
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      stopLoop();
    };
  }, [sourceType, mode, engineVersion, ensureEngine, startLoop, stopLoop]);

  // cleanup
  useEffect(
    () => () => {
      stopLoop();
      streamRef.current?.getTracks().forEach((t) => t.stop());
      engines.current.forEach((e) => e.dispose());
      engines.current.clear();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    [stopLoop],
  );

  // ---- handlers ------------------------------------------------------------
  const selectMode = (task: Task) => {
    if (task === mode) return;
    ui.current.mode = task;
    setMode(task);
    detsRef.current = [];
    setDetections([]);
    setStats({ inferMs: 0, postMs: 0, fps: null });
    // Restore this mode's confidence (defect mode defaults to 0.50).
    const nextConf = confByTask.current[task];
    ui.current.conf = nextConf;
    setConf(nextConf);
    void ensureEngine(task).catch(() => {});

    // Don't leave a photo loaded that the new mode can't serve (e.g. a
    // football photo in defect mode). Swap in a sample that fits, unless the
    // user is on their own uploaded image.
    if (sourceTypeRef.current === "still" && sampleId) {
      const current = SAMPLES.find((s) => s.id === sampleId);
      if (!current || !current.modes.includes(task)) {
        const next = SAMPLES.find((s) => s.modes.includes(task));
        if (next) {
          setSampleId(next.id);
          loadStill(next.src, `sample:${next.id}`);
        }
      }
    }
  };

  const selectSample = (sample: (typeof SAMPLES)[number]) => {
    if (sourceTypeRef.current === "camera") stopCamera();
    setSourceType("still");
    sourceTypeRef.current = "still";
    setSampleId(sample.id);
    loadStill(sample.src, `sample:${sample.id}`);
  };

  const toggleCamera = () => {
    if (sourceType === "camera") stopCamera();
    else void startCamera();
  };

  const resetEngines = () => {
    generation.current += 1;
    engines.current.forEach((e) => e.dispose());
    engines.current.clear();
    pending.current.clear();
    setBackends({ seg: null, pose: null, detect: null });
    setLoads({ seg: idleState(), pose: idleState(), detect: idleState() });
  };

  const changeBackend = (pref: BackendPref) => {
    if (pref === backendPref) return;
    resetEngines();
    setBackendPref(pref);
  };

  const wipeCache = async () => {
    await clearModelCache();
    // Force a fresh load of the active model so the new download is used,
    // even when the backend preference is unchanged.
    resetEngines();
    void ensureEngine(ui.current.mode).catch(() => {});
  };

  const activeLoad = loads[mode];
  const activeBackend = backends[mode];
  const showOverlay = activeLoad.state !== "ready";
  const showEmptyHint =
    activeLoad.state === "ready" && detections.length === 0 && !analyzing;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-white/10 bg-[#070a0f]/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-3 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-sky-400 to-violet-500 font-black text-[#05070b]">
              Y
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-tight">YOLO26 Studio</h1>
              <p className="text-[11px] leading-tight text-white/45">
                On-device · seg, pose &amp; defect QA
              </p>
            </div>
          </div>

          <div className="ml-auto flex items-center gap-2 text-[11px] text-white/45">
            <span
              className={
                "rounded-md px-1.5 py-0.5 font-semibold " +
                (activeBackend === "webgpu"
                  ? "bg-emerald-400/15 text-emerald-300"
                  : "bg-amber-400/15 text-amber-300")
              }
            >
              {activeBackend ? activeBackend.toUpperCase() : "…"}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1400px] gap-5 p-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        {/* ---------------- viewport ---------------- */}
        <section className="panel overflow-hidden">
          {/* big, unmissable mode switch */}
          <div className="grid grid-cols-1 gap-2 p-2 sm:grid-cols-3">
            {MODES.map((m) => {
              const load = loads[m.task];
              const active = mode === m.task;
              return (
                <button
                  key={m.task}
                  onClick={() => selectMode(m.task)}
                  aria-pressed={active}
                  className={
                    "flex items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition " +
                    (active
                      ? "border-transparent bg-gradient-to-r from-sky-400/90 to-violet-400/90 text-[#05070b]"
                      : "border-white/10 bg-white/[0.03] text-white/80 hover:bg-white/[0.07]")
                  }
                >
                  <span className="text-xl leading-none" aria-hidden>
                    {m.icon}
                  </span>
                  <span className="min-w-0">
                    <span className="block text-sm font-semibold">{m.label}</span>
                    <span
                      className={
                        "block truncate text-[11px] " +
                        (active ? "text-[#05070b]/70" : "text-white/45")
                      }
                    >
                      {m.blurb}
                    </span>
                  </span>
                  <ModeBadge load={load} active={active} />
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-2 border-y border-white/10 px-3 py-2">
            <span className="text-[11px] font-medium uppercase tracking-wider text-white/40">
              Source
            </span>
            <label className="cursor-pointer rounded-lg border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs hover:bg-white/[0.08]">
              Upload image
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onFile(f);
                  e.target.value = "";
                }}
              />
            </label>
            <button
              onClick={toggleCamera}
              className={
                "rounded-lg border px-3 py-1.5 text-xs transition " +
                (sourceType === "camera"
                  ? "border-rose-400/40 bg-rose-500/20 text-rose-200 hover:bg-rose-500/30"
                  : "border-white/10 bg-white/[0.04] hover:bg-white/[0.08]")
              }
            >
              {sourceType === "camera" ? "◼ Stop camera" : "● Live camera"}
            </button>

            {imageSize.w > 0 && (
              <span className="ml-auto text-[11px] tabular-nums text-white/45">
                {imageSize.w}×{imageSize.h}
              </span>
            )}
          </div>

          <div
            className={"relative checker " + (dragging ? "ring-2 ring-sky-400/70" : "")}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) onFile(f);
            }}
          >
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video ref={videoRef} playsInline muted className="hidden" />
            <canvas ref={canvasRef} className="block h-auto w-full" />

            {showOverlay && (
              <div className="absolute inset-0 grid place-items-center bg-[#070a0f]/70 backdrop-blur-sm">
                <div className="w-[min(380px,86%)] rounded-2xl border border-white/10 bg-[#0b1220]/90 p-5">
                  <div className="mb-2 flex items-baseline justify-between">
                    <span className="text-sm font-semibold">
                      Preparing {MODELS[mode].id}
                    </span>
                    <span className="text-xs tabular-nums text-white/50">
                      {activeLoad.pct}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-sky-400 to-violet-400 transition-all"
                      style={{ width: `${activeLoad.pct}%` }}
                    />
                  </div>
                  <p className="mt-3 text-xs leading-relaxed text-white/50">
                    {activeLoad.state === "error"
                      ? activeLoad.message
                      : `Downloading / loading the model — ${activeLoad.message}`}
                  </p>
                  <p className="mt-1 text-[11px] leading-relaxed text-white/35">
                    Models load on demand and are cached on this device for
                    offline use.
                  </p>
                </div>
              </div>
            )}

            {cameraError && (
              <div className="absolute inset-x-0 bottom-0 bg-rose-950/80 px-4 py-3 text-xs text-rose-200">
                Camera: {cameraError}
              </div>
            )}
          </div>

          {showEmptyHint && (
            <div
              className={
                "border-t px-4 py-2 text-xs " +
                (mode === "detect"
                  ? "border-white/10 bg-emerald-400/10 text-emerald-200"
                  : "border-white/10 bg-amber-400/10 text-amber-200")
              }
            >
              {mode === "pose" ? (
                <>
                  Pose only detects people.{" "}
                  <button
                    className="underline"
                    onClick={() =>
                      selectSample(SAMPLES.find((s) => s.id === "zidane")!)
                    }
                  >
                    Try the Footballers sample
                  </button>
                  .
                </>
              ) : mode === "detect" ? (
                <>
                  No defects above {conf.toFixed(2)} confidence — this board reads
                  clean. Lower the threshold to catch fainter defects.
                </>
              ) : (
                <>
                  Nothing above {conf.toFixed(2)} confidence. Lower the threshold or
                  try another image.
                </>
              )}
            </div>
          )}

          <div className="grid grid-cols-2 gap-x-4 gap-y-3 border-t border-white/10 px-4 py-3 sm:grid-cols-4">
            <Metric
              label="Inference"
              value={stats.inferMs ? `${stats.inferMs.toFixed(0)} ms` : "—"}
            />
            <Metric
              label="Post-process"
              value={stats.postMs ? `${stats.postMs.toFixed(1)} ms` : "—"}
            />
            <Metric
              label={
                sourceType === "camera"
                  ? "FPS"
                  : mode === "detect"
                    ? "Defects"
                    : "Objects"
              }
              value={
                sourceType === "camera"
                  ? stats.fps
                    ? stats.fps.toFixed(1)
                    : "—"
                  : String(detections.length)
              }
            />
            <Metric
              label="Compute"
              value={activeBackend ? activeBackend.toUpperCase() : "—"}
            />
          </div>
        </section>

        {/* ---------------- sidebar ---------------- */}
        <aside className="flex flex-col gap-4">
          {/* models */}
          <div className="panel p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">Models</h2>
              <button
                onClick={wipeCache}
                className="text-[11px] text-white/40 underline-offset-2 hover:text-white/70 hover:underline"
              >
                re-download
              </button>
            </div>

            <div className="space-y-2">
              {TASKS.map((t) => {
                const load = loads[t];
                return (
                  <div
                    key={t}
                    className="rounded-xl border border-white/10 bg-white/[0.03] p-3"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-white/90">
                        {MODELS[t].id}
                      </span>
                      <span className="text-white/45">
                        {formatBytes(MODELS[t].size)}
                      </span>
                    </div>
                    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                      <div
                        className={
                          "h-full rounded-full transition-all " +
                          (load.state === "error"
                            ? "bg-rose-400"
                            : load.state === "ready"
                              ? "bg-emerald-400"
                              : "bg-gradient-to-r from-sky-400 to-violet-400")
                        }
                        style={{ width: `${load.state === "ready" ? 100 : load.pct}%` }}
                      />
                    </div>
                    <div className="mt-1.5 flex items-center gap-2 text-[11px]">
                      <span
                        className={
                          "rounded-md px-1.5 py-0.5 font-medium " +
                          (load.state === "ready"
                            ? "bg-emerald-400/15 text-emerald-300"
                            : load.state === "error"
                              ? "bg-rose-400/15 text-rose-300"
                              : "bg-white/10 text-white/50")
                        }
                      >
                        {load.state === "ready"
                          ? load.cached
                            ? "on device"
                            : "loaded"
                          : load.state === "error"
                            ? "error"
                            : load.state === "idle"
                              ? "idle"
                              : `${load.pct}%`}
                      </span>
                      <span className="truncate text-white/40">{load.message}</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-3">
              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-white/40">
                Compute
              </div>
              <div className="flex gap-1 rounded-lg border border-white/10 bg-white/[0.03] p-1">
                {(["auto", "webgpu", "wasm"] as BackendPref[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => changeBackend(p)}
                    className={
                      "flex-1 rounded-md px-2 py-1 text-[11px] font-medium transition " +
                      (backendPref === p
                        ? "bg-white/15 text-white"
                        : "text-white/50 hover:text-white")
                    }
                  >
                    {p === "auto" ? "Auto" : p.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* samples */}
          <div className="panel p-4">
            <h2 className="mb-3 text-sm font-semibold">Sample images</h2>
            <div className="grid grid-cols-3 gap-2">
              {SAMPLES.filter((s) => s.modes.includes(mode)).map((s) => (
                <button
                  key={s.id}
                  onClick={() => selectSample(s)}
                  className={
                    "group relative overflow-hidden rounded-lg border transition " +
                    (sampleId === s.id
                      ? "border-sky-400/70"
                      : "border-white/10 hover:border-white/30")
                  }
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={s.src}
                    alt={s.label}
                    className="aspect-square w-full object-cover opacity-90 group-hover:opacity-100"
                  />
                  <span className="absolute inset-x-0 bottom-0 bg-black/65 px-1.5 py-1 text-left text-[10px] leading-tight">
                    {s.label}
                    <span className="block text-white/50">{s.hint}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* controls */}
          <div className="panel space-y-4 p-4">
            <h2 className="text-sm font-semibold">Controls</h2>
            <Slider
              label="Confidence"
              value={conf}
              min={0.05}
              max={0.9}
              step={0.01}
              display={conf.toFixed(2)}
              onChange={(v) => {
                ui.current.conf = v;
                confByTask.current[mode] = v;
                setConf(v);
              }}
            />
            {mode === "seg" ? (
              <Slider
                label="Mask opacity"
                value={maskOpacity}
                min={0.1}
                max={0.9}
                step={0.05}
                display={maskOpacity.toFixed(2)}
                onChange={(v) => {
                  ui.current.maskOpacity = v;
                  setMaskOpacity(v);
                }}
              />
            ) : mode === "pose" ? (
              <Slider
                label="Keypoint threshold"
                value={kptThreshold}
                min={0.1}
                max={0.9}
                step={0.05}
                display={kptThreshold.toFixed(2)}
                onChange={(v) => {
                  ui.current.kptThreshold = v;
                  setKptThreshold(v);
                }}
              />
            ) : (
              <p className="text-xs leading-relaxed text-white/45">
                Every detection is a manufacturing defect. Lower the confidence
                threshold to surface fainter faults; the head is NMS-free, so
                each listed box is already a final verdict.
              </p>
            )}
          </div>

          {/* detections */}
          <div className="panel p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold">Detections</h2>
              <span className="text-[11px] text-white/40">
                {analyzing ? "analysing…" : `${detections.length} found`}
              </span>
            </div>
            {detections.length === 0 ? (
              <p className="text-xs leading-relaxed text-white/40">
                {mode === "pose"
                  ? "No people detected in this image."
                  : mode === "detect"
                    ? `No defects found — the board passes at ${conf.toFixed(2)}.`
                    : `Nothing above ${conf.toFixed(2)} confidence.`}
              </p>
            ) : (
              <ul className="max-h-[280px] space-y-1.5 overflow-y-auto pr-1">
                {detections.slice(0, 40).map((d, i) => (
                  <li
                    key={`${d.classId}-${i}`}
                    className="flex items-center gap-2 rounded-lg bg-white/[0.03] px-2.5 py-1.5 text-xs"
                  >
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ background: classColor(d.classId) }}
                    />
                    <span className="flex-1 truncate capitalize">{d.className}</span>
                    {d.task === "pose" && (
                      <span className="text-[10px] text-white/35">
                        {d.keypoints.filter((k) => k.score > kptThreshold).length} kp
                      </span>
                    )}
                    <span className="tabular-nums text-white/50">
                      {(d.score * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </aside>
      </div>

      <footer className="mx-auto max-w-[1400px] px-4 pb-8 pt-2 text-[11px] leading-relaxed text-white/35">
        <p>
          Weights: Ultralytics YOLO26n (AGPL-3.0), exported to ONNX by the community
          and fetched from Hugging Face. Inference runs locally with ONNX Runtime Web —
          no image ever leaves your device.
        </p>
        <p className="mt-1.5">
          Defect inspection uses the YOLO26 PCB detector by{" "}
          <a
            href="https://huggingface.co/betty0/pcb-defect-detection"
            target="_blank"
            rel="noreferrer"
            className="underline underline-offset-2 hover:text-white/60"
          >
            betty0
          </a>{" "}
          (AGPL-3.0), fine-tuned on the HRIPCB / PKU-Market-PCB dataset (Huang &amp;
          Wei, arXiv:1901.08204). Sample board images courtesy of that model repo.
        </p>
      </footer>
    </div>
  );
}

function ModeBadge({ load, active }: { load: LoadState; active: boolean }) {
  if (load.state === "ready") {
    return (
      <span className="ml-auto shrink-0 text-sm" title="Model ready">
        {active ? "✓" : "●"}
      </span>
    );
  }
  if (load.state === "error") {
    return <span className="ml-auto shrink-0 text-sm text-rose-400">▲</span>;
  }
  return (
    <span className="ml-auto shrink-0 text-[11px] tabular-nums opacity-70">
      {load.state === "working"
        ? `${load.pct}%`
        : load.state === "idle"
          ? "○"
          : "…"}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] font-medium uppercase tracking-wider text-white/35">
        {label}
      </div>
      <div className="text-sm tabular-nums text-white/90">{value}</div>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="text-white/70">{label}</span>
        <span className="tabular-nums text-white/40">{display}</span>
      </div>
      <input
        type="range"
        className="w-full"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
