// Copies the ONNX Runtime Web WASM/JS runtime assets into /public/ort so the
// inference engine runs from our own origin. This keeps the app fully offline
// after the first load and avoids version drift with the JS bundle.
import { cp, mkdir, readdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const distDir = path.join(root, "node_modules", "onnxruntime-web", "dist");
const outDir = path.join(root, "public", "ort");

if (!existsSync(distDir)) {
  console.warn("[copy-ort-wasm] onnxruntime-web/dist not found, skipping.");
  process.exit(0);
}

await rm(outDir, { recursive: true, force: true });
await mkdir(outDir, { recursive: true });

const files = await readdir(distDir);
// The wasm runtimes (loaded by the ORT glue) plus the standalone JS bundles
// that we import at runtime from /ort/ instead of going through the bundler.
const wanted = files.filter(
  (f) =>
    /\.(wasm|mjs)$/.test(f) &&
    (/ort-wasm/.test(f) || /^ort\.(webgpu|wasm)\.bundle\.min\.mjs$/.test(f)),
);

for (const file of wanted) {
  await cp(path.join(distDir, file), path.join(outDir, file));
}

console.log(`[copy-ort-wasm] copied ${wanted.length} runtime file(s) to public/ort`);
