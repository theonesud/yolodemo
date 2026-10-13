/** @type {import('next').NextConfig} */
const nextConfig = {
  // Cross-origin isolation lets ONNX Runtime Web use multi-threaded WASM,
  // which is a big speed-up for the fallback backend on laptops and phones.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
