import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Pin workspace root — without this, Turbopack walks up to the nearest
  // package-lock.json (often outside the repo) and fans out workers
  // across that tree.
  turbopack: {
    root: path.join(import.meta.dirname),
  },
};

export default nextConfig;
