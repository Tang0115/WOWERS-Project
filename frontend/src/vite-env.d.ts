/// <reference types="vite/client" />

/**
 * Vite ?url import for any GeoJSON file.
 * Usage: import sitesUrl from "path/to/file.geojson?url";
 * At build time Vite copies the file to dist/assets/ with a content hash.
 * In dev mode it serves it via /@fs/ (requires server.fs.allow: [".."] in vite.config.ts).
 */
declare module "*.geojson?url" {
  const value: string;
  export default value;
}
