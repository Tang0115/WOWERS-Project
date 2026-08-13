import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      // Allow serving files from the repo root (one level above frontend/).
      // Required so `import sitesUrl from "../../../exports/viable_sites.geojson?url"`
      // resolves correctly in dev mode.
      allow: [".."],
    },
  },
})
