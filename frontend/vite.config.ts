import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Dev server proxies the API and WebSocket to the FastAPI backend so the app
 * runs same-origin in development. That keeps CORS out of the local loop and
 * means the production build works unchanged behind any reverse proxy.
 *
 * Override the target with VITE_API_TARGET if the backend is elsewhere.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8008'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/ws': { target, ws: true, changeOrigin: true },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: mode !== 'production',
      chunkSizeWarningLimit: 700,
    },
  }
})
