/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Backend origin for a separately hosted API (build-time), e.g.
   * `https://ppe-detection-api.onrender.com`. Empty means same-origin.
   */
  readonly VITE_API_URL?: string
  /** Older name for VITE_API_URL; still honoured as a fallback. */
  readonly VITE_API_BASE?: string
  /** Explicit WebSocket origin; derived from VITE_API_URL when unset. */
  readonly VITE_WS_URL?: string
  /** Shared secret, required only when the backend sets API_KEY. */
  readonly VITE_API_KEY?: string
  /** Dev-server proxy target for /api and /ws. */
  readonly VITE_API_TARGET?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
