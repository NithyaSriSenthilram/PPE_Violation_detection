/**
 * API client.
 *
 * One place that knows about transport concerns: the base URL, the optional
 * API key header, and how the backend reports errors. Every call surfaces a
 * typed `ApiError` carrying the server's own message, so the UI can show the
 * real reason a request failed instead of "something went wrong".
 */

import type {
  Camera,
  CameraRuntime,
  Diagnostics,
  DetectionProfile,
  EventPage,
  EventStatus,
  EvidenceManifest,
  Health,
  SecurityEvent,
  Statistics,
  VideoJob,
  VideoJobResult,
  Zone,
} from './types'

/**
 * Backend origin. Empty means same-origin: the Vite dev server proxies /api
 * and /ws, and a production build served behind a reverse proxy works
 * unchanged. When the frontend is hosted separately from the API (Render
 * Static Site + Web Service), set VITE_API_URL to the API origin at build
 * time, e.g. `https://ppe-detection-api.onrender.com`. VITE_API_BASE is the
 * older name and still honoured.
 */
const BASE = (import.meta.env.VITE_API_URL ?? import.meta.env.VITE_API_BASE ?? '')
  .trim()
  .replace(/\/+$/, '')
/** Optional explicit WebSocket origin; derived from BASE when unset. */
const WS_BASE = (import.meta.env.VITE_WS_URL ?? '').trim().replace(/\/+$/, '')
const API_KEY = import.meta.env.VITE_API_KEY ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
    readonly requestId?: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

function headers(json = true): HeadersInit {
  const h: Record<string, string> = {}
  if (json) h['Content-Type'] = 'application/json'
  if (API_KEY) h['X-API-Key'] = API_KEY
  return h
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}/api${path}`, init)
  } catch (cause) {
    // Network-level failure: the server is down or unreachable. Say that,
    // rather than letting a TypeError bubble up as a blank screen.
    throw new ApiError(
      'Cannot reach the SentinelVision backend. Check that it is running.',
      0,
      cause,
    )
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  let body: unknown = text
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      /* keep raw text */
    }
  }

  if (!response.ok) {
    const record = (body ?? {}) as Record<string, unknown>
    const detail = record.detail
    const message =
      (typeof record.error === 'string' && record.error) ||
      (typeof detail === 'string' && detail) ||
      `Request failed (${response.status})`
    throw new ApiError(
      message,
      response.status,
      detail,
      typeof record.request_id === 'string' ? record.request_id : undefined,
    )
  }

  return body as T
}

function query(params: Record<string, unknown>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

/* ── System ──────────────────────────────────────────────────────────────── */
export const api = {
  health: () => request<Health>('/health'),
  diagnostics: () => request<Diagnostics>('/diagnostics'),

  /* ── Cameras ───────────────────────────────────────────────────────────── */
  cameras: () => request<Camera[]>('/cameras'),
  camera: (id: string) => request<Camera>(`/cameras/${id}`),
  createCamera: (body: Partial<Camera>) =>
    request<Camera>('/cameras', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify(body),
    }),
  updateCamera: (id: string, body: Partial<Camera>) =>
    request<Camera>(`/cameras/${id}`, {
      method: 'PUT',
      headers: headers(),
      body: JSON.stringify(body),
    }),
  deleteCamera: (id: string) =>
    request<void>(`/cameras/${id}`, { method: 'DELETE', headers: headers(false) }),
  startCamera: (id: string) =>
    request<{ started: boolean; message: string }>(`/cameras/${id}/start`, {
      method: 'POST',
      headers: headers(false),
    }),
  stopCamera: (id: string) =>
    request<{ stopped: boolean }>(`/cameras/${id}/stop`, {
      method: 'POST',
      headers: headers(false),
    }),
  cameraRuntime: (id: string) => request<CameraRuntime>(`/cameras/${id}/runtime`),
  /** MJPEG preview URL — set as an <img> src. */
  streamUrl: (id: string, width = 640, fps = 8) =>
    `${BASE}/api/cameras/${id}/stream${query({ width, fps })}`,

  /* ── Zones ─────────────────────────────────────────────────────────────── */
  zones: (cameraId: string) => request<Zone[]>(`/zones/${cameraId}`),
  createZone: (body: Partial<Zone> & { camera_id: string }) =>
    request<Zone>('/zones', {
      method: 'POST',
      headers: headers(),
      body: JSON.stringify(body),
    }),
  updateZone: (id: string, body: Partial<Zone>) =>
    request<Zone>(`/zones/detail/${id}`, {
      method: 'PUT',
      headers: headers(),
      body: JSON.stringify(body),
    }),
  deleteZone: (id: string) =>
    request<void>(`/zones/detail/${id}`, {
      method: 'DELETE',
      headers: headers(false),
    }),

  /* ── Events & alerts ───────────────────────────────────────────────────── */
  events: (params: Record<string, unknown> = {}) =>
    request<EventPage>(`/events${query(params)}`),
  event: (id: string) => request<SecurityEvent>(`/events/${id}`),
  alerts: (params: Record<string, unknown> = {}) =>
    request<EventPage>(`/alerts${query(params)}`),
  setEventStatus: (id: string, status: EventStatus, notes?: string) =>
    request<SecurityEvent>(`/events/${id}/status`, {
      method: 'PATCH',
      headers: headers(),
      body: JSON.stringify({ status, notes }),
    }),

  /* ── Analytics ─────────────────────────────────────────────────────────── */
  statistics: (params: Record<string, unknown> = {}) =>
    request<Statistics>(`/statistics${query(params)}`),

  /* ── Evidence ──────────────────────────────────────────────────────────── */
  evidence: (eventId: string) =>
    request<EvidenceManifest>(`/evidence/${eventId}`),
  snapshotUrl: (eventId: string) => `${BASE}/api/evidence/${eventId}/snapshot`,
  clipUrl: (eventId: string) => `${BASE}/api/evidence/${eventId}/clip`,
  exportUrl: (eventId: string) => `${BASE}/api/evidence/${eventId}/export`,

  /* ── Video analysis ────────────────────────────────────────────────────── */
  profiles: () =>
    request<{ profiles: DetectionProfile[]; default: string }>(
      '/videos/profiles',
    ),
  jobs: () => request<VideoJob[]>('/videos/jobs'),
  job: (id: string) => request<VideoJob>(`/videos/jobs/${id}`),
  jobResult: (id: string) => request<VideoJobResult>(`/videos/jobs/${id}/result`),
  /**
   * Absolute URLs so the <video> element and download links work regardless
   * of where the dev server is proxying from.
   */
  annotatedVideoUrl: (id: string) => `${BASE}/api/videos/jobs/${id}/video`,
  annotatedDownloadUrl: (id: string) => `${BASE}/api/videos/jobs/${id}/download`,
  originalDownloadUrl: (id: string) => `${BASE}/api/videos/jobs/${id}/original`,
  cancelJob: (id: string) =>
    request<{ cancelled: boolean }>(`/videos/jobs/${id}/cancel`, {
      method: 'POST',
      headers: headers(false),
    }),
  deleteJob: (id: string) =>
    request<void>(`/videos/jobs/${id}`, { method: 'DELETE', headers: headers(false) }),

  /**
   * Upload with progress. XHR rather than fetch because fetch still cannot
   * report upload progress, and a surveillance clip is large enough that a
   * progress bar is the difference between "working" and "frozen".
   */
  uploadVideo(
    file: File,
    profile: string,
    onProgress?: (fraction: number) => void,
    signal?: AbortSignal,
  ): Promise<{ job_id: string; filename: string; message: string }> {
    return new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      form.append('profile', profile)

      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/api/videos/upload`)
      if (API_KEY) xhr.setRequestHeader('X-API-Key', API_KEY)

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
      }
      xhr.onload = () => {
        let body: Record<string, unknown> = {}
        try {
          body = JSON.parse(xhr.responseText)
        } catch {
          /* ignore */
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body as never)
        } else {
          const detail = body.detail
          reject(
            new ApiError(
              (typeof body.error === 'string' && body.error) ||
                (typeof detail === 'string' && detail) ||
                `Upload failed (${xhr.status})`,
              xhr.status,
              detail,
            ),
          )
        }
      }
      xhr.onerror = () =>
        reject(new ApiError('Upload failed — the backend is unreachable.', 0))
      xhr.onabort = () => reject(new ApiError('Upload cancelled.', 0))
      signal?.addEventListener('abort', () => xhr.abort())
      xhr.send(form)
    })
  },
}

/** WebSocket URL for the realtime feed. */
export function websocketUrl(topics?: string[], cameras?: string[]): string {
  const origin = WS_BASE || BASE || window.location.origin
  const url = new URL('/ws', origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  if (topics?.length) url.searchParams.set('topics', topics.join(','))
  if (cameras?.length) url.searchParams.set('cameras', cameras.join(','))
  if (API_KEY) url.searchParams.set('api_key', API_KEY)
  return url.toString()
}
