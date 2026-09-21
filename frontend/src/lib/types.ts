/**
 * Types mirroring the backend Pydantic schemas (backend/schemas.py).
 * Kept hand-written and minimal rather than generated: the surface is small,
 * and an explicit contract makes a backend change fail at compile time here.
 */

export type Severity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
export type EventStatus = 'OPEN' | 'ACKNOWLEDGED' | 'RESOLVED' | 'DISMISSED'
export type SourceType = 'rtsp' | 'webcam' | 'file' | 'http'
export type ZoneType = 'restricted' | 'monitored' | 'crowd' | 'loitering'
export type CameraStatus =
  | 'idle'
  | 'connecting'
  | 'online'
  | 'offline'
  | 'error'
  | 'ended'

export type EventType =
  | 'PPE_VIOLATION'
  | 'MISSING_HELMET'
  | 'MISSING_VEST'
  | 'RESTRICTED_AREA'
  | 'LOITERING'
  | 'ABNORMAL_MOVEMENT'
  | 'POSSIBLE_FALL'
  | 'CROWD_ANOMALY'

export interface Camera {
  camera_id: string
  name: string
  location: string
  source_type: SourceType
  stream_url: string
  status: CameraStatus
  enabled: boolean
  analytics_enabled: boolean
  settings_json: Record<string, unknown>
  last_frame_at: string | null
  last_error: string | null
  fps: number
  people_count: number
  created_at: string
  updated_at: string
  zone_count: number
}

export interface CameraRuntime {
  camera_id: string
  status: string
  fps: number
  people_count: number
  active_tracks: number
  inference_ms: number
  processing_ms: number
  frames_processed: number
  frames_dropped: number
  backend: string | null
  last_error: string | null
  uptime_seconds: number
}

export interface Zone {
  zone_id: string
  camera_id: string
  name: string
  zone_type: ZoneType
  /** Normalised 0..1 points: [[x, y], ...] */
  polygon: [number, number][]
  enabled: boolean
  loitering_threshold: number | null
  crowd_threshold: number | null
  colour: string
  created_at: string
  updated_at: string
}

export interface SecurityEvent {
  event_id: string
  event_type: EventType | string
  camera_id: string | null
  person_id: number | null
  zone_id: string | null
  severity: Severity
  confidence: number
  status: EventStatus
  description: string
  bbox: number[] | null
  snapshot_path: string | null
  clip_path: string | null
  detection_metadata: Record<string, unknown>
  job_id: string | null
  video_timestamp: number | null
  timestamp: string
  acknowledged_at: string | null
  resolved_at: string | null
  notes: string | null
  camera_name: string | null
  zone_name: string | null
  /** Server-computed */
  label: string
  advisory: boolean
  has_snapshot: boolean
  has_clip: boolean
}

export interface EventPage {
  items: SecurityEvent[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface Kpis {
  people_detected: number
  active_cameras: number
  total_cameras: number
  ppe_violations: number
  intrusions: number
  suspicious_events: number
  open_alerts: number
  critical_alerts: number
  events_today: number
}

export interface TimeBucket {
  bucket: string
  total: number
  by_type: Record<string, number>
  by_severity: Record<string, number>
}

export interface CameraActivity {
  camera_id: string
  camera_name: string
  event_count: number
  people_detected: number
  status: string
}

export interface Statistics {
  kpis: Kpis
  by_type: Record<string, number>
  by_severity: Record<string, number>
  by_status: Record<string, number>
  timeline: TimeBucket[]
  camera_activity: CameraActivity[]
  range_start: string
  range_end: string
  granularity: string
}

export interface Health {
  status: 'ok' | 'degraded'
  app: string
  version: string
  environment: string
  uptime_seconds: number
  database: boolean
  inference_backend: string
  model_loaded: boolean
  cameras_online: number
  cameras_total: number
  timestamp: string
}

export interface BackendInfo {
  name: string
  available: boolean
  active: boolean
  reason: string
  detail: Record<string, unknown>
}

export interface Diagnostics {
  app: string
  version: string
  environment: string
  uptime_seconds: number
  platform: Record<string, unknown>
  python_version: string
  database: Record<string, unknown>
  inference: Record<string, unknown>
  backends: BackendInfo[]
  mojo: {
    mode: string
    available: boolean
    active: boolean
    library: string | null
    abi_version: number | null
    reason: string
    kernels: Record<
      string,
      {
        active: boolean
        implementation: string
        mojo_ms?: number
        numpy_ms?: number
        speedup?: number
        correct?: number
      }
    >
  }
  ppe: Record<string, unknown>
  pipeline: Record<string, unknown>
  config: Record<string, unknown>
  timestamp: string
}

export type JobStatus =
  | 'PENDING'
  | 'QUEUED'
  | 'RUNNING'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'

export interface VideoJob {
  job_id: string
  filename: string
  profile: string
  status: JobStatus
  progress: number
  message: string
  total_frames: number
  processed_frames: number
  duration_seconds: number
  fps: number
  resolution: string
  events_created: number
  people_detected: number
  annotated_path: string | null
  output: Record<string, unknown> | null
  zone_camera_id: string | null
  summary: Record<string, unknown>
  created_at: string
  started_at: string | null
  finished_at: string | null
}

/** One incident placed on the source timeline — the seek targets. */
export interface TimelineEntry {
  event_id: string
  event_type: EventType | string
  label: string
  severity: Severity
  person_id: number | null
  confidence: number
  video_timestamp: number | null
  snapshot_available: boolean
}

/**
 * The finished analysis. Carries API URLs, never filesystem paths — the
 * player is handed a route it can stream from, not a location on the host.
 */
export interface VideoJobResult {
  job_id: string
  filename: string
  status: JobStatus
  progress: number
  message: string

  annotated_ready: boolean
  video_url: string | null
  download_url: string | null
  original_url: string | null

  duration_seconds: number
  fps: number
  frames: number
  resolution: string
  size_bytes: number
  codec: string
  has_audio: boolean
  browser_compatible: boolean
  warnings: string[]

  summary: Record<string, unknown>
  events_created: number
  people_detected: number
  timeline: TimelineEntry[]
}

export interface DetectionProfile {
  name: string
  analysers: string[]
  description: string
}

export interface EvidenceManifest {
  event_id: string
  event_type: string
  timestamp: string
  snapshot: { available: boolean; url: string | null; size_bytes: number }
  clip: {
    available: boolean
    pending: boolean
    url: string | null
    size_bytes: number
  }
  export_url: string
}

/* ── WebSocket payloads ─────────────────────────────────────────────────── */

export interface DetectionBox {
  track_id: number | null
  label: string
  confidence: number
  /** Normalised [x1, y1, x2, y2] */
  bbox: [number, number, number, number]
  helmet: boolean | null
  vest: boolean | null
  ppe_method: string | null
  violation: boolean
  speed: number
  state: string
  zones: string[]
}

export interface DetectionFrame {
  camera_id: string
  frame_index: number
  timestamp: string
  people_count: number
  boxes: DetectionBox[]
  fps: number
  inference_ms: number
  processing_ms: number
  backend: string
}

export interface CameraStatusMessage {
  camera_id: string
  name: string
  status: string
  last_error: string | null
  fps: number
  people_count: number
}

export interface JobProgressMessage {
  job_id: string
  status: JobStatus
  progress?: number
  processed_frames?: number
  total_frames?: number
  events_created?: number
  people_detected?: number
  ppe_violations?: number
  processing_fps?: number
  eta_seconds?: number | null
  annotated_ready?: boolean
  summary?: Record<string, unknown>
  message?: string
}

export type WsMessage =
  | { type: 'hello'; payload: Record<string, unknown>; ts: string }
  | { type: 'detections'; payload: DetectionFrame; ts: string }
  | { type: 'event'; payload: SecurityEvent; ts: string }
  | { type: 'camera_status'; payload: CameraStatusMessage; ts: string }
  | { type: 'job_progress'; payload: JobProgressMessage; ts: string }
  | { type: 'pong'; payload: Record<string, unknown>; ts: string }
