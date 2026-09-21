/**
 * Application root.
 *
 * The WebSocket lives here, above the router, for a reason: detection frames
 * must keep arriving as the operator moves between pages, and a socket that
 * reconnected on every navigation would drop frames and hammer the backend.
 * Frames are held in a ref-backed map and mirrored into state on a fixed
 * cadence, so a 12 fps stream from several cameras does not trigger a React
 * render per frame.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { useWebSocket } from './hooks/useWebSocket'
import { usePolling } from './hooks/usePolling'
import { api } from './lib/api'
import type {
  DetectionFrame,
  JobProgressMessage,
  SecurityEvent,
} from './lib/types'
import { Alerts } from './pages/Alerts'
import { Analytics } from './pages/Analytics'
import { Cameras } from './pages/Cameras'
import { Dashboard } from './pages/Dashboard'
import { Diagnostics } from './pages/Diagnostics'
import { LiveCameras } from './pages/LiveCameras'
import { VideoAnalysis } from './pages/VideoAnalysis'

/** Overlay refresh rate. Faster than this is imperceptible; slower feels laggy. */
const RENDER_HZ = 8

export function App() {
  // Latest frame per camera. Written on every message, read on a timer.
  const framesRef = useRef(new Map<string, DetectionFrame>())
  const [frames, setFrames] = useState<Map<string, DetectionFrame>>(new Map())
  const [jobProgress, setJobProgress] = useState<Map<string, JobProgressMessage>>(
    new Map(),
  )
  const [toast, setToast] = useState<SecurityEvent | null>(null)

  const onDetections = useCallback((frame: DetectionFrame) => {
    framesRef.current.set(frame.camera_id, frame)
  }, [])

  const onJobProgress = useCallback((progress: JobProgressMessage) => {
    setJobProgress((current) => {
      const next = new Map(current)
      next.set(progress.job_id, progress)
      return next
    })
  }, [])

  const onEvent = useCallback((event: SecurityEvent) => {
    // Only surface genuinely new, unhandled incidents — a status change
    // echoed back should not pop a toast at the operator who made it.
    if (event.status === 'OPEN') setToast(event)
  }, [])

  const onCameraStatus = useCallback(() => {
    // Camera cards poll their own status; nothing extra to do here.
  }, [])

  const { state } = useWebSocket({
    onDetections,
    onEvent,
    onCameraStatus,
    onJobProgress,
  })

  // Mirror the frame map into state at a fixed rate.
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (framesRef.current.size) setFrames(new Map(framesRef.current))
    }, 1000 / RENDER_HZ)
    return () => window.clearInterval(timer)
  }, [])

  const { data: alerts } = usePolling(() => api.alerts({ limit: 1 }), 20_000)

  return (
    <AppShell connection={state} openAlertCount={alerts?.total ?? 0}>
      <Routes>
        <Route path="/" element={<Dashboard frames={frames} />} />
        <Route path="/live" element={<LiveCameras frames={frames} />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/analysis" element={<VideoAnalysis jobProgress={jobProgress} />} />
        <Route path="/cameras" element={<Cameras />} />
        <Route path="/diagnostics" element={<Diagnostics />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>

      {toast && <IncidentToast event={toast} onDismiss={() => setToast(null)} />}
    </AppShell>
  )
}

/**
 * A newly raised incident, announced without stealing focus. It is a live
 * region so a screen reader hears it, and it dismisses itself — an operator
 * watching cameras should not have to clear notifications.
 */
function IncidentToast({
  event,
  onDismiss,
}: {
  event: SecurityEvent
  onDismiss: () => void
}) {
  useEffect(() => {
    const timer = window.setTimeout(onDismiss, 7000)
    return () => window.clearTimeout(timer)
  }, [event.event_id, onDismiss])

  const tone =
    event.severity === 'CRITICAL'
      ? 'var(--sev-critical)'
      : event.severity === 'HIGH'
        ? 'var(--sev-high)'
        : event.severity === 'MEDIUM'
          ? 'var(--sev-medium)'
          : 'var(--sev-low)'

  return (
    <div
      role="status"
      aria-live="polite"
      className="glass fade-in"
      style={{
        position: 'fixed',
        right: 'var(--s5)',
        bottom: 'var(--s5)',
        maxWidth: 340,
        padding: 'var(--s3) var(--s4)',
        borderRadius: 'var(--r-sm)',
        borderLeft: `3px solid ${tone}`,
        boxShadow: 'var(--shadow-lg)',
        zIndex: 90,
      }}
    >
      <div className="row-between" style={{ gap: 'var(--s3)' }}>
        <div className="col" style={{ gap: 2, minWidth: 0 }}>
          <div className="row" style={{ gap: 'var(--s2)' }}>
            <span
              className="mono"
              style={{
                fontSize: 'var(--fs-micro)',
                fontWeight: 700,
                letterSpacing: '0.1em',
                color: tone,
              }}
            >
              {event.severity}
            </span>
            <span style={{ fontSize: 'var(--fs-tiny)', fontWeight: 600 }}>
              {event.label}
            </span>
          </div>
          <span className="dim" style={{ fontSize: 'var(--fs-tiny)', lineHeight: 1.45 }}>
            {event.description}
          </span>
          {event.camera_name && (
            <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
              {event.camera_name.toUpperCase()}
            </span>
          )}
        </div>
        <button
          className="btn btn-ghost btn-sm"
          onClick={onDismiss}
          aria-label="Dismiss notification"
        >
          <svg width="12" height="12" viewBox="0 0 20 20" aria-hidden>
            <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </div>
  )
}
