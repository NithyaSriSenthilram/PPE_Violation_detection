/**
 * One camera in the grid: MJPEG frame, vector overlay, and a HUD.
 *
 * The stream is an <img> pointed at the MJPEG endpoint — the simplest thing
 * that works in every browser with no player dependency and no WebRTC
 * signalling. Detections arrive separately over the WebSocket, which is what
 * lets the overlay stay crisp and toggleable.
 */

import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { statusTone } from '../lib/format'
import type { Camera, DetectionFrame, Zone } from '../lib/types'
import { DetectionOverlay, type OverlayLayers } from './DetectionOverlay'
import { StatusDot } from './Primitives'
import { VideoFrame } from './VideoFrame'

interface Props {
  camera: Camera
  frame?: DetectionFrame
  zones?: Zone[]
  layers: OverlayLayers
  paused: boolean
  onTogglePause?: () => void
  onFullscreen?: () => void
  onSelect?: () => void
  compact?: boolean
}

export function CameraTile({
  camera,
  frame,
  zones = [],
  layers,
  paused,
  onTogglePause,
  onFullscreen,
  onSelect,
  compact = false,
}: Props) {
  const [streamError, setStreamError] = useState(false)
  // Bumped to force the <img> to re-request the stream after an error or a
  // resume; MJPEG connections do not recover on their own.
  const [streamKey, setStreamKey] = useState(0)

  const online = camera.status === 'online'

  useEffect(() => {
    if (!paused && online) {
      setStreamError(false)
      setStreamKey((k) => k + 1)
    }
  }, [paused, online])

  const boxes = frame?.boxes ?? []
  const people = frame?.people_count ?? camera.people_count
  const fps = frame?.fps ?? camera.fps
  const violations = boxes.filter((b) => b.violation).length

  return (
    <article
      className="panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        borderColor: violations > 0 ? 'var(--sev-critical)' : 'var(--line)',
        transition: 'border-color var(--t-base) var(--ease)',
      }}
    >
      {/* ── Tile header ──────────────────────────────────────────────────── */}
      <div
        className="row-between"
        style={{
          padding: '0.4rem 0.6rem',
          borderBottom: '1px solid var(--line-soft)',
          background: 'var(--bg-elev)',
          gap: 'var(--s2)',
        }}
      >
        <div className="row grow" style={{ gap: 'var(--s2)', minWidth: 0 }}>
          <StatusDot status={camera.status} />
          <button
            onClick={onSelect}
            className="truncate"
            style={{
              fontSize: 'var(--fs-sm)',
              fontWeight: 600,
              textAlign: 'left',
              color: 'inherit',
            }}
            title={`${camera.name} — ${camera.location || 'no location set'}`}
          >
            {camera.name}
          </button>
        </div>
        <div className="row mono" style={{ gap: 'var(--s3)', fontSize: 'var(--fs-micro)' }}>
          <span title="People currently tracked">
            <span className="dim-2">PPL </span>
            <span style={{ color: people > 0 ? 'var(--hivis)' : 'var(--text-3)', fontWeight: 700 }}>
              {people}
            </span>
          </span>
          <span title="Pipeline frames per second">
            <span className="dim-2">FPS </span>
            <span style={{ fontWeight: 600 }}>{fps.toFixed(1)}</span>
          </span>
          {violations > 0 && (
            <span
              style={{ color: 'var(--sev-critical)', fontWeight: 700 }}
              title={`${violations} PPE violation(s) in view`}
            >
              ⚠ {violations}
            </span>
          )}
        </div>
      </div>

      {/* ── Video + overlay ──────────────────────────────────────────────── */}
      <div style={{ position: 'relative' }}>
        <VideoFrame
          src={
            online && !paused && !streamError
              ? api.streamUrl(camera.camera_id, compact ? 480 : 960, compact ? 6 : 10)
              : null
          }
          alt={`Live view from ${camera.name}`}
          reloadKey={streamKey}
          onError={() => setStreamError(true)}
          placeholder={
            <OfflinePlate
              camera={camera}
              paused={paused}
              streamError={streamError}
              onRetry={() => {
                setStreamError(false)
                setStreamKey((k) => k + 1)
              }}
            />
          }
        >
          {(size) => (
            <>
              <DetectionOverlay
                boxes={boxes}
                zones={zones}
                layers={layers}
                width={size.width}
                height={size.height}
              />

              {/* Corner HUD on glass, so it stays readable over any footage. */}
              {!compact && (
                <div
                  className="glass mono tile-hud"
                  style={{
                    position: 'absolute',
                    left: 8,
                    bottom: 8,
                    opacity: 0,
                    transition: 'opacity var(--t-fast) var(--ease)',
                    padding: '0.25rem 0.45rem',
                    borderRadius: 'var(--r-xs)',
                    fontSize: 'var(--fs-micro)',
                    letterSpacing: '0.06em',
                    display: 'flex',
                    gap: 'var(--s3)',
                    color: 'var(--text-2)',
                    pointerEvents: 'none',
                  }}
                >
                  <span className="truncate" style={{ maxWidth: '18ch' }}>
                    {camera.location || 'UNASSIGNED'}
                  </span>
                  {frame && (
                    <>
                      <span className="dim-2">|</span>
                      <span title="Inference latency per detected frame">
                        {frame.inference_ms.toFixed(0)}ms
                      </span>
                      <span className="dim-2">|</span>
                      <span title="Active inference backend">{frame.backend}</span>
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </VideoFrame>

        {/* Controls: appear on hover, so they never obscure footage while
            an operator is watching. */}
        <div
          className="tile-controls"
          style={{
            position: 'absolute',
            right: 8,
            top: 8,
            display: 'flex',
            gap: 4,
            opacity: 0,
            transition: 'opacity var(--t-fast) var(--ease)',
          }}
        >
          {onTogglePause && (
            <TileButton
              label={paused ? 'Resume stream' : 'Pause stream'}
              onClick={onTogglePause}
            >
              {paused ? (
                <path d="M6 4l9 6-9 6z" fill="currentColor" stroke="none" />
              ) : (
                <>
                  <rect x="6" y="4.5" width="2.8" height="11" fill="currentColor" stroke="none" />
                  <rect x="11.2" y="4.5" width="2.8" height="11" fill="currentColor" stroke="none" />
                </>
              )}
            </TileButton>
          )}
          {onFullscreen && (
            <TileButton label="Fullscreen" onClick={onFullscreen}>
              <path
                d="M3 7V3h4M17 7V3h-4M17 13v4h-4M3 13v4h4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
              />
            </TileButton>
          )}
        </div>
      </div>
    </article>
  )
}

function TileButton({
  label,
  onClick,
  children,
}: {
  label: string
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      className="glass"
      onClick={onClick}
      aria-label={label}
      title={label}
      style={{
        width: 26,
        height: 26,
        borderRadius: 'var(--r-xs)',
        display: 'grid',
        placeItems: 'center',
        color: 'var(--text)',
      }}
    >
      <svg width="15" height="15" viewBox="0 0 20 20" aria-hidden>
        {children}
      </svg>
    </button>
  )
}

/**
 * Shown instead of video. States an operator must be able to tell apart:
 * paused by them, stream dropped, or the camera reporting a real fault —
 * with the backend's own error text when there is one.
 */
function OfflinePlate({
  camera,
  paused,
  streamError,
  onRetry,
}: {
  camera: Camera
  paused: boolean
  streamError: boolean
  onRetry: () => void
}) {
  const title = paused
    ? 'Paused'
    : streamError
      ? 'Stream interrupted'
      : camera.status === 'error'
        ? 'Camera fault'
        : camera.status === 'connecting'
          ? 'Connecting'
          : camera.status === 'ended'
            ? 'Source ended'
            : 'Offline'

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'grid',
        placeItems: 'center',
        // A faint diagonal hazard hatch: reads as "no signal" without shouting.
        backgroundImage:
          'repeating-linear-gradient(45deg, rgba(255,255,255,0.018) 0 8px, transparent 8px 16px)',
      }}
    >
      <div className="col" style={{ alignItems: 'center', gap: 'var(--s2)', padding: 'var(--s4)' }}>
        <span
          className="mono"
          style={{
            fontSize: 'var(--fs-tiny)',
            letterSpacing: '0.2em',
            textTransform: 'uppercase',
            color: statusTone(paused ? 'idle' : camera.status),
            fontWeight: 700,
          }}
        >
          {title}
        </span>
        {camera.last_error && !paused && (
          <span
            className="dim"
            style={{
              fontSize: 'var(--fs-micro)',
              maxWidth: '30ch',
              textAlign: 'center',
              lineHeight: 1.5,
            }}
          >
            {camera.last_error}
          </span>
        )}
        {streamError && !paused && (
          <button className="btn btn-sm" onClick={onRetry}>
            Reconnect
          </button>
        )}
      </div>
    </div>
  )
}
