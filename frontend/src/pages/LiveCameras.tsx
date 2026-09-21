/**
 * The camera wall.
 *
 * Layer toggles, grid density and fullscreen all live here because they are
 * the controls an operator reaches for constantly. Detections are overlaid in
 * the browser, so toggling a layer is instant and costs the server nothing.
 */

import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import type { Camera, DetectionFrame, Zone } from '../lib/types'
import { CameraTile } from '../components/CameraTile'
import {
  DEFAULT_LAYERS,
  DetectionOverlay,
  type OverlayLayers,
} from '../components/DetectionOverlay'
import { VideoFrame } from '../components/VideoFrame'
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  Readout,
  Segmented,
} from '../components/Primitives'

type Density = '1' | '2' | '3'

const LAYER_LABELS: { key: keyof OverlayLayers; label: string }[] = [
  { key: 'boxes', label: 'Boxes' },
  { key: 'ids', label: 'IDs' },
  { key: 'ppe', label: 'PPE' },
  { key: 'zones', label: 'Zones' },
]

export function LiveCameras({ frames }: { frames: Map<string, DetectionFrame> }) {
  const [density, setDensity] = useState<Density>('2')
  const [layers, setLayers] = useState<OverlayLayers>(DEFAULT_LAYERS)
  const [paused, setPaused] = useState<Set<string>>(new Set())
  const [fullscreen, setFullscreen] = useState<string | null>(null)
  const [zonesByCamera, setZonesByCamera] = useState<Record<string, Zone[]>>({})

  const { data: cameras, error, refresh } = usePolling(() => api.cameras(), 6_000)
  const enabled = useMemo(() => (cameras ?? []).filter((c) => c.enabled), [cameras])

  // Zones change rarely; fetch once per camera and refresh on demand only.
  useEffect(() => {
    let alive = true
    Promise.all(
      enabled.map(async (camera) => {
        if (camera.zone_count === 0) return [camera.camera_id, [] as Zone[]] as const
        try {
          return [camera.camera_id, await api.zones(camera.camera_id)] as const
        } catch {
          return [camera.camera_id, [] as Zone[]] as const
        }
      }),
    ).then((pairs) => {
      if (alive) setZonesByCamera(Object.fromEntries(pairs))
    })
    return () => {
      alive = false
    }
  }, [enabled.map((c) => `${c.camera_id}:${c.zone_count}`).join(',')])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFullscreen(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function togglePause(id: string) {
    setPaused((current) => {
      const next = new Set(current)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const fullscreenCamera = enabled.find((c) => c.camera_id === fullscreen)
  const totals = useMemo(() => {
    let people = 0
    let violations = 0
    for (const camera of enabled) {
      const frame = frames.get(camera.camera_id)
      people += frame?.people_count ?? 0
      violations += frame?.boxes.filter((b) => b.violation).length ?? 0
    }
    return { people, violations }
  }, [enabled, frames])

  return (
    <>
      <PageHeader
        eyebrow="Monitoring"
        title="Camera Wall"
        subtitle="Detections are drawn in the browser from the live feed — toggling a layer never interrupts the stream."
      />

      {/* ── Control bar ──────────────────────────────────────────────────── */}
      <div
        className="panel row-between"
        style={{ padding: 'var(--s3)', marginBottom: 'var(--s4)', flexWrap: 'wrap', gap: 'var(--s3)' }}
      >
        <div className="row" style={{ gap: 'var(--s5)' }}>
          <Readout
            label="Streaming"
            value={`${enabled.filter((c) => c.status === 'online').length}/${enabled.length}`}
            size="sm"
          />
          <Readout
            label="People in view"
            value={String(totals.people)}
            size="sm"
            tone={totals.people > 0 ? 'var(--hivis)' : undefined}
          />
          <Readout
            label="PPE violations"
            value={String(totals.violations)}
            size="sm"
            tone={totals.violations > 0 ? 'var(--sev-critical)' : undefined}
          />
        </div>

        <div className="row" style={{ gap: 'var(--s4)', flexWrap: 'wrap' }}>
          <div className="row" style={{ gap: 'var(--s2)' }}>
            <span className="eyebrow">Overlay</span>
            <div className="row" style={{ gap: 4 }}>
              {LAYER_LABELS.map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setLayers((l) => ({ ...l, [key]: !l[key] }))}
                  aria-pressed={layers[key]}
                  className="chip"
                  style={{
                    cursor: 'pointer',
                    color: layers[key] ? '#10140a' : 'var(--text-3)',
                    background: layers[key] ? 'var(--hivis)' : 'var(--surface-2)',
                    borderColor: layers[key] ? 'var(--hivis)' : 'var(--line-strong)',
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="row" style={{ gap: 'var(--s2)' }}>
            <span className="eyebrow">Grid</span>
            <Segmented
              ariaLabel="Grid density"
              value={density}
              onChange={setDensity}
              options={[
                { value: '1', label: '1×' },
                { value: '2', label: '2×' },
                { value: '3', label: '3×' },
              ]}
            />
          </div>
        </div>
      </div>

      {error && (
        <div style={{ marginBottom: 'var(--s3)' }}>
          <ErrorBanner message={error} onRetry={refresh} />
        </div>
      )}

      {enabled.length === 0 ? (
        <div className="panel">
          <EmptyState
            title="No cameras enabled"
            hint="Add a camera, or enable one that is currently switched off."
            action={
              <Link className="btn btn-primary" to="/cameras">
                Manage cameras
              </Link>
            }
          />
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            // Never leave an empty column: a wall with one camera should show
            // one large tile, not a quarter-size tile beside dead space.
            gridTemplateColumns: `repeat(${Math.min(Number(density), enabled.length)}, minmax(0, 1fr))`,
            gap: 'var(--s3)',
          }}
        >
          {enabled.map((camera) => (
            <CameraTile
              key={camera.camera_id}
              camera={camera}
              frame={frames.get(camera.camera_id)}
              zones={zonesByCamera[camera.camera_id] ?? []}
              layers={layers}
              paused={paused.has(camera.camera_id)}
              onTogglePause={() => togglePause(camera.camera_id)}
              onFullscreen={() => setFullscreen(camera.camera_id)}
              compact={density === '3'}
            />
          ))}
        </div>
      )}

      {fullscreenCamera && (
        <FullscreenView
          camera={fullscreenCamera}
          frame={frames.get(fullscreenCamera.camera_id)}
          zones={zonesByCamera[fullscreenCamera.camera_id] ?? []}
          layers={layers}
          onClose={() => setFullscreen(null)}
        />
      )}
    </>
  )
}

/** Fullscreen: one camera, maximum resolution, HUD kept out of the frame. */
function FullscreenView({
  camera,
  frame,
  zones,
  layers,
  onClose,
}: {
  camera: Camera
  frame?: DetectionFrame
  zones: Zone[]
  layers: OverlayLayers
  onClose: () => void
}) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Fullscreen view: ${camera.name}`}
      style={{
        position: 'fixed',
        inset: 0,
        background: '#04050a',
        zIndex: 120,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <header
        className="row-between"
        style={{
          padding: 'var(--s3) var(--s5)',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-elev)',
        }}
      >
        <div className="row" style={{ gap: 'var(--s4)' }}>
          <div className="col" style={{ gap: 0 }}>
            <span style={{ fontSize: 'var(--fs-md)', fontWeight: 700 }}>
              {camera.name}
            </span>
            <span className="eyebrow">{camera.location || 'Unassigned'}</span>
          </div>
          {frame && (
            <div className="row" style={{ gap: 'var(--s5)' }}>
              <Readout label="People" value={String(frame.people_count)} size="sm" />
              <Readout label="FPS" value={frame.fps.toFixed(1)} size="sm" />
              <Readout label="Inference" value={`${frame.inference_ms.toFixed(0)}ms`} size="sm" />
              <Readout label="Backend" value={frame.backend} size="sm" />
            </div>
          )}
        </div>
        <button className="btn" onClick={onClose}>
          Exit fullscreen <span className="chip">ESC</span>
        </button>
      </header>

      <VideoFrame
        src={api.streamUrl(camera.camera_id, 1280, 12)}
        alt={`Live view from ${camera.name}`}
        mode="fit"
      >
        {(size) => (
          <DetectionOverlay
            boxes={frame?.boxes ?? []}
            zones={zones}
            layers={layers}
            width={size.width}
            height={size.height}
          />
        )}
      </VideoFrame>
    </div>
  )
}
