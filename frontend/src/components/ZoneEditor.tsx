/**
 * Draw restricted zones directly on the camera's own frame.
 *
 * Clicking places vertices on a live still from the camera, which is the only
 * way an operator can place a boundary accurately — a zone drawn on an
 * abstract rectangle never lines up with the doorway they meant.
 *
 * Points are stored normalised (0..1), so a zone survives a resolution change
 * and renders identically on the wall display and on a phone.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../lib/api'
import type { Camera, Zone, ZoneType } from '../lib/types'
import { ErrorBanner } from './Primitives'

type Point = [number, number]

const ZONE_TYPES: { value: ZoneType; label: string; hint: string }[] = [
  { value: 'restricted', label: 'Restricted', hint: 'Entry raises an intrusion alert' },
  { value: 'monitored', label: 'Monitored', hint: 'Tracks dwell time, no entry alert' },
  { value: 'loitering', label: 'Loitering', hint: 'Dwell time only' },
  { value: 'crowd', label: 'Crowd', hint: 'Counts people for density alerts' },
]

const PALETTE = ['#f43f5e', '#ff8a3d', '#ffc53d', '#5b9cf5', '#34d97b', '#b48cf5']

export function ZoneEditor({
  camera,
  onClose,
}: {
  camera: Camera
  onClose: () => void
}) {
  const [zones, setZones] = useState<Zone[]>([])
  const [draft, setDraft] = useState<Point[]>([])
  const [name, setName] = useState('')
  const [type, setType] = useState<ZoneType>('restricted')
  const [colour, setColour] = useState(PALETTE[0])
  const [loiterThreshold, setLoiterThreshold] = useState('')
  const [crowdThreshold, setCrowdThreshold] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [frameSize, setFrameSize] = useState({ width: 960, height: 540 })
  const frameRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    try {
      setZones(await api.zones(camera.camera_id))
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not load zones')
    }
  }, [camera.camera_id])

  useEffect(() => {
    void load()
  }, [load])

  // Pixel-space geometry keeps the vertex handles circular and the polygon
  // edges an even weight whatever the frame's aspect ratio.
  useEffect(() => {
    const element = frameRef.current
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect
      if (rect?.width) setFrameSize({ width: rect.width, height: rect.height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        draft.length ? setDraft([]) : onClose()
      }
      // Undo the last vertex — essential when placing a boundary by hand.
      if ((e.key === 'z' && (e.metaKey || e.ctrlKey)) || e.key === 'Backspace') {
        e.preventDefault()
        setDraft((d) => d.slice(0, -1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [draft.length, onClose])

  function addPoint(event: React.MouseEvent<HTMLDivElement>) {
    const box = frameRef.current?.getBoundingClientRect()
    if (!box) return
    const x = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width))
    const y = Math.min(1, Math.max(0, (event.clientY - box.top) / box.height))
    setDraft((d) => (d.length >= 64 ? d : [...d, [Number(x.toFixed(4)), Number(y.toFixed(4))]]))
  }

  async function save() {
    if (draft.length < 3) {
      setError('A zone needs at least three points.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      await api.createZone({
        camera_id: camera.camera_id,
        name: name.trim() || `Zone ${zones.length + 1}`,
        zone_type: type,
        polygon: draft,
        colour,
        loitering_threshold: loiterThreshold ? Number(loiterThreshold) : null,
        crowd_threshold: crowdThreshold ? Number(crowdThreshold) : null,
      })
      setDraft([])
      setName('')
      setLoiterThreshold('')
      setCrowdThreshold('')
      setColour(PALETTE[(zones.length + 1) % PALETTE.length])
      await load()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not save the zone')
    } finally {
      setBusy(false)
    }
  }

  async function remove(zoneId: string) {
    setBusy(true)
    try {
      await api.deleteZone(zoneId)
      await load()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not delete the zone')
    } finally {
      setBusy(false)
    }
  }

  async function toggle(zone: Zone) {
    setBusy(true)
    try {
      await api.updateZone(zone.zone_id, { enabled: !zone.enabled })
      await load()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not update the zone')
    } finally {
      setBusy(false)
    }
  }

  const draftPath = useMemo(
    () =>
      draft
        .map(([x, y]) => `${x * frameSize.width},${y * frameSize.height}`)
        .join(' '),
    [draft, frameSize],
  )

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Zones for ${camera.name}`}
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(4,5,7,0.76)',
        backdropFilter: 'blur(3px)',
        display: 'grid',
        placeItems: 'center',
        padding: 'var(--s5)',
        zIndex: 110,
      }}
    >
      <div
        className="panel fade-in"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(1080px, 100%)',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: 'var(--shadow-lg)',
        }}
      >
        <header className="panel-head">
          <div className="col" style={{ gap: 1 }}>
            <span className="eyebrow">Zone editor</span>
            <h2 style={{ fontSize: 'var(--fs-md)' }}>{camera.name}</h2>
          </div>
          <button className="btn btn-ghost" onClick={onClose} aria-label="Close zone editor">
            <svg width="16" height="16" viewBox="0 0 20 20" aria-hidden>
              <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            </svg>
          </button>
        </header>

        <div
          className="scroll-y zone-grid"
          style={{
            padding: 'var(--s4)',
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1.5fr) minmax(0, 1fr)',
            gap: 'var(--s4)',
            alignItems: 'start',
          }}
        >
          {/* ── Canvas ───────────────────────────────────────────────────── */}
          <div className="col" style={{ gap: 'var(--s2)' }}>
            <div
              ref={frameRef}
              onClick={addPoint}
              style={{
                position: 'relative',
                aspectRatio: '16 / 9',
                background: '#050609',
                border: '1px solid var(--line-strong)',
                borderRadius: 'var(--r-sm)',
                overflow: 'hidden',
                cursor: 'crosshair',
              }}
            >
              {camera.status === 'online' ? (
                <img
                  src={api.streamUrl(camera.camera_id, 960, 4)}
                  alt={`Frame from ${camera.name} for placing zones`}
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                  draggable={false}
                />
              ) : (
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    display: 'grid',
                    placeItems: 'center',
                    color: 'var(--text-4)',
                    fontSize: 'var(--fs-sm)',
                    textAlign: 'center',
                    padding: 'var(--s4)',
                  }}
                >
                  Camera is offline — start it to place zones against a real frame.
                </div>
              )}

              <svg
                viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                }}
              >
                {/* Saved zones */}
                {zones
                  .filter((z) => z.polygon.length >= 3)
                  .map((zone) => (
                    <polygon
                      key={zone.zone_id}
                      points={zone.polygon
                        .map(
                          ([x, y]) =>
                            `${x * frameSize.width},${y * frameSize.height}`,
                        )
                        .join(' ')}
                      fill={zone.colour}
                      fillOpacity={zone.enabled ? 0.14 : 0.05}
                      stroke={zone.colour}
                      strokeOpacity={zone.enabled ? 1 : 0.4}
                      strokeWidth={1.5}
                      strokeDasharray={zone.enabled ? undefined : '5 4'}
                    />
                  ))}

                {/* Draft */}
                {draft.length >= 2 && (
                  <polygon
                    points={draftPath}
                    fill={colour}
                    fillOpacity={0.18}
                    stroke={colour}
                    strokeWidth={1.8}
                    strokeDasharray="6 4"
                  />
                )}
                {draft.map(([x, y], index) => (
                  <g key={index}>
                    <circle
                      cx={x * frameSize.width}
                      cy={y * frameSize.height}
                      r={5}
                      fill={colour}
                      stroke="#08090c"
                      strokeWidth={1.5}
                    />
                    {/* Vertex order matters when reading a drawn polygon back. */}
                    <text
                      x={x * frameSize.width + 8}
                      y={y * frameSize.height - 6}
                      fontFamily="var(--font-mono)"
                      fontSize={10}
                      fontWeight={700}
                      fill={colour}
                    >
                      {index + 1}
                    </text>
                  </g>
                ))}
              </svg>
            </div>

            <div className="row-between" style={{ gap: 'var(--s3)', flexWrap: 'wrap' }}>
              <span className="dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                Click to place points · {draft.length} placed
                {draft.length > 0 && ' · Backspace removes the last'}
              </span>
              {draft.length > 0 && (
                <button className="btn btn-sm" onClick={() => setDraft([])}>
                  Clear points
                </button>
              )}
            </div>
          </div>

          {/* ── Form + list ──────────────────────────────────────────────── */}
          <div className="col" style={{ gap: 'var(--s4)' }}>
            {error && <ErrorBanner message={error} />}

            <div className="col" style={{ gap: 'var(--s3)' }}>
              <span className="eyebrow">New zone</span>

              <div className="field">
                <label className="eyebrow" htmlFor="z-name">Name</label>
                <input
                  id="z-name"
                  className="input"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={`Zone ${zones.length + 1}`}
                />
              </div>

              <div className="field">
                <label className="eyebrow" htmlFor="z-type">Type</label>
                <select
                  id="z-type"
                  className="select"
                  value={type}
                  onChange={(e) => setType(e.target.value as ZoneType)}
                >
                  {ZONE_TYPES.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <span className="dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                  {ZONE_TYPES.find((o) => o.value === type)?.hint}
                </span>
              </div>

              {(type === 'loitering' || type === 'monitored' || type === 'restricted') && (
                <div className="field">
                  <label className="eyebrow" htmlFor="z-loiter">
                    Loitering threshold (seconds)
                  </label>
                  <input
                    id="z-loiter"
                    className="input"
                    type="number"
                    min={1}
                    value={loiterThreshold}
                    onChange={(e) => setLoiterThreshold(e.target.value)}
                    placeholder="Inherit global setting"
                  />
                </div>
              )}

              {type === 'crowd' && (
                <div className="field">
                  <label className="eyebrow" htmlFor="z-crowd">
                    Crowd threshold (people)
                  </label>
                  <input
                    id="z-crowd"
                    className="input"
                    type="number"
                    min={1}
                    value={crowdThreshold}
                    onChange={(e) => setCrowdThreshold(e.target.value)}
                    placeholder="Inherit global setting"
                  />
                </div>
              )}

              <div className="field">
                <span className="eyebrow">Colour</span>
                <div className="row" style={{ gap: 6 }}>
                  {PALETTE.map((option) => (
                    <button
                      key={option}
                      onClick={() => setColour(option)}
                      aria-label={`Use colour ${option}`}
                      aria-pressed={colour === option}
                      style={{
                        width: 22,
                        height: 22,
                        borderRadius: 'var(--r-xs)',
                        background: option,
                        border:
                          colour === option
                            ? '2px solid var(--text)'
                            : '1px solid var(--line-strong)',
                      }}
                    />
                  ))}
                </div>
              </div>

              <button
                className="btn btn-primary"
                onClick={save}
                disabled={busy || draft.length < 3}
              >
                {draft.length < 3
                  ? `Place ${3 - draft.length} more point${3 - draft.length === 1 ? '' : 's'}`
                  : 'Save zone'}
              </button>
            </div>

            <div className="col" style={{ gap: 'var(--s2)' }}>
              <span className="eyebrow">Existing zones ({zones.length})</span>
              {zones.length === 0 ? (
                <span className="dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                  None yet. Zone-based rules stay dormant until one is drawn.
                </span>
              ) : (
                zones.map((zone) => (
                  <div
                    key={zone.zone_id}
                    className="row-between"
                    style={{
                      padding: 'var(--s2) var(--s3)',
                      border: '1px solid var(--line-soft)',
                      borderRadius: 'var(--r-sm)',
                      background: 'var(--surface-2)',
                      gap: 'var(--s2)',
                      opacity: zone.enabled ? 1 : 0.55,
                    }}
                  >
                    <div className="row grow" style={{ gap: 'var(--s2)', minWidth: 0 }}>
                      <span
                        aria-hidden
                        style={{
                          width: 10,
                          height: 10,
                          borderRadius: 2,
                          background: zone.colour,
                          flexShrink: 0,
                        }}
                      />
                      <div className="col" style={{ gap: 0, minWidth: 0 }}>
                        <span className="truncate" style={{ fontSize: 'var(--fs-tiny)', fontWeight: 600 }}>
                          {zone.name}
                        </span>
                        <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
                          {zone.zone_type.toUpperCase()} · {zone.polygon.length} PTS
                          {zone.loitering_threshold ? ` · ${zone.loitering_threshold}s` : ''}
                          {zone.crowd_threshold ? ` · ${zone.crowd_threshold}p` : ''}
                        </span>
                      </div>
                    </div>
                    <div className="row" style={{ gap: 4 }}>
                      <button
                        className="btn btn-sm btn-ghost"
                        onClick={() => toggle(zone)}
                        disabled={busy}
                        title={zone.enabled ? 'Disable this zone' : 'Enable this zone'}
                      >
                        {zone.enabled ? 'On' : 'Off'}
                      </button>
                      <button
                        className="btn btn-sm btn-ghost btn-danger"
                        onClick={() => remove(zone.zone_id)}
                        disabled={busy}
                        aria-label={`Delete zone ${zone.name}`}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
