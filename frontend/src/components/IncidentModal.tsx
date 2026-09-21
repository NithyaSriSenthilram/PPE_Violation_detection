/**
 * Incident detail: everything recorded about one event, plus the actions an
 * operator takes on it.
 *
 * Evidence is presented honestly. A clip whose file has not finished writing
 * shows as *being written* rather than as missing, because the two mean
 * different things to someone building a case. Advisory event types carry the
 * AI-inference notice in full, not as a footnote.
 */

import { useEffect, useState } from 'react'
import { api, ApiError } from '../lib/api'
import { fullStamp, percent, personLabel, videoTime } from '../lib/format'
import type { EventStatus, EvidenceManifest, SecurityEvent } from '../lib/types'
import { AdvisoryNotice, ErrorBanner, Readout, SeverityBadge } from './Primitives'
import { PPEMethodBadge, PPEMethodLine, presentationFor } from './PPEMethod'

const STATUS_ACTIONS: { status: EventStatus; label: string; primary?: boolean }[] = [
  { status: 'ACKNOWLEDGED', label: 'Acknowledge', primary: true },
  { status: 'RESOLVED', label: 'Resolve' },
  { status: 'DISMISSED', label: 'Dismiss' },
]

/** Metadata keys rendered as the readout grid; everything else goes to the raw view. */
const HIGHLIGHT_KEYS = [
  'ppe_method',
  'summary',
  'dwell_seconds',
  'threshold_seconds',
  'speed_body_heights_per_second',
  'people_count',
  'baseline',
  'aspect_ratio',
  'drop_fraction',
  'horizontal_seconds',
  'frames_inside',
  'trigger',
  'duplicates_suppressed',
]

const KEY_LABELS: Record<string, string> = {
  ppe_method: 'Detection method',
  summary: 'PPE state',
  dwell_seconds: 'Dwell',
  threshold_seconds: 'Threshold',
  speed_body_heights_per_second: 'Speed',
  people_count: 'People',
  baseline: 'Baseline',
  aspect_ratio: 'Aspect ratio',
  drop_fraction: 'Vertical drop',
  horizontal_seconds: 'Horizontal for',
  frames_inside: 'Frames inside',
  trigger: 'Trigger',
  duplicates_suppressed: 'Duplicates suppressed',
}

export function IncidentModal({
  event,
  onClose,
  onUpdated,
}: {
  event: SecurityEvent
  onClose: () => void
  onUpdated?: (event: SecurityEvent) => void
}) {
  const [current, setCurrent] = useState(event)
  const [manifest, setManifest] = useState<EvidenceManifest | null>(null)
  const [notes, setNotes] = useState(event.notes ?? '')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showRaw, setShowRaw] = useState(false)

  useEffect(() => {
    setCurrent(event)
    setNotes(event.notes ?? '')
  }, [event])

  useEffect(() => {
    let alive = true
    api
      .evidence(event.event_id)
      .then((m) => alive && setManifest(m))
      .catch(() => alive && setManifest(null))
    return () => {
      alive = false
    }
  }, [event.event_id])

  // Escape closes — a modal that traps the operator is a liability.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function setStatus(status: EventStatus) {
    setBusy(true)
    setError(null)
    try {
      const updated = await api.setEventStatus(
        current.event_id,
        status,
        notes || undefined,
      )
      setCurrent(updated)
      onUpdated?.(updated)
    } catch (cause) {
      setError(
        cause instanceof ApiError ? cause.message : 'Could not update this incident',
      )
    } finally {
      setBusy(false)
    }
  }

  const metadata = current.detection_metadata ?? {}
  const highlights = HIGHLIGHT_KEYS.filter(
    (key) => metadata[key] !== undefined && metadata[key] !== null,
  )
  // PPE provenance is a first-class part of the incident, not a footnote:
  // an operator acting on "no helmet" needs to know whether a trained model
  // saw a bare head or a colour test failed to find helmet-coloured pixels.
  const ppeMethod = metadata.ppe_method as string | undefined
  const missing = Array.isArray(metadata.missing) ? (metadata.missing as string[]) : []
  const ppeFinding = missing.length
    ? missing.map((item) => `${item[0].toUpperCase()}${item.slice(1)} Missing`).join(' + ')
    : undefined

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Incident: ${current.label}`}
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(4, 5, 7, 0.74)',
        backdropFilter: 'blur(3px)',
        display: 'grid',
        placeItems: 'center',
        padding: 'var(--s5)',
        zIndex: 100,
      }}
    >
      <div
        className="panel fade-in"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(980px, 100%)',
          maxHeight: '90vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: 'var(--shadow-lg)',
        }}
      >
        {/* ── Header ───────────────────────────────────────────────────────── */}
        <header
          className="row-between"
          style={{
            padding: 'var(--s4)',
            borderBottom: '1px solid var(--line)',
            gap: 'var(--s3)',
          }}
        >
          <div className="col" style={{ gap: 5, minWidth: 0 }}>
            <div className="row" style={{ gap: 'var(--s2)', flexWrap: 'wrap' }}>
              <SeverityBadge severity={current.severity} />
              <span className="chip">{current.status}</span>
              {current.advisory && (
                <span
                  className="chip"
                  style={{
                    color: 'var(--advisory)',
                    borderColor: 'var(--advisory)',
                    background: 'rgba(180,140,245,0.12)',
                  }}
                >
                  Advisory
                </span>
              )}
              {ppeMethod && <PPEMethodBadge method={ppeMethod} />}
            </div>
            <h2 style={{ fontSize: 'var(--fs-lg)' }}>{current.label}</h2>
            <p className="dim" style={{ fontSize: 'var(--fs-sm)' }}>
              {current.description}
            </p>
          </div>
          <button className="btn btn-ghost" onClick={onClose} aria-label="Close incident">
            <svg width="16" height="16" viewBox="0 0 20 20" aria-hidden>
              <path
                d="M5 5l10 10M15 5L5 15"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </header>

        {/* ── Body ─────────────────────────────────────────────────────────── */}
        <div className="scroll-y" style={{ padding: 'var(--s4)', display: 'grid', gap: 'var(--s4)' }}>
          {error && <ErrorBanner message={error} />}
          {current.advisory && <AdvisoryNotice />}
          {ppeMethod && (
            <PPEMethodLine
              method={ppeMethod}
              confidence={current.confidence}
              finding={ppeFinding}
            />
          )}

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 1.15fr) minmax(0, 1fr)',
              gap: 'var(--s4)',
            }}
            className="incident-grid"
          >
            {/* Evidence */}
            <section className="col" style={{ gap: 'var(--s3)' }}>
              <span className="eyebrow">Evidence</span>
              {manifest?.snapshot.available ? (
                <a
                  href={api.snapshotUrl(current.event_id)}
                  target="_blank"
                  rel="noreferrer"
                  title="Open the full-size snapshot"
                >
                  <img
                    src={api.snapshotUrl(current.event_id)}
                    alt={`Snapshot of ${current.label}`}
                    style={{
                      width: '100%',
                      borderRadius: 'var(--r-sm)',
                      border: '1px solid var(--line)',
                      display: 'block',
                      background: '#050609',
                    }}
                  />
                </a>
              ) : (
                <div
                  style={{
                    aspectRatio: '16 / 9',
                    display: 'grid',
                    placeItems: 'center',
                    border: '1px dashed var(--line)',
                    borderRadius: 'var(--r-sm)',
                    color: 'var(--text-4)',
                    fontSize: 'var(--fs-sm)',
                  }}
                >
                  No snapshot captured
                </div>
              )}

              {manifest?.clip.available ? (
                <video
                  src={api.clipUrl(current.event_id)}
                  controls
                  preload="metadata"
                  style={{
                    width: '100%',
                    borderRadius: 'var(--r-sm)',
                    border: '1px solid var(--line)',
                    background: '#050609',
                  }}
                />
              ) : (
                <div
                  className="row"
                  style={{
                    padding: 'var(--s3)',
                    border: '1px dashed var(--line)',
                    borderRadius: 'var(--r-sm)',
                    fontSize: 'var(--fs-sm)',
                    color: 'var(--text-3)',
                    gap: 'var(--s2)',
                  }}
                >
                  {manifest?.clip.pending ? (
                    <>
                      <span className="skeleton" style={{ width: 12, height: 12, borderRadius: 2 }} />
                      <span>Clip is still being written — reopen shortly.</span>
                    </>
                  ) : (
                    <span>No video clip for this incident.</span>
                  )}
                </div>
              )}
            </section>

            {/* Facts */}
            <section className="col" style={{ gap: 'var(--s4)' }}>
              <div>
                <span className="eyebrow">Detection</span>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                    gap: 'var(--s3)',
                    marginTop: 'var(--s2)',
                  }}
                >
                  <Readout label="Camera" value={current.camera_name ?? '—'} size="sm" />
                  <Readout
                    label="Person"
                    value={personLabel(current.person_id)}
                    size="sm"
                  />
                  <Readout
                    label="Confidence"
                    value={percent(current.confidence, 1)}
                    size="sm"
                    tone={current.confidence < 0.6 ? 'var(--sev-medium)' : undefined}
                    title="Detector confidence for this finding"
                  />
                  <Readout label="Zone" value={current.zone_name ?? '—'} size="sm" />
                  <Readout
                    label="Detected at"
                    value={fullStamp(current.timestamp)}
                    size="sm"
                  />
                  {current.video_timestamp !== null && (
                    <Readout
                      label="Video position"
                      value={videoTime(current.video_timestamp)}
                      size="sm"
                    />
                  )}
                  {current.bbox && (
                    <Readout
                      label="Bounding box"
                      value={current.bbox.map((v) => Math.round(v)).join(', ')}
                      size="sm"
                      title="x1, y1, x2, y2 in source-frame pixels"
                    />
                  )}
                  <Readout label="Event ID" value={current.event_id.slice(0, 8)} size="sm" title={current.event_id} />
                </div>
              </div>

              {highlights.length > 0 && (
                <div>
                  <span className="eyebrow">Measurements</span>
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                      gap: 'var(--s3)',
                      marginTop: 'var(--s2)',
                    }}
                  >
                    {highlights.map((key) => (
                      <Readout
                        key={key}
                        label={KEY_LABELS[key] ?? key.replace(/_/g, ' ')}
                        // The detection method is shown as the operator-facing
                        // label ("AI MODEL"), never the raw enum value.
                        value={
                          key === 'ppe_method'
                            ? presentationFor(String(metadata[key]))?.label ??
                              String(metadata[key])
                            : String(metadata[key])
                        }
                        size="sm"
                      />
                    ))}
                  </div>
                </div>
              )}

              <div className="field">
                <label className="eyebrow" htmlFor="incident-notes">
                  Operator notes
                </label>
                <textarea
                  id="incident-notes"
                  className="input"
                  rows={3}
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="What did you verify? Recorded with the status change."
                  style={{ resize: 'vertical', fontFamily: 'var(--font-ui)' }}
                />
              </div>

              <div>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => setShowRaw((v) => !v)}
                  aria-expanded={showRaw}
                >
                  {showRaw ? 'Hide' : 'Show'} full detection metadata
                </button>
                {showRaw && (
                  <pre
                    className="mono scroll-y"
                    style={{
                      marginTop: 'var(--s2)',
                      padding: 'var(--s3)',
                      background: 'var(--bg)',
                      border: '1px solid var(--line)',
                      borderRadius: 'var(--r-sm)',
                      fontSize: 'var(--fs-micro)',
                      maxHeight: 200,
                      color: 'var(--text-2)',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}
                  >
                    {JSON.stringify(metadata, null, 2)}
                  </pre>
                )}
              </div>
            </section>
          </div>
        </div>

        {/* ── Actions ──────────────────────────────────────────────────────── */}
        <footer
          className="row-between"
          style={{
            padding: 'var(--s3) var(--s4)',
            borderTop: '1px solid var(--line)',
            gap: 'var(--s3)',
            flexWrap: 'wrap',
          }}
        >
          <div className="row mono" style={{ gap: 'var(--s3)', fontSize: 'var(--fs-micro)', color: 'var(--text-4)' }}>
            {current.acknowledged_at && (
              <span>ACK {fullStamp(current.acknowledged_at)}</span>
            )}
            {current.resolved_at && <span>CLOSED {fullStamp(current.resolved_at)}</span>}
          </div>
          <div className="row" style={{ gap: 'var(--s2)' }}>
            <a
              className="btn"
              href={api.exportUrl(current.event_id)}
              download
              title="Download snapshot, clip and metadata as a zip"
            >
              Export evidence
            </a>
            {STATUS_ACTIONS.filter((a) => a.status !== current.status).map((action) => (
              <button
                key={action.status}
                className={`btn ${action.primary ? 'btn-primary' : ''}`}
                onClick={() => setStatus(action.status)}
                disabled={busy}
              >
                {action.label}
              </button>
            ))}
          </div>
        </footer>
      </div>
    </div>
  )
}
