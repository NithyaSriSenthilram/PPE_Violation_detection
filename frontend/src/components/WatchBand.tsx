/**
 * The Watch Band — this console's signature instrument.
 *
 * One lane per camera, time running left to right across the selected window,
 * and every incident a tick placed at the moment it happened. It answers the
 * question an operator actually opens a surveillance console with — *what has
 * been happening, where, and is it still happening* — in a single glance,
 * which no table of rows or KPI card can do.
 *
 * The form is borrowed from the subject's own world: a multi-track video
 * timeline, with a playhead at "now" that advances in real time.
 *
 * Severity is encoded twice, on purpose. Colour carries it for readers who can
 * use colour; tick *height* carries the same order for those who cannot, and
 * for anyone glancing at it from across a room.
 */

import { useMemo, useState } from 'react'
import { severityColour } from '../lib/format'
import type { Camera, SecurityEvent, Severity } from '../lib/types'

const LANE_HEIGHT = 26
const LABEL_WIDTH = 132

/** Tick height as a fraction of the lane — the redundant severity channel. */
const TICK_SCALE: Record<Severity, number> = {
  LOW: 0.34,
  MEDIUM: 0.55,
  HIGH: 0.78,
  CRITICAL: 1,
}

interface Props {
  cameras: Camera[]
  events: SecurityEvent[]
  /** Window length in hours. */
  hours?: number
  onSelectEvent?: (event: SecurityEvent) => void
}

export function WatchBand({ cameras, events, hours = 12, onSelectEvent }: Props) {
  const [hover, setHover] = useState<{
    x: number
    y: number
    events: SecurityEvent[]
  } | null>(null)

  const now = Date.now()
  const start = now - hours * 3600_000

  // Events grouped per camera lane. Anything without a camera (uploaded-video
  // analysis) gets its own lane rather than being dropped — the operator
  // should still see it on the timeline.
  const lanes = useMemo(() => {
    const byCamera = new Map<string, SecurityEvent[]>()
    for (const camera of cameras) byCamera.set(camera.camera_id, [])
    const uploads: SecurityEvent[] = []

    for (const event of events) {
      const at = new Date(event.timestamp).getTime()
      if (at < start) continue
      if (event.camera_id && byCamera.has(event.camera_id)) {
        byCamera.get(event.camera_id)!.push(event)
      } else {
        uploads.push(event)
      }
    }

    const result = cameras.map((camera) => ({
      id: camera.camera_id,
      name: camera.name,
      status: camera.status,
      events: byCamera.get(camera.camera_id) ?? [],
    }))
    if (uploads.length) {
      result.push({
        id: '__uploads__',
        name: 'Uploaded video',
        status: 'idle',
        events: uploads,
      })
    }
    return result
  }, [cameras, events, start])

  const position = (timestamp: string): number => {
    const at = new Date(timestamp).getTime()
    return Math.max(0, Math.min(100, ((at - start) / (now - start)) * 100))
  }

  // Hour gridlines, thinned so a 30-day window does not draw 720 rules.
  const gridlines = useMemo(() => {
    const step = hours <= 6 ? 1 : hours <= 24 ? 3 : Math.ceil(hours / 8)
    const marks: { fraction: number; label: string }[] = []
    for (let h = hours; h >= 0; h -= step) {
      const at = new Date(now - h * 3600_000)
      marks.push({
        fraction: ((hours - h) / hours) * 100,
        label:
          hours > 48
            ? at.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
            : `${String(at.getHours()).padStart(2, '0')}:00`,
      })
    }
    return marks
  }, [hours, now])

  if (!lanes.length) {
    return (
      <div
        style={{
          padding: 'var(--s6)',
          textAlign: 'center',
          color: 'var(--text-4)',
          fontSize: 'var(--fs-sm)',
        }}
      >
        Add a camera to start building a timeline.
      </div>
    )
  }

  return (
    <div style={{ position: 'relative' }}>
      {/* Time axis */}
      <div
        style={{
          display: 'flex',
          marginLeft: LABEL_WIDTH,
          borderBottom: '1px solid var(--line-soft)',
          paddingBottom: 3,
          marginBottom: 3,
          position: 'relative',
          height: 14,
        }}
      >
        {gridlines.map((mark) => (
          <span
            key={mark.fraction}
            className="mono"
            style={{
              position: 'absolute',
              left: `${mark.fraction}%`,
              transform:
                mark.fraction > 92
                  ? 'translateX(-100%)'
                  : mark.fraction < 4
                    ? 'none'
                    : 'translateX(-50%)',
              fontSize: 'var(--fs-micro)',
              color: 'var(--text-4)',
              whiteSpace: 'nowrap',
            }}
          >
            {mark.label}
          </span>
        ))}
      </div>

      {/* Lanes */}
      <div className="col" style={{ gap: 2 }}>
        {lanes.map((lane) => (
          <div key={lane.id} style={{ display: 'flex', alignItems: 'center' }}>
            <div
              className="row truncate"
              style={{
                width: LABEL_WIDTH,
                paddingRight: 'var(--s3)',
                gap: 6,
                flexShrink: 0,
              }}
            >
              <span
                className="tally"
                style={{
                  background:
                    lane.status === 'online' ? 'var(--live)' : 'var(--offline)',
                }}
              />
              <span
                className="truncate"
                style={{ fontSize: 'var(--fs-tiny)', color: 'var(--text-2)' }}
                title={lane.name}
              >
                {lane.name}
              </span>
            </div>

            <div
              style={{
                position: 'relative',
                flex: 1,
                height: LANE_HEIGHT,
                background: 'var(--bg-elev)',
                borderRadius: 2,
                overflow: 'hidden',
              }}
              onMouseLeave={() => setHover(null)}
            >
              {/* Gridlines behind the ticks */}
              {gridlines.map((mark) => (
                <span
                  key={mark.fraction}
                  aria-hidden
                  style={{
                    position: 'absolute',
                    left: `${mark.fraction}%`,
                    top: 0,
                    bottom: 0,
                    width: 1,
                    background: 'var(--grid)',
                  }}
                />
              ))}

              {lane.events.map((event) => {
                const severity = event.severity
                const scale = TICK_SCALE[severity] ?? 0.5
                return (
                  <button
                    key={event.event_id}
                    onClick={() => onSelectEvent?.(event)}
                    onMouseEnter={(e) => {
                      const box = (
                        e.currentTarget.parentElement as HTMLElement
                      ).getBoundingClientRect()
                      setHover({
                        x: e.clientX - box.left + LABEL_WIDTH,
                        y: 0,
                        events: [event],
                      })
                    }}
                    aria-label={`${event.severity} ${event.label} at ${new Date(event.timestamp).toLocaleTimeString()}`}
                    title={`${event.label} · ${event.severity}`}
                    style={{
                      position: 'absolute',
                      left: `${position(event.timestamp)}%`,
                      // Height encodes severity as well as colour.
                      height: `${scale * 100}%`,
                      bottom: 0,
                      width: 3,
                      marginLeft: -1.5,
                      borderRadius: 1,
                      background: severityColour[severity],
                      // Unresolved incidents stay at full strength; handled
                      // ones recede, so the band shows outstanding work.
                      opacity:
                        event.status === 'RESOLVED' || event.status === 'DISMISSED'
                          ? 0.32
                          : 1,
                      cursor: 'pointer',
                      padding: 0,
                    }}
                  />
                )
              })}

              {/* Playhead at "now" */}
              <span
                aria-hidden
                style={{
                  position: 'absolute',
                  right: 0,
                  top: 0,
                  bottom: 0,
                  width: 1,
                  background: 'var(--hivis)',
                  opacity: 0.55,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Footer: legend + window */}
      <div
        className="row-between"
        style={{ marginTop: 'var(--s3)', marginLeft: LABEL_WIDTH, gap: 'var(--s4)' }}
      >
        <div className="row" style={{ gap: 'var(--s4)', flexWrap: 'wrap' }}>
          {(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as Severity[]).map((severity) => (
            <span key={severity} className="row" style={{ gap: 5 }}>
              <span
                aria-hidden
                style={{
                  width: 3,
                  height: 12 * TICK_SCALE[severity] + 3,
                  borderRadius: 1,
                  background: severityColour[severity],
                }}
              />
              <span style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-3)' }}>
                {severity}
              </span>
            </span>
          ))}
        </div>
        <span className="mono" style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-4)' }}>
          LAST {hours}H · {events.length} INCIDENT{events.length === 1 ? '' : 'S'}
        </span>
      </div>

      {hover && hover.events[0] && (
        <div
          className="glass"
          style={{
            position: 'absolute',
            left: Math.min(hover.x + 8, 560),
            top: -6,
            padding: 'var(--s2) var(--s3)',
            borderRadius: 'var(--r-sm)',
            pointerEvents: 'none',
            zIndex: 6,
            maxWidth: 280,
            boxShadow: 'var(--shadow-md)',
          }}
        >
          <div className="row" style={{ gap: 6, marginBottom: 2 }}>
            <span
              aria-hidden
              style={{
                width: 7,
                height: 7,
                borderRadius: 2,
                background: severityColour[hover.events[0].severity],
              }}
            />
            <span style={{ fontSize: 'var(--fs-tiny)', fontWeight: 700 }}>
              {hover.events[0].label}
            </span>
          </div>
          <div
            className="mono"
            style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-3)' }}
          >
            {new Date(hover.events[0].timestamp).toLocaleTimeString()} ·{' '}
            {hover.events[0].severity} · {hover.events[0].status}
          </div>
        </div>
      )}
    </div>
  )
}
