/**
 * One alert in the stack.
 *
 * Built as a row of instrument readouts rather than a table cell: severity is
 * a gate bar (colour + height), the snapshot is inline so the operator can
 * triage without opening anything, and the age ticks so a stale alert is
 * obvious.
 */

import { api } from '../lib/api'
import { ago, percent, personLabel, shortStamp } from '../lib/format'
import type { SecurityEvent } from '../lib/types'
import { SeverityGate } from './Primitives'
import { PPEMethodBadge } from './PPEMethod'

/** Short, unambiguous status codes — a sliced word reads as a rendering bug. */
const STATUS_CODE: Record<string, string> = {
  OPEN: 'OPEN',
  ACKNOWLEDGED: 'ACK',
  RESOLVED: 'CLOSED',
  DISMISSED: 'DISM',
}

export function AlertRow({
  event,
  onOpen,
  showCamera = true,
}: {
  event: SecurityEvent
  onOpen: (event: SecurityEvent) => void
  showCamera?: boolean
}) {
  const handled = event.status === 'RESOLVED' || event.status === 'DISMISSED'
  // PPE findings state their provenance in the row itself: a colour
  // estimate must never look like a trained-model detection at a glance.
  const ppeMethod = event.detection_metadata?.ppe_method as string | undefined

  return (
    <button
      onClick={() => onOpen(event)}
      className="alert-row"
      style={{
        display: 'flex',
        alignItems: 'stretch',
        gap: 'var(--s3)',
        width: '100%',
        padding: 'var(--s3)',
        textAlign: 'left',
        borderRadius: 'var(--r-sm)',
        border: '1px solid var(--line-soft)',
        background: 'var(--surface)',
        opacity: handled ? 0.62 : 1,
        transition:
          'background var(--t-fast) var(--ease), border-color var(--t-fast) var(--ease), opacity var(--t-fast) var(--ease)',
      }}
    >
      <SeverityGate severity={event.severity} />

      {/* Snapshot thumbnail: triage without a click. */}
      <div
        className="alert-thumb"
        style={{
          width: 76,
          height: 44,
          flexShrink: 0,
          borderRadius: 2,
          overflow: 'hidden',
          background: '#050609',
          border: '1px solid var(--line)',
          display: 'grid',
          placeItems: 'center',
        }}
      >
        {event.has_snapshot ? (
          <img
            src={api.snapshotUrl(event.event_id)}
            alt=""
            loading="lazy"
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
            NO IMG
          </span>
        )}
      </div>

      {/* Identity */}
      <div className="col grow" style={{ gap: 2, minWidth: 0 }}>
        <div className="row" style={{ gap: 'var(--s2)', minWidth: 0 }}>
          <span
            className="truncate"
            style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}
          >
            {event.label}
          </span>
          {event.advisory && (
            <span
              className="chip"
              style={{
                color: 'var(--advisory)',
                borderColor: 'var(--advisory)',
                background: 'rgba(180,140,245,0.1)',
              }}
            >
              Advisory
            </span>
          )}
          {ppeMethod && <PPEMethodBadge method={ppeMethod} short />}
        </div>
        <span
          className="truncate dim"
          style={{ fontSize: 'var(--fs-tiny)' }}
          title={event.description}
        >
          {event.description}
        </span>
      </div>

      {/* Readouts — fixed columns so the stack scans vertically. */}
      <div
        className="row alert-meta"
        style={{ gap: 'var(--s4)', flexShrink: 0, alignItems: 'center' }}
      >
        {showCamera && (
          <Cell
            label="Camera"
            value={event.camera_name ?? 'Upload'}
            width={124}
            className="alert-cell-camera"
          />
        )}
        <Cell
          label="Person"
          value={personLabel(event.person_id)}
          width={48}
          className="alert-cell-person"
        />
        <Cell
          label="Conf"
          value={percent(event.confidence)}
          width={44}
          className="alert-cell-conf"
          tone={event.confidence < 0.6 ? 'var(--sev-medium)' : undefined}
        />
        <Cell
          label="Status"
          value={STATUS_CODE[event.status] ?? event.status}
          width={52}
          tone={event.status === 'OPEN' ? 'var(--sev-high)' : 'var(--text-3)'}
          title={event.status}
        />
        <Cell
          label="Age"
          value={ago(event.timestamp)}
          width={40}
          title={shortStamp(event.timestamp)}
        />
      </div>
    </button>
  )
}

function Cell({
  label,
  value,
  width,
  tone,
  title,
  className,
}: {
  label: string
  value: string
  width: number
  tone?: string
  title?: string
  className?: string
}) {
  return (
    <div
      className={`col ${className ?? ''}`}
      style={{ gap: 1, width, minWidth: 0 }}
      title={title}
    >
      <span className="eyebrow" style={{ fontSize: '0.5625rem' }}>
        {label}
      </span>
      <span
        className="mono truncate"
        style={{ fontSize: 'var(--fs-tiny)', fontWeight: 600, color: tone }}
      >
        {value}
      </span>
    </div>
  )
}
