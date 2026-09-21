/**
 * Alert Centre — triage.
 *
 * Filters sit in one row above the stack. Severity is a segmented control with
 * live counts, because "how many criticals are open" is the question and the
 * filter should answer it before you click.
 */

import { useMemo, useState } from 'react'
import { api } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import { EVENT_TYPES, eventLabel } from '../lib/format'
import type { EventStatus, SecurityEvent, Severity } from '../lib/types'
import { AlertRow } from '../components/AlertRow'
import { IncidentModal } from '../components/IncidentModal'
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  Segmented,
} from '../components/Primitives'

type SeverityFilter = 'ALL' | Severity
const PAGE_SIZE = 25

export function Alerts() {
  const [severity, setSeverity] = useState<SeverityFilter>('ALL')
  const [cameraId, setCameraId] = useState('')
  const [eventType, setEventType] = useState('')
  const [status, setStatus] = useState<EventStatus | ''>('')
  const [date, setDate] = useState('')
  const [search, setSearch] = useState('')
  const [includeResolved, setIncludeResolved] = useState(false)
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<SecurityEvent | null>(null)

  const { data: cameras } = usePolling(() => api.cameras(), 30_000)

  // A single date picker becomes a whole-day window server-side.
  const range = useMemo(() => {
    if (!date) return {}
    const start = new Date(`${date}T00:00:00`)
    const end = new Date(`${date}T23:59:59`)
    return { start: start.toISOString(), end: end.toISOString() }
  }, [date])

  const params = useMemo(
    () => ({
      severity: severity === 'ALL' ? undefined : severity,
      camera_id: cameraId || undefined,
      event_type: eventType || undefined,
      status: status || undefined,
      search: search.trim() || undefined,
      include_resolved: includeResolved,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      ...range,
    }),
    [severity, cameraId, eventType, status, search, includeResolved, page, range],
  )

  const { data, error, loading, refresh } = usePolling(
    () => api.alerts(params),
    10_000,
    [JSON.stringify(params)],
  )

  // Counts for the segmented control: one small unfiltered request.
  const { data: counts } = usePolling(
    () => api.statistics({ range: '30d' }),
    30_000,
  )
  const bySeverity = counts?.by_severity ?? {}

  function reset() {
    setSeverity('ALL')
    setCameraId('')
    setEventType('')
    setStatus('')
    setDate('')
    setSearch('')
    setIncludeResolved(false)
    setPage(0)
  }

  const filtersActive =
    severity !== 'ALL' || cameraId || eventType || status || date || search || includeResolved

  return (
    <>
      <PageHeader
        eyebrow="Triage"
        title="Alert Centre"
        subtitle="Every incident the system has raised. Acknowledge to claim it, resolve when it is dealt with."
        actions={
          filtersActive ? (
            <button className="btn" onClick={reset}>
              Clear filters
            </button>
          ) : undefined
        }
      />

      {/* ── Filter row ───────────────────────────────────────────────────── */}
      <div
        className="panel"
        style={{ padding: 'var(--s3)', marginBottom: 'var(--s4)' }}
      >
        <div
          className="row"
          style={{ gap: 'var(--s3)', flexWrap: 'wrap', alignItems: 'flex-end' }}
        >
          <Segmented
            ariaLabel="Filter by severity"
            value={severity}
            onChange={(v) => {
              setSeverity(v)
              setPage(0)
            }}
            options={[
              { value: 'ALL', label: 'All' },
              { value: 'CRITICAL', label: 'Critical', count: bySeverity.CRITICAL ?? 0 },
              { value: 'HIGH', label: 'High', count: bySeverity.HIGH ?? 0 },
              { value: 'MEDIUM', label: 'Medium', count: bySeverity.MEDIUM ?? 0 },
              { value: 'LOW', label: 'Low', count: bySeverity.LOW ?? 0 },
            ]}
          />

          <div className="field" style={{ minWidth: 150 }}>
            <label className="eyebrow" htmlFor="f-camera">Camera</label>
            <select
              id="f-camera"
              className="select"
              value={cameraId}
              onChange={(e) => {
                setCameraId(e.target.value)
                setPage(0)
              }}
            >
              <option value="">All cameras</option>
              {(cameras ?? []).map((camera) => (
                <option key={camera.camera_id} value={camera.camera_id}>
                  {camera.name}
                </option>
              ))}
            </select>
          </div>

          <div className="field" style={{ minWidth: 160 }}>
            <label className="eyebrow" htmlFor="f-type">Event type</label>
            <select
              id="f-type"
              className="select"
              value={eventType}
              onChange={(e) => {
                setEventType(e.target.value)
                setPage(0)
              }}
            >
              <option value="">All types</option>
              {EVENT_TYPES.map((type) => (
                <option key={type} value={type}>
                  {eventLabel(type)}
                </option>
              ))}
            </select>
          </div>

          <div className="field" style={{ minWidth: 130 }}>
            <label className="eyebrow" htmlFor="f-status">Status</label>
            <select
              id="f-status"
              className="select"
              value={status}
              onChange={(e) => {
                setStatus(e.target.value as EventStatus | '')
                setPage(0)
              }}
            >
              <option value="">Any status</option>
              <option value="OPEN">Open</option>
              <option value="ACKNOWLEDGED">Acknowledged</option>
              <option value="RESOLVED">Resolved</option>
              <option value="DISMISSED">Dismissed</option>
            </select>
          </div>

          <div className="field" style={{ minWidth: 130 }}>
            <label className="eyebrow" htmlFor="f-date">Date</label>
            <input
              id="f-date"
              type="date"
              className="input"
              value={date}
              onChange={(e) => {
                setDate(e.target.value)
                setPage(0)
              }}
            />
          </div>

          <div className="field grow" style={{ minWidth: 160 }}>
            <label className="eyebrow" htmlFor="f-search">Search</label>
            <input
              id="f-search"
              className="input"
              placeholder="Description or type"
              value={search}
              onChange={(e) => {
                setSearch(e.target.value)
                setPage(0)
              }}
            />
          </div>

          <label
            className="row"
            style={{ gap: 6, fontSize: 'var(--fs-sm)', cursor: 'pointer', paddingBottom: 6 }}
          >
            <input
              type="checkbox"
              checked={includeResolved}
              onChange={(e) => {
                setIncludeResolved(e.target.checked)
                setPage(0)
              }}
              style={{ accentColor: 'var(--hivis)' }}
            />
            <span className="dim">Include closed</span>
          </label>
        </div>
      </div>

      {error && (
        <div style={{ marginBottom: 'var(--s3)' }}>
          <ErrorBanner message={error} onRetry={refresh} />
        </div>
      )}

      {/* ── Stack ────────────────────────────────────────────────────────── */}
      {loading && !data ? (
        <div className="col" style={{ gap: 'var(--s2)' }}>
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 66 }} />
          ))}
        </div>
      ) : !data?.items.length ? (
        <div className="panel">
          <EmptyState
            title={filtersActive ? 'No alerts match these filters' : 'No alerts raised'}
            hint={
              filtersActive
                ? 'Widen the filters, or clear them to see everything.'
                : 'Incidents appear here as soon as the pipeline detects one.'
            }
            action={
              filtersActive ? (
                <button className="btn" onClick={reset}>
                  Clear filters
                </button>
              ) : undefined
            }
          />
        </div>
      ) : (
        <>
          <div className="col alert-list" style={{ gap: 'var(--s2)' }}>
            {data.items.map((event) => (
              <AlertRow key={event.event_id} event={event} onOpen={setSelected} />
            ))}
          </div>

          <div
            className="row-between"
            style={{ marginTop: 'var(--s4)', gap: 'var(--s3)' }}
          >
            <span className="mono dim" style={{ fontSize: 'var(--fs-tiny)' }}>
              {data.offset + 1}–{data.offset + data.items.length} of {data.total}
            </span>
            <div className="row" style={{ gap: 'var(--s2)' }}>
              <button
                className="btn btn-sm"
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                Previous
              </button>
              <button
                className="btn btn-sm"
                disabled={!data.has_more}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}

      {selected && (
        <IncidentModal
          event={selected}
          onClose={() => setSelected(null)}
          onUpdated={refresh}
        />
      )}
    </>
  )
}
