/**
 * Analytics.
 *
 * Every chart here answers a question a safety officer actually asks, and each
 * one also offers its numbers as a table — colour and bar length are never the
 * only way to read a value.
 */

import { useState } from 'react'
import { api } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import { compactNumber, eventLabel, fullStamp } from '../lib/format'
import {
  CameraActivityChart,
  EventsOverTime,
  FAMILIES,
  SeverityBars,
  TypeBreakdown,
  familyOf,
} from '../components/Charts'
import {
  ErrorBanner,
  PageHeader,
  Readout,
  Segmented,
} from '../components/Primitives'

type Range = 'today' | '24h' | '7d' | '30d' | 'custom'

export function Analytics() {
  const [range, setRange] = useState<Range>('7d')
  const [customStart, setCustomStart] = useState('')
  const [customEnd, setCustomEnd] = useState('')
  const [showTables, setShowTables] = useState(false)

  const params =
    range === 'custom' && customStart && customEnd
      ? {
          range: 'custom',
          start: new Date(`${customStart}T00:00:00`).toISOString(),
          end: new Date(`${customEnd}T23:59:59`).toISOString(),
        }
      : { range: range === 'custom' ? '7d' : range }

  const { data, error, loading, refresh } = usePolling(
    () => api.statistics(params),
    30_000,
    [JSON.stringify(params)],
  )

  const familyTotals = FAMILIES.map((family) => ({
    family,
    total: family.types.reduce(
      (sum, type) => sum + (data?.by_type[type] ?? 0),
      0,
    ),
  }))
  const grandTotal = Object.values(data?.by_type ?? {}).reduce((a, b) => a + b, 0)

  return (
    <>
      <PageHeader
        eyebrow="Reporting"
        title="Analytics"
        subtitle="Incident patterns over time, by category and by camera."
        actions={
          <button
            className="btn"
            onClick={() => setShowTables((v) => !v)}
            aria-pressed={showTables}
          >
            {showTables ? 'Hide' : 'Show'} data tables
          </button>
        }
      />

      {/* ── Range ────────────────────────────────────────────────────────── */}
      <div
        className="panel row-between"
        style={{ padding: 'var(--s3)', marginBottom: 'var(--s4)', gap: 'var(--s3)', flexWrap: 'wrap' }}
      >
        <Segmented
          ariaLabel="Reporting range"
          value={range}
          onChange={setRange}
          options={[
            { value: 'today', label: 'Today' },
            { value: '24h', label: '24h' },
            { value: '7d', label: '7 days' },
            { value: '30d', label: '30 days' },
            { value: 'custom', label: 'Custom' },
          ]}
        />
        {range === 'custom' && (
          <div className="row" style={{ gap: 'var(--s2)', alignItems: 'flex-end' }}>
            <div className="field">
              <label className="eyebrow" htmlFor="a-start">From</label>
              <input
                id="a-start"
                type="date"
                className="input"
                value={customStart}
                onChange={(e) => setCustomStart(e.target.value)}
              />
            </div>
            <div className="field">
              <label className="eyebrow" htmlFor="a-end">To</label>
              <input
                id="a-end"
                type="date"
                className="input"
                value={customEnd}
                onChange={(e) => setCustomEnd(e.target.value)}
              />
            </div>
          </div>
        )}
        {data && (
          <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
            {fullStamp(data.range_start)} → {fullStamp(data.range_end)} ·{' '}
            {data.granularity.toUpperCase()} BUCKETS
          </span>
        )}
      </div>

      {error && (
        <div style={{ marginBottom: 'var(--s3)' }}>
          <ErrorBanner message={error} onRetry={refresh} />
        </div>
      )}

      {loading && !data ? (
        <div className="skeleton" style={{ height: 320 }} />
      ) : (
        <div className="col" style={{ gap: 'var(--s4)' }}>
          {/* Summary readouts */}
          <section
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: 'var(--s3)',
            }}
          >
            <div className="panel" style={{ padding: 'var(--s3) var(--s4)' }}>
              <Readout label="Total incidents" value={compactNumber(grandTotal)} size="lg" />
            </div>
            {familyTotals.map(({ family, total }) => (
              <div key={family.key} className="panel" style={{ padding: 'var(--s3) var(--s4)' }}>
                <Readout
                  label={family.label}
                  value={compactNumber(total)}
                  size="lg"
                  tone={total > 0 ? family.colour : undefined}
                />
              </div>
            ))}
            <div className="panel" style={{ padding: 'var(--s3) var(--s4)' }}>
              <Readout
                label="People tracked"
                value={compactNumber(data?.kpis.people_detected ?? 0)}
                size="lg"
              />
            </div>
          </section>

          {/* Events over time */}
          <section className="panel">
            <div className="panel-head">
              <div className="col" style={{ gap: 1 }}>
                <span className="eyebrow">Trend</span>
                <h2 style={{ fontSize: 'var(--fs-md)' }}>Incidents over time</h2>
              </div>
            </div>
            <div className="panel-body">
              <EventsOverTime
                buckets={data?.timeline ?? []}
                granularity={data?.granularity ?? 'hour'}
                height={220}
              />
              {showTables && (
                <DataTable
                  caption="Incidents per bucket by family"
                  head={['Bucket', ...FAMILIES.map((f) => f.label), 'Total']}
                  rows={(data?.timeline ?? [])
                    .filter((b) => b.total > 0)
                    .map((bucket) => [
                      bucket.bucket,
                      ...FAMILIES.map((family) =>
                        String(
                          family.types.reduce(
                            (sum, type) => sum + (bucket.by_type[type] ?? 0),
                            0,
                          ),
                        ),
                      ),
                      String(bucket.total),
                    ])}
                />
              )}
            </div>
          </section>

          {/* Breakdowns */}
          <div
            className="analytics-grid"
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: 'var(--s4)',
              alignItems: 'start',
            }}
          >
            <section className="panel">
              <div className="panel-head">
                <div className="col" style={{ gap: 1 }}>
                  <span className="eyebrow">By category</span>
                  <h2 style={{ fontSize: 'var(--fs-md)' }}>Incident types</h2>
                </div>
              </div>
              <div className="panel-body">
                <TypeBreakdown byType={data?.by_type ?? {}} />
                {showTables && (
                  <DataTable
                    caption="Incidents by type"
                    head={['Type', 'Family', 'Count']}
                    rows={Object.entries(data?.by_type ?? {})
                      .sort((a, b) => b[1] - a[1])
                      .map(([type, count]) => [
                        eventLabel(type),
                        familyOf(type).label,
                        String(count),
                      ])}
                  />
                )}
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <div className="col" style={{ gap: 1 }}>
                  <span className="eyebrow">By severity</span>
                  <h2 style={{ fontSize: 'var(--fs-md)' }}>Severity mix</h2>
                </div>
              </div>
              <div className="panel-body">
                <SeverityBars counts={data?.by_severity ?? {}} />
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <div className="col" style={{ gap: 1 }}>
                  <span className="eyebrow">By source</span>
                  <h2 style={{ fontSize: 'var(--fs-md)' }}>Camera activity</h2>
                </div>
              </div>
              <div className="panel-body">
                <CameraActivityChart rows={data?.camera_activity ?? []} />
                {showTables && (
                  <DataTable
                    caption="Activity by camera"
                    head={['Camera', 'Status', 'Incidents', 'People']}
                    rows={(data?.camera_activity ?? []).map((row) => [
                      row.camera_name,
                      row.status,
                      String(row.event_count),
                      String(row.people_detected),
                    ])}
                  />
                )}
              </div>
            </section>

            <section className="panel">
              <div className="panel-head">
                <div className="col" style={{ gap: 1 }}>
                  <span className="eyebrow">Workflow</span>
                  <h2 style={{ fontSize: 'var(--fs-md)' }}>Handling status</h2>
                </div>
              </div>
              <div className="panel-body">
                <div className="col" style={{ gap: 'var(--s3)' }}>
                  {Object.entries(data?.by_status ?? {}).length === 0 ? (
                    <span className="dim" style={{ fontSize: 'var(--fs-sm)' }}>
                      No incidents in this window.
                    </span>
                  ) : (
                    Object.entries(data?.by_status ?? {})
                      .sort((a, b) => b[1] - a[1])
                      .map(([status, count]) => (
                        <div key={status} className="row-between">
                          <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-2)' }}>
                            {status}
                          </span>
                          <span className="mono" style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}>
                            {count}
                          </span>
                        </div>
                      ))
                  )}
                </div>
              </div>
            </section>
          </div>
        </div>
      )}
    </>
  )
}

/** The table view every chart offers — values readable without colour. */
function DataTable({
  caption,
  head,
  rows,
}: {
  caption: string
  head: string[]
  rows: string[][]
}) {
  if (!rows.length) return null
  return (
    <div
      className="scroll-y"
      style={{ marginTop: 'var(--s4)', maxHeight: 240, overflowX: 'auto' }}
    >
      <table
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontSize: 'var(--fs-tiny)',
        }}
      >
        <caption className="eyebrow" style={{ textAlign: 'left', paddingBottom: 'var(--s2)' }}>
          {caption}
        </caption>
        <thead>
          <tr>
            {head.map((cell) => (
              <th
                key={cell}
                className="eyebrow"
                style={{
                  textAlign: 'left',
                  padding: '4px 8px 4px 0',
                  borderBottom: '1px solid var(--line)',
                  whiteSpace: 'nowrap',
                }}
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={j === 0 ? '' : 'mono'}
                  style={{
                    padding: '4px 8px 4px 0',
                    borderBottom: '1px solid var(--line-soft)',
                    color: j === 0 ? 'var(--text-2)' : 'var(--text)',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
