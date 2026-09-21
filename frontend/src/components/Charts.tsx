/**
 * Charts, hand-built in SVG.
 *
 * No charting library: the four forms here are simple, and owning the SVG is
 * what lets every mark obey the same rules — thin marks, a 2px surface gap
 * between fills, recessive axes, direct labels rather than a number on every
 * point, and a hover layer on all of them.
 *
 * Colour assignments come from validated palettes (see tokens.css): three
 * categorical slots for nominal families, a single-hue ordinal ramp for
 * severity. Text always wears text tokens, never the series colour.
 */

import { useId, useMemo, useState } from 'react'
import type { CameraActivity, TimeBucket } from '../lib/types'
import { compactNumber, eventLabel } from '../lib/format'

/* ══════════════════════════════════════════════════════════════════════════
   Event families
   The eight event types collapse into three families so the stacked chart
   uses only all-pairs-CVD-safe slots. The grouping is operational: what an
   operator would escalate to different people.
   ══════════════════════════════════════════════════════════════════════════ */
export const FAMILIES = [
  {
    key: 'ppe',
    label: 'PPE',
    colour: 'var(--series-1)',
    types: ['PPE_VIOLATION', 'MISSING_HELMET', 'MISSING_VEST'],
  },
  {
    key: 'perimeter',
    label: 'Perimeter',
    colour: 'var(--series-2)',
    types: ['RESTRICTED_AREA'],
  },
  {
    key: 'behaviour',
    label: 'Behaviour',
    colour: 'var(--series-3)',
    types: [
      'LOITERING',
      'ABNORMAL_MOVEMENT',
      'POSSIBLE_FALL',
      'CROWD_ANOMALY',
    ],
  },
] as const

export function familyOf(eventType: string): (typeof FAMILIES)[number] {
  return (
    FAMILIES.find((f) => (f.types as readonly string[]).includes(eventType)) ??
    FAMILIES[2]
  )
}

export const SEVERITY_FILL: Record<string, string> = {
  LOW: 'var(--sev-fill-low)',
  MEDIUM: 'var(--sev-fill-medium)',
  HIGH: 'var(--sev-fill-high)',
  CRITICAL: 'var(--sev-fill-critical)',
}

/* ── Legend ──────────────────────────────────────────────────────────────── */
export function Legend({
  items,
}: {
  items: { label: string; colour: string; value?: number }[]
}) {
  return (
    <ul
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 'var(--s4)',
        listStyle: 'none',
        margin: 0,
      }}
    >
      {items.map((item) => (
        <li key={item.label} className="row" style={{ gap: 6 }}>
          <span
            aria-hidden
            style={{
              width: 9,
              height: 9,
              borderRadius: 2,
              background: item.colour,
              flexShrink: 0,
            }}
          />
          <span style={{ fontSize: 'var(--fs-tiny)', color: 'var(--text-2)' }}>
            {item.label}
          </span>
          {item.value !== undefined && (
            <span className="mono dim-2" style={{ fontSize: 'var(--fs-tiny)' }}>
              {compactNumber(item.value)}
            </span>
          )}
        </li>
      ))}
    </ul>
  )
}

/* ── Tooltip ─────────────────────────────────────────────────────────────── */
interface TipState {
  x: number
  y: number
  title: string
  rows: { label: string; value: string; colour?: string }[]
}

function Tooltip({ tip, hostWidth }: { tip: TipState; hostWidth: number }) {
  // Flip before the tooltip would run off the right edge of the plot.
  const flip = tip.x > hostWidth - 150
  return (
    <div
      role="tooltip"
      className="glass"
      style={{
        position: 'absolute',
        left: flip ? undefined : tip.x + 12,
        right: flip ? hostWidth - tip.x + 12 : undefined,
        top: Math.max(0, tip.y - 12),
        padding: 'var(--s2) var(--s3)',
        borderRadius: 'var(--r-sm)',
        pointerEvents: 'none',
        minWidth: 120,
        zIndex: 5,
        boxShadow: 'var(--shadow-md)',
      }}
    >
      <div
        className="mono"
        style={{
          fontSize: 'var(--fs-micro)',
          letterSpacing: '0.08em',
          color: 'var(--text-3)',
          marginBottom: 4,
          textTransform: 'uppercase',
        }}
      >
        {tip.title}
      </div>
      {tip.rows.map((row) => (
        <div
          key={row.label}
          className="row-between"
          style={{ gap: 'var(--s3)', fontSize: 'var(--fs-tiny)' }}
        >
          <span className="row" style={{ gap: 5 }}>
            {row.colour && (
              <span
                aria-hidden
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: 2,
                  background: row.colour,
                }}
              />
            )}
            <span style={{ color: 'var(--text-2)' }}>{row.label}</span>
          </span>
          <span className="mono" style={{ fontWeight: 600 }}>
            {row.value}
          </span>
        </div>
      ))}
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════════
   Stacked column chart: events over time, by family
   ══════════════════════════════════════════════════════════════════════════ */
export function EventsOverTime({
  buckets,
  granularity,
  height = 200,
}: {
  buckets: TimeBucket[]
  granularity: string
  height?: number
}) {
  const [tip, setTip] = useState<TipState | null>(null)
  const [width, setWidth] = useState(760)
  const clipId = useId()

  const padding = { top: 10, right: 8, bottom: 22, left: 34 }
  const plotW = Math.max(40, width - padding.left - padding.right)
  const plotH = Math.max(40, height - padding.top - padding.bottom)

  const series = useMemo(() => {
    return buckets.map((bucket) => {
      const values = FAMILIES.map((family) => ({
        family,
        value: family.types.reduce(
          (sum, type) => sum + (bucket.by_type[type] ?? 0),
          0,
        ),
      }))
      return { bucket, values, total: values.reduce((s, v) => s + v.value, 0) }
    })
  }, [buckets])

  const max = Math.max(1, ...series.map((s) => s.total))
  const ticks = niceTicks(max, 3)
  const scaleMax = ticks[ticks.length - 1]

  // Thin marks: cap bar width and keep a real gap between columns.
  const slot = plotW / Math.max(1, series.length)
  const barW = Math.max(2, Math.min(22, slot - 3))

  if (!buckets.length) {
    return <ChartEmpty height={height} message="No events in this window" />
  }

  return (
    <figure style={{ margin: 0, position: 'relative' }}>
      <Measure onWidth={setWidth}>
        <svg
          width="100%"
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Events over time by family, ${granularity} buckets`}
          onMouseLeave={() => setTip(null)}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x={0} y={0} width={width} height={height} />
            </clipPath>
          </defs>

          {/* Recessive grid: hairlines behind the data, no vertical rules. */}
          {ticks.map((tick) => {
            const y = padding.top + plotH - (tick / scaleMax) * plotH
            return (
              <g key={tick}>
                <line
                  x1={padding.left}
                  x2={width - padding.right}
                  y1={y}
                  y2={y}
                  stroke="var(--grid)"
                  strokeWidth={1}
                />
                <text
                  x={padding.left - 6}
                  y={y + 3}
                  textAnchor="end"
                  fontFamily="var(--font-mono)"
                  fontSize={9}
                  fill="var(--text-4)"
                >
                  {tick}
                </text>
              </g>
            )
          })}

          <g clipPath={`url(#${clipId})`}>
            {series.map((entry, index) => {
              const x = padding.left + index * slot + (slot - barW) / 2
              let cursor = padding.top + plotH
              return (
                <g
                  key={entry.bucket.bucket}
                  onMouseMove={(event) => {
                    const box = event.currentTarget.ownerSVGElement?.getBoundingClientRect()
                    if (!box) return
                    setTip({
                      x: event.clientX - box.left,
                      y: event.clientY - box.top,
                      title: bucketLabel(entry.bucket.bucket, granularity, true),
                      rows: [
                        ...entry.values
                          .filter((v) => v.value > 0)
                          .map((v) => ({
                            label: v.family.label,
                            value: String(v.value),
                            colour: v.family.colour,
                          })),
                        { label: 'Total', value: String(entry.total) },
                      ],
                    })
                  }}
                >
                  {/* Invisible full-height hit target: bigger than the mark. */}
                  <rect
                    x={padding.left + index * slot}
                    y={padding.top}
                    width={slot}
                    height={plotH}
                    fill="transparent"
                  />
                  {entry.values.map((item) => {
                    if (item.value <= 0) return null
                    const barH = (item.value / scaleMax) * plotH
                    cursor -= barH
                    const y = cursor
                    // 2px surface gap between stacked segments.
                    cursor -= 2
                    return (
                      <rect
                        key={item.family.key}
                        x={x}
                        y={y}
                        width={barW}
                        height={Math.max(1, barH)}
                        fill={item.family.colour}
                        rx={2}
                      />
                    )
                  })}
                </g>
              )
            })}
          </g>

          {/* Baseline only — no box frame. */}
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={padding.top + plotH}
            y2={padding.top + plotH}
            stroke="var(--axis)"
            strokeWidth={1}
          />

          {/* Sparse x labels: never more than ~7, so nothing collides. */}
          {series.map((entry, index) => {
            const step = Math.ceil(series.length / 7)
            if (index % step !== 0) return null
            return (
              <text
                key={entry.bucket.bucket}
                x={padding.left + index * slot + slot / 2}
                y={height - 6}
                textAnchor="middle"
                fontFamily="var(--font-mono)"
                fontSize={9}
                fill="var(--text-4)"
              >
                {bucketLabel(entry.bucket.bucket, granularity)}
              </text>
            )
          })}
        </svg>
        {tip && <Tooltip tip={tip} hostWidth={width} />}
      </Measure>
      <figcaption style={{ marginTop: 'var(--s3)' }}>
        <Legend
          items={FAMILIES.map((family) => ({
            label: family.label,
            colour: family.colour,
            value: series.reduce(
              (sum, s) =>
                sum + (s.values.find((v) => v.family.key === family.key)?.value ?? 0),
              0,
            ),
          }))}
        />
      </figcaption>
    </figure>
  )
}

/* ══════════════════════════════════════════════════════════════════════════
   Horizontal bars: one hue, direct-labelled. Used for per-type breakdowns
   and camera activity — bar length already encodes the value, so colouring
   each bar differently would spend the identity channel re-encoding it.
   ══════════════════════════════════════════════════════════════════════════ */
export function RankedBars({
  rows,
  colour = 'var(--series-1)',
  emptyMessage = 'No data in this window',
  maxRows = 8,
  unit,
}: {
  rows: { label: string; value: number; hint?: string }[]
  colour?: string
  emptyMessage?: string
  maxRows?: number
  unit?: string
}) {
  const sorted = useMemo(
    () => [...rows].sort((a, b) => b.value - a.value).slice(0, maxRows),
    [rows, maxRows],
  )
  const max = Math.max(1, ...sorted.map((r) => r.value))

  if (!sorted.length || max === 0) {
    return <ChartEmpty height={140} message={emptyMessage} />
  }

  return (
    <div className="col" style={{ gap: 'var(--s3)' }}>
      {sorted.map((row) => (
        <div key={row.label} className="col" style={{ gap: 4 }}>
          <div className="row-between" style={{ gap: 'var(--s3)' }}>
            <span
              className="truncate"
              style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-2)' }}
              title={row.hint ?? row.label}
            >
              {row.label}
            </span>
            {/* Direct label — the value is always visible, no hover needed. */}
            <span
              className="mono"
              style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, flexShrink: 0 }}
            >
              {compactNumber(row.value)}
              {unit && <span className="dim-2"> {unit}</span>}
            </span>
          </div>
          <div
            style={{
              height: 6,
              background: 'var(--line-soft)',
              borderRadius: 3,
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${(row.value / max) * 100}%`,
                height: '100%',
                background: colour,
                borderRadius: 3,
                transition: 'width var(--t-slow) var(--ease)',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

/* ══════════════════════════════════════════════════════════════════════════
   Severity distribution: ordinal ramp + always-visible labels
   ══════════════════════════════════════════════════════════════════════════ */
export function SeverityBars({
  counts,
}: {
  counts: Record<string, number>
}) {
  const order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
  const total = order.reduce((sum, key) => sum + (counts[key] ?? 0), 0)

  if (!total) return <ChartEmpty height={120} message="No events in this window" />

  return (
    <div className="col" style={{ gap: 'var(--s3)' }}>
      {/* A single proportional bar reads faster than four separate ones when
          the question is "what is the mix". 2px gaps between fills. */}
      <div style={{ display: 'flex', gap: 2, height: 8 }}>
        {order.map((key) => {
          const value = counts[key] ?? 0
          if (!value) return null
          return (
            <div
              key={key}
              title={`${key}: ${value}`}
              style={{
                flex: value,
                // A category with a single event must still be visible and
                // hoverable rather than collapsing to a hairline.
                minWidth: 6,
                background: SEVERITY_FILL[key],
                borderRadius: 2,
              }}
            />
          )
        })}
      </div>
      <div className="col" style={{ gap: 'var(--s2)' }}>
        {order.map((key) => {
          const value = counts[key] ?? 0
          return (
            <div key={key} className="row-between" style={{ gap: 'var(--s3)' }}>
              <span className="row" style={{ gap: 6 }}>
                <span
                  aria-hidden
                  style={{
                    width: 9,
                    height: 9,
                    borderRadius: 2,
                    background: SEVERITY_FILL[key],
                  }}
                />
                <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--text-2)' }}>
                  {key}
                </span>
              </span>
              <span className="mono" style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}>
                {value}
                <span className="dim-2" style={{ marginLeft: 6 }}>
                  {total ? `${Math.round((value / total) * 100)}%` : '0%'}
                </span>
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── Camera activity ─────────────────────────────────────────────────────── */
export function CameraActivityChart({ rows }: { rows: CameraActivity[] }) {
  return (
    <RankedBars
      rows={rows.map((row) => ({
        label: row.camera_name,
        value: row.event_count,
        hint: `${row.camera_name} — ${row.status}`,
      }))}
      colour="var(--series-2)"
      emptyMessage="No camera activity recorded"
      unit="events"
    />
  )
}

/* ── Type breakdown ──────────────────────────────────────────────────────── */
export function TypeBreakdown({ byType }: { byType: Record<string, number> }) {
  return (
    <RankedBars
      rows={Object.entries(byType).map(([type, value]) => ({
        label: eventLabel(type),
        value,
      }))}
      colour="var(--series-1)"
      emptyMessage="No events in this window"
    />
  )
}

/* ══════════════════════════════════════════════════════════════════════════
   Helpers
   ══════════════════════════════════════════════════════════════════════════ */
function ChartEmpty({ height, message }: { height: number; message: string }) {
  return (
    <div
      style={{
        height,
        display: 'grid',
        placeItems: 'center',
        border: '1px dashed var(--line)',
        borderRadius: 'var(--r-sm)',
        color: 'var(--text-4)',
        fontSize: 'var(--fs-sm)',
      }}
    >
      {message}
    </div>
  )
}

/** Width observer — charts need real pixels to place labels honestly. */
function Measure({
  children,
  onWidth,
}: {
  children: React.ReactNode
  onWidth: (width: number) => void
}) {
  const ref = (element: HTMLDivElement | null) => {
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width
      if (w) onWidth(w)
    })
    observer.observe(element)
  }
  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      {children}
    </div>
  )
}

/** Round axis maxima so ticks land on readable numbers. */
function niceTicks(max: number, count: number): number[] {
  const raw = max / count
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10
  const ticks: number[] = []
  for (let value = step; value <= Math.ceil(max / step) * step; value += step) {
    ticks.push(value)
  }
  return ticks.length ? ticks : [1]
}

function bucketLabel(bucket: string, granularity: string, long = false): string {
  const date = new Date(bucket)
  if (Number.isNaN(date.getTime())) return bucket
  if (granularity === 'day') {
    return long
      ? date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
      : date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  }
  return long
    ? `${date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })} ${String(date.getHours()).padStart(2, '0')}:00`
    : `${String(date.getHours()).padStart(2, '0')}:00`
}
