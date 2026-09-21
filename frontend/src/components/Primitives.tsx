/**
 * Small shared UI pieces. Each does exactly one job.
 */

import type { CSSProperties, ReactNode } from 'react'
import { severityColour, severityWash } from '../lib/format'
import type { Severity } from '../lib/types'

/* ── Severity badge ──────────────────────────────────────────────────────── */
export function SeverityBadge({
  severity,
  size = 'md',
}: {
  severity: Severity
  size?: 'sm' | 'md'
}) {
  return (
    <span
      className="chip"
      style={{
        color: severityColour[severity],
        background: severityWash[severity],
        borderColor: severityColour[severity],
        fontSize: size === 'sm' ? 'var(--fs-micro)' : 'var(--fs-tiny)',
      }}
    >
      {severity}
    </span>
  )
}

/**
 * The vertical severity gate on an alert row. Colour *and* height both encode
 * severity, so the stack is scannable without relying on colour alone —
 * which matters for colour-blind operators.
 */
export function SeverityGate({ severity }: { severity: Severity }) {
  const heights: Record<Severity, string> = {
    LOW: '34%',
    MEDIUM: '58%',
    HIGH: '80%',
    CRITICAL: '100%',
  }
  return (
    <span
      aria-hidden
      style={{
        position: 'relative',
        width: 3,
        alignSelf: 'stretch',
        minHeight: 28,
        borderRadius: 99,
        background: 'var(--line)',
        flexShrink: 0,
      }}
    >
      <span
        style={{
          position: 'absolute',
          inset: 'auto 0 0 0',
          height: heights[severity],
          borderRadius: 99,
          background: severityColour[severity],
        }}
      />
    </span>
  )
}

/* ── Status dot ──────────────────────────────────────────────────────────── */
export function StatusDot({
  status,
  label,
}: {
  status: string
  label?: string
}) {
  const live = status === 'online'
  const bad = status === 'error' || status === 'offline'
  return (
    <span className="row" style={{ gap: 6 }}>
      <span
        className={`tally ${live ? 'tally-live' : bad && status === 'error' ? 'tally-error' : ''}`}
        style={
          !live && status !== 'error'
            ? { background: status === 'connecting' ? 'var(--sev-medium)' : 'var(--offline)' }
            : undefined
        }
      />
      {label !== undefined && (
        <span
          className="mono"
          style={{
            fontSize: 'var(--fs-micro)',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            color: live ? 'var(--live)' : 'var(--text-3)',
          }}
        >
          {label}
        </span>
      )}
    </span>
  )
}

/* ── Advisory notice ─────────────────────────────────────────────────────── */
/**
 * Shown wherever an advisory event (possible fall, abnormal movement) is
 * presented. These are AI inferences, and the interface says so rather than
 * letting an operator read them as findings.
 */
export function AdvisoryNotice({ compact = false }: { compact?: boolean }) {
  if (compact) {
    return (
      <span
        className="chip"
        title="AI-generated detection — requires human verification"
        style={{
          color: 'var(--advisory)',
          borderColor: 'var(--advisory)',
          background: 'rgba(180, 140, 245, 0.12)',
        }}
      >
        Advisory
      </span>
    )
  }
  return (
    <div
      style={{
        display: 'flex',
        gap: 'var(--s2)',
        padding: 'var(--s3)',
        borderRadius: 'var(--r-sm)',
        border: '1px solid rgba(180, 140, 245, 0.3)',
        background: 'rgba(180, 140, 245, 0.08)',
        fontSize: 'var(--fs-sm)',
        color: 'var(--text-2)',
        lineHeight: 1.55,
      }}
    >
      <span style={{ color: 'var(--advisory)', flexShrink: 0, fontWeight: 700 }}>!</span>
      <span>
        <strong style={{ color: 'var(--advisory)' }}>AI-generated detection.</strong>{' '}
        This result is inferred from movement and body-box geometry. It requires
        human verification and is not a diagnosis or a determination of intent.
      </span>
    </div>
  )
}

/* ── Empty state ─────────────────────────────────────────────────────────── */
export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string
  hint?: string
  action?: ReactNode
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 'var(--s3)',
        padding: 'var(--s12) var(--s6)',
        textAlign: 'center',
      }}
    >
      {/* A viewfinder frame: empty, but unmistakably part of this product. */}
      <svg width="44" height="30" viewBox="0 0 44 30" fill="none" aria-hidden>
        {[
          'M1 9V1h8', 'M35 1h8v8', 'M43 21v8h-8', 'M9 29H1v-8',
        ].map((d) => (
          <path key={d} d={d} stroke="var(--line-strong)" strokeWidth="1.5" />
        ))}
        <circle cx="22" cy="15" r="2.5" fill="var(--line-strong)" />
      </svg>
      <div style={{ fontSize: 'var(--fs-md)', fontWeight: 600 }}>{title}</div>
      {hint && (
        <div
          className="dim"
          style={{ fontSize: 'var(--fs-sm)', maxWidth: '34ch', lineHeight: 1.6 }}
        >
          {hint}
        </div>
      )}
      {action}
    </div>
  )
}

/* ── Error banner ────────────────────────────────────────────────────────── */
export function ErrorBanner({
  message,
  onRetry,
}: {
  message: string
  onRetry?: () => void
}) {
  return (
    <div
      role="alert"
      className="row-between"
      style={{
        padding: 'var(--s3) var(--s4)',
        borderRadius: 'var(--r-sm)',
        border: '1px solid var(--error)',
        background: 'var(--sev-critical-wash)',
        fontSize: 'var(--fs-sm)',
      }}
    >
      <span>{message}</span>
      {onRetry && (
        <button className="btn btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  )
}

/* ── Metric readout ──────────────────────────────────────────────────────── */
/**
 * A labelled value. Used across the HUD and detail panes; the label sits above
 * in the eyebrow style and the value is always mono, so columns of readouts
 * align even as numbers change width.
 */
export function Readout({
  label,
  value,
  tone,
  size = 'md',
  title,
}: {
  label: string
  value: ReactNode
  tone?: string
  size?: 'sm' | 'md' | 'lg'
  title?: string
}) {
  const fontSize =
    size === 'lg' ? 'var(--fs-lg)' : size === 'sm' ? 'var(--fs-sm)' : 'var(--fs-md)'
  return (
    <div className="col" style={{ gap: 2, minWidth: 0 }} title={title}>
      <span className="eyebrow">{label}</span>
      <span
        className="mono truncate"
        style={{ fontSize, fontWeight: 600, color: tone ?? 'var(--text)' }}
      >
        {value}
      </span>
    </div>
  )
}

/* ── Segmented control ───────────────────────────────────────────────────── */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { value: T; label: string; count?: number }[]
  value: T
  onChange: (value: T) => void
  ariaLabel: string
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      style={{
        display: 'inline-flex',
        padding: 2,
        gap: 2,
        background: 'var(--bg-elev)',
        border: '1px solid var(--line)',
        borderRadius: 'var(--r-sm)',
      }}
    >
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(option.value)}
            className="mono"
            style={{
              padding: '0.25rem 0.6rem',
              borderRadius: 2,
              fontSize: 'var(--fs-tiny)',
              fontWeight: 600,
              letterSpacing: '0.05em',
              textTransform: 'uppercase',
              color: active ? '#10140a' : 'var(--text-3)',
              background: active ? 'var(--hivis)' : 'transparent',
              transition: 'background var(--t-fast) var(--ease), color var(--t-fast) var(--ease)',
            }}
          >
            {option.label}
            {option.count !== undefined && (
              <span style={{ opacity: 0.65, marginLeft: 5 }}>{option.count}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
export function Progress({
  value,
  tone = 'var(--hivis)',
  height = 4,
  style,
}: {
  value: number
  tone?: string
  height?: number
  style?: CSSProperties
}) {
  const clamped = Math.max(0, Math.min(1, value))
  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(clamped * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
      style={{
        height,
        borderRadius: 99,
        background: 'var(--line)',
        overflow: 'hidden',
        ...style,
      }}
    >
      <div
        style={{
          height: '100%',
          width: `${clamped * 100}%`,
          background: tone,
          borderRadius: 99,
          transition: 'width var(--t-slow) var(--ease)',
        }}
      />
    </div>
  )
}

/* ── Page header ─────────────────────────────────────────────────────────── */
export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow: string
  title: string
  subtitle?: string
  actions?: ReactNode
}) {
  return (
    <header
      className="row-between"
      style={{ marginBottom: 'var(--s5)', flexWrap: 'wrap', gap: 'var(--s3)' }}
    >
      <div className="col" style={{ gap: 3 }}>
        <span className="eyebrow" style={{ color: 'var(--hivis-dim)' }}>
          {eyebrow}
        </span>
        <h1>{title}</h1>
        {subtitle && (
          <p className="dim" style={{ fontSize: 'var(--fs-sm)', maxWidth: '72ch' }}>
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="row" style={{ gap: 'var(--s2)' }}>{actions}</div>}
    </header>
  )
}
