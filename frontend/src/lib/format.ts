/**
 * Formatting helpers.
 *
 * Surveillance systems are read at a glance, so values are formatted for
 * scanning: fixed-width timecodes, short relative ages, and severity/type
 * labels that stay the same length wherever they appear.
 */

import type { EventType, Severity } from './types'

const pad = (n: number, width = 2) => String(Math.floor(n)).padStart(width, '0')

/** HH:MM:SS in the viewer's local time — the console's primary clock. */
export function timecode(date: Date | string | number = new Date()): string {
  const d = new Date(date)
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

/** YYYY-MM-DD HH:MM:SS — for records and tooltips. */
export function fullStamp(value: string | Date | number): string {
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return '—'
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${timecode(d)}`
}

/** Short date + time, e.g. "29 Aug 13:46:02". */
export function shortStamp(value: string | Date | number): string {
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return '—'
  const month = d.toLocaleString('en-GB', { month: 'short' })
  return `${d.getDate()} ${month} ${timecode(d)}`
}

/** Compact age: "12s", "4m", "3h", "2d". */
export function ago(value: string | Date | number): string {
  const then = new Date(value).getTime()
  if (Number.isNaN(then)) return '—'
  const seconds = Math.max(0, (Date.now() - then) / 1000)
  if (seconds < 60) return `${Math.floor(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`
  return `${Math.floor(seconds / 86400)}d`
}

/** Duration in seconds → "1m 04s" / "01:04:09". */
export function duration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${pad(h)}:${pad(m)}:${pad(s)}`
  if (m > 0) return `${m}m ${pad(s)}s`
  return `${s}s`
}

/** Video position → "0:14.3". */
export function videoTime(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return '—'
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toFixed(1).padStart(4, '0')}`
}

export function percent(value: number, digits = 0): string {
  return `${(value * 100).toFixed(digits)}%`
}

/** Thousands separators, and k/M above 10k so KPI cards never wrap. */
export function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (Math.abs(value) >= 10_000) return `${(value / 1000).toFixed(1)}k`
  return value.toLocaleString('en-GB')
}

export function bytes(value: number): string {
  if (!value) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)))
  return `${(value / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

/* ── Severity ────────────────────────────────────────────────────────────── */
export const SEVERITIES: Severity[] = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

export const severityColour: Record<Severity, string> = {
  LOW: 'var(--sev-low)',
  MEDIUM: 'var(--sev-medium)',
  HIGH: 'var(--sev-high)',
  CRITICAL: 'var(--sev-critical)',
}

export const severityWash: Record<Severity, string> = {
  LOW: 'var(--sev-low-wash)',
  MEDIUM: 'var(--sev-medium-wash)',
  HIGH: 'var(--sev-high-wash)',
  CRITICAL: 'var(--sev-critical-wash)',
}

export const severityRank: Record<Severity, number> = {
  CRITICAL: 0,
  HIGH: 1,
  MEDIUM: 2,
  LOW: 3,
}

/* ── Event types ─────────────────────────────────────────────────────────── */
export const EVENT_TYPES: EventType[] = [
  'PPE_VIOLATION',
  'MISSING_HELMET',
  'MISSING_VEST',
  'RESTRICTED_AREA',
  'LOITERING',
  'ABNORMAL_MOVEMENT',
  'POSSIBLE_FALL',
  'CROWD_ANOMALY',
]

const EVENT_LABELS: Record<string, string> = {
  PPE_VIOLATION: 'PPE Violation',
  MISSING_HELMET: 'Missing Helmet',
  MISSING_VEST: 'Missing Vest',
  RESTRICTED_AREA: 'Restricted Area',
  LOITERING: 'Loitering',
  ABNORMAL_MOVEMENT: 'Abnormal Movement',
  POSSIBLE_FALL: 'Possible Fall',
  CROWD_ANOMALY: 'Crowd Anomaly',
}

/** Tolerant of event types added on the server before this build knows them. */
export function eventLabel(type: string): string {
  return (
    EVENT_LABELS[type] ??
    type
      .toLowerCase()
      .split('_')
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ')
  )
}

/** Distinct hue per event type, for charts. Kept clear of the severity ramp. */
export const eventColour: Record<string, string> = {
  PPE_VIOLATION: '#ff6b6b',
  MISSING_HELMET: '#ff9f43',
  MISSING_VEST: '#ffd93d',
  RESTRICTED_AREA: '#ff4d6d',
  LOITERING: '#4dabf7',
  ABNORMAL_MOVEMENT: '#a78bfa',
  POSSIBLE_FALL: '#f472b6',
  CROWD_ANOMALY: '#38d9a9',
}

export function colourForType(type: string): string {
  return eventColour[type] ?? 'var(--text-3)'
}

/** Person ID rendered the way the system labels people: "Person #004". */
export function personLabel(id: number | null): string {
  return id === null ? '—' : `#${String(id).padStart(3, '0')}`
}

export function statusTone(status: string): string {
  switch (status) {
    case 'online':
      return 'var(--live)'
    case 'connecting':
      return 'var(--sev-medium)'
    case 'error':
      return 'var(--error)'
    case 'ended':
      return 'var(--text-3)'
    default:
      return 'var(--offline)'
  }
}
