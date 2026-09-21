/**
 * The console frame: a persistent navigation rail and a status bar that is
 * always truthful about system state.
 *
 * The status bar reports the *actual* inference backend and warns plainly when
 * detections are synthetic, because an operator must never mistake a degraded
 * system for a working one.
 */

import { NavLink, useLocation } from 'react-router-dom'
import { useEffect, useState, type ReactNode } from 'react'
import { api } from '../lib/api'
import { timecode } from '../lib/format'
import { useClock, usePolling } from '../hooks/usePolling'
import { StatusDot } from './Primitives'
import type { ConnectionState } from '../hooks/useWebSocket'

interface NavItem {
  to: string
  label: string
  icon: ReactNode
  badgeKey?: 'alerts'
}

/* Line icons at a single 1.5px weight — drawn inline so there is no icon
   dependency and every glyph matches the hairline language of the panels. */
const stroke = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.6,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

const Icon = ({ children }: { children: ReactNode }) => (
  <svg width="16" height="16" viewBox="0 0 20 20" {...stroke} aria-hidden>
    {children}
  </svg>
)

const NAV: NavItem[] = [
  {
    to: '/',
    label: 'Overview',
    icon: (
      <Icon>
        <rect x="2.5" y="2.5" width="6" height="6" rx="1" />
        <rect x="11.5" y="2.5" width="6" height="6" rx="1" />
        <rect x="2.5" y="11.5" width="6" height="6" rx="1" />
        <rect x="11.5" y="11.5" width="6" height="6" rx="1" />
      </Icon>
    ),
  },
  {
    to: '/live',
    label: 'Live Cameras',
    icon: (
      <Icon>
        <path d="M2.5 5.5h11v9h-11z" />
        <path d="M13.5 9l4-2.5v7L13.5 11z" />
      </Icon>
    ),
  },
  {
    to: '/alerts',
    label: 'Alert Centre',
    icon: (
      <Icon>
        <path d="M10 2.5l7.5 13h-15z" />
        <path d="M10 7.5v4M10 13.8v.2" />
      </Icon>
    ),
    badgeKey: 'alerts',
  },
  {
    to: '/analytics',
    label: 'Analytics',
    icon: (
      <Icon>
        <path d="M2.5 17.5h15" />
        <path d="M5 17.5v-6M9 17.5v-11M13 17.5v-8M17 17.5v-4" />
      </Icon>
    ),
  },
  {
    to: '/analysis',
    label: 'Video Analysis',
    icon: (
      <Icon>
        <rect x="2.5" y="3.5" width="15" height="13" rx="1.5" />
        <path d="M8 7.5l5 2.5-5 2.5z" />
      </Icon>
    ),
  },
  {
    to: '/cameras',
    label: 'Cameras & Zones',
    icon: (
      <Icon>
        <circle cx="10" cy="10" r="3" />
        <path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M15.3 4.7l-1.4 1.4M6.1 13.9l-1.4 1.4" />
      </Icon>
    ),
  },
  {
    to: '/diagnostics',
    label: 'Diagnostics',
    icon: (
      <Icon>
        <path d="M2.5 10h3l2-4 2.5 8 2.5-6 1.5 2h3.5" />
      </Icon>
    ),
  },
]

export function AppShell({
  children,
  connection,
  openAlertCount,
}: {
  children: ReactNode
  connection: ConnectionState
  openAlertCount: number
}) {
  const [collapsed, setCollapsed] = useState(
    () => window.localStorage.getItem('sv.rail') === 'collapsed',
  )
  const location = useLocation()
  const now = useClock()
  const { data: health } = usePolling(() => api.health(), 10_000)

  useEffect(() => {
    // Remembering the rail state is a per-viewer convenience; failing to read
    // or write it must never break the console.
    try {
      window.localStorage.setItem('sv.rail', collapsed ? 'collapsed' : 'expanded')
    } catch {
      /* private mode / blocked storage */
    }
  }, [collapsed])

  const synthetic = health?.inference_backend === 'mock'
  const degraded = health?.status === 'degraded'

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `${collapsed ? 'var(--rail-w-collapsed)' : 'var(--rail-w)'} 1fr`,
        gridTemplateRows: 'var(--topbar-h) 1fr',
        height: '100vh',
        transition: 'grid-template-columns var(--t-base) var(--ease)',
      }}
    >
      {/* ── Wordmark ─────────────────────────────────────────────────────── */}
      <div
        style={{
          gridArea: '1 / 1',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s2)',
          padding: `0 ${collapsed ? '0' : 'var(--s4)'}`,
          justifyContent: collapsed ? 'center' : 'flex-start',
          borderBottom: '1px solid var(--line)',
          borderRight: '1px solid var(--line)',
          background: 'var(--bg-elev)',
          minWidth: 0,
        }}
      >
        <Wordmark />
        {!collapsed && (
          <div className="col" style={{ gap: 0, minWidth: 0 }}>
            <span
              style={{
                fontSize: 'var(--fs-sm)',
                fontWeight: 800,
                letterSpacing: '-0.01em',
                lineHeight: 1.1,
              }}
            >
              SENTINELVISION
            </span>
            <span
              className="mono"
              style={{
                fontSize: 'var(--fs-micro)',
                letterSpacing: '0.24em',
                color: 'var(--hivis-dim)',
                lineHeight: 1.2,
              }}
            >
              AI
            </span>
          </div>
        )}
      </div>

      {/* ── Status bar ───────────────────────────────────────────────────── */}
      <header
        style={{
          gridArea: '1 / 2',
          display: 'flex',
          alignItems: 'center',
          gap: 'var(--s4)',
          padding: '0 var(--s5)',
          borderBottom: '1px solid var(--line)',
          background: 'var(--bg-elev)',
          minWidth: 0,
        }}
      >
        <StatusDot
          status={degraded ? 'error' : 'online'}
          label={degraded ? 'System Degraded' : 'System Online'}
        />

        {synthetic && (
          <span
            className="chip"
            title="No inference backend could load. Detections are synthetic and must not be treated as real."
            style={{
              color: 'var(--sev-critical)',
              borderColor: 'var(--sev-critical)',
              background: 'var(--sev-critical-wash)',
            }}
          >
            Synthetic detections
          </span>
        )}

        <div className="grow" />

        <div className="row" style={{ gap: 'var(--s4)' }}>
          <Meter label="Cameras">
            <span style={{ color: health?.cameras_online ? 'var(--live)' : 'var(--text-3)' }}>
              {health?.cameras_online ?? 0}
            </span>
            <span className="dim-2">/{health?.cameras_total ?? 0}</span>
          </Meter>

          <Meter label="Backend">
            <span style={{ color: synthetic ? 'var(--sev-critical)' : 'var(--hivis)' }}>
              {health?.inference_backend ?? '—'}
            </span>
          </Meter>

          <Meter label="Feed">
            <span
              style={{
                color:
                  connection === 'open'
                    ? 'var(--live)'
                    : connection === 'connecting'
                      ? 'var(--sev-medium)'
                      : 'var(--error)',
              }}
            >
              {connection === 'open' ? 'live' : connection}
            </span>
          </Meter>

          <div
            style={{
              width: 1,
              height: 22,
              background: 'var(--line)',
            }}
          />

          {/* The clock is the console's anchor: mono, tabular, always ticking. */}
          <div className="col" style={{ gap: 0, alignItems: 'flex-end' }}>
            <span
              className="mono"
              style={{ fontSize: 'var(--fs-md)', fontWeight: 600, lineHeight: 1.1 }}
            >
              {timecode(now)}
            </span>
            <span className="eyebrow" style={{ letterSpacing: '0.1em' }}>
              {now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })}
            </span>
          </div>

          <Operator />
        </div>
      </header>

      {/* ── Rail ─────────────────────────────────────────────────────────── */}
      <nav
        aria-label="Main"
        style={{
          gridArea: '2 / 1',
          display: 'flex',
          flexDirection: 'column',
          borderRight: '1px solid var(--line)',
          background: 'var(--bg-elev)',
          padding: 'var(--s3) var(--s2)',
          gap: 2,
          overflow: 'hidden',
        }}
      >
        {NAV.map((item) => {
          const active =
            item.to === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.to)
          const badge = item.badgeKey === 'alerts' ? openAlertCount : 0
          return (
            <NavLink
              key={item.to}
              to={item.to}
              title={collapsed ? item.label : undefined}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--s3)',
                padding: collapsed ? '0.55rem 0' : '0.5rem 0.65rem',
                justifyContent: collapsed ? 'center' : 'flex-start',
                borderRadius: 'var(--r-sm)',
                fontSize: 'var(--fs-sm)',
                fontWeight: active ? 600 : 500,
                color: active ? 'var(--text)' : 'var(--text-3)',
                background: active ? 'var(--surface-2)' : 'transparent',
                position: 'relative',
                transition: 'color var(--t-fast) var(--ease), background var(--t-fast) var(--ease)',
              }}
            >
              {/* Active marker: a hi-vis tick on the rail edge, not a fill. */}
              {active && (
                <span
                  aria-hidden
                  style={{
                    position: 'absolute',
                    left: -8,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    width: 2,
                    height: 16,
                    borderRadius: 99,
                    background: 'var(--hivis)',
                  }}
                />
              )}
              <span style={{ display: 'grid', placeItems: 'center', flexShrink: 0 }}>
                {item.icon}
              </span>
              {!collapsed && <span className="truncate grow">{item.label}</span>}
              {badge > 0 && (
                <span
                  className="mono"
                  aria-label={`${badge} open alerts`}
                  style={{
                    fontSize: 'var(--fs-micro)',
                    fontWeight: 700,
                    padding: collapsed ? '0 3px' : '0.05rem 0.3rem',
                    borderRadius: 99,
                    background: 'var(--sev-critical)',
                    color: '#fff',
                    position: collapsed ? 'absolute' : 'static',
                    top: collapsed ? 4 : undefined,
                    right: collapsed ? 6 : undefined,
                  }}
                >
                  {badge > 99 ? '99+' : badge}
                </span>
              )}
            </NavLink>
          )
        })}

        <div className="grow" />

        <button
          className="btn btn-ghost btn-sm"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
          style={{ justifyContent: collapsed ? 'center' : 'flex-start' }}
        >
          <svg width="14" height="14" viewBox="0 0 20 20" {...stroke} aria-hidden>
            <path d={collapsed ? 'M7 5l5 5-5 5' : 'M13 5l-5 5 5 5'} />
          </svg>
          {!collapsed && <span>Collapse</span>}
        </button>
      </nav>

      {/* ── Content ──────────────────────────────────────────────────────── */}
      <main
        key={location.pathname}
        className="scroll-y fade-in"
        style={{ gridArea: '2 / 2', padding: 'var(--s5) var(--s6)', minWidth: 0 }}
      >
        {children}
      </main>
    </div>
  )
}

/**
 * The mark: a viewfinder frame with an iris. It is the same corner-tick
 * language used on the detection overlays and evidence snapshots, so the brand
 * and the product's own output look like one system.
 */
function Wordmark() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden style={{ flexShrink: 0 }}>
      <g fill="none" stroke="var(--hivis)" strokeWidth="1.8" strokeLinecap="round">
        <path d="M2 8V3.5A1.5 1.5 0 013.5 2H8" />
        <path d="M16 2h4.5A1.5 1.5 0 0122 3.5V8" />
        <path d="M22 16v4.5a1.5 1.5 0 01-1.5 1.5H16" />
        <path d="M8 22H3.5A1.5 1.5 0 012 20.5V16" />
      </g>
      <circle cx="12" cy="12" r="3.6" fill="none" stroke="var(--hivis)" strokeWidth="1.8" />
      <circle cx="12" cy="12" r="1.2" fill="var(--hivis)" />
    </svg>
  )
}

function Meter({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="col" style={{ gap: 0, alignItems: 'flex-end' }}>
      <span className="eyebrow">{label}</span>
      <span
        className="mono"
        style={{ fontSize: 'var(--fs-sm)', fontWeight: 600, lineHeight: 1.3 }}
      >
        {children}
      </span>
    </div>
  )
}

/**
 * Operator identity. This build has no authentication beyond an optional
 * shared API key, so it shows the shift role rather than inventing a user —
 * a fake signed-in name would misrepresent what the system knows.
 */
function Operator() {
  return (
    <div className="row" style={{ gap: 'var(--s2)' }} title="Local session — no user accounts configured">
      <div
        style={{
          width: 26,
          height: 26,
          borderRadius: 'var(--r-sm)',
          border: '1px solid var(--line-strong)',
          background: 'var(--surface-2)',
          display: 'grid',
          placeItems: 'center',
          fontSize: 'var(--fs-micro)',
          fontFamily: 'var(--font-mono)',
          fontWeight: 700,
          color: 'var(--hivis)',
        }}
      >
        OP
      </div>
      <div className="col" style={{ gap: 0 }}>
        <span style={{ fontSize: 'var(--fs-tiny)', fontWeight: 600, lineHeight: 1.2 }}>
          Operator
        </span>
        <span className="eyebrow" style={{ letterSpacing: '0.08em' }}>
          Local session
        </span>
      </div>
    </div>
  )
}
