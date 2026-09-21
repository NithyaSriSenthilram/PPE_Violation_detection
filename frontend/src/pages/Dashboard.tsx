/**
 * Overview — the screen a shift starts on.
 *
 * Ordered by what an operator needs first: is anything wrong right now (KPI
 * strip), what has been happening and where (the Watch Band), then live
 * cameras and the newest incidents side by side.
 */

import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import { compactNumber } from '../lib/format'
import type { DetectionFrame, SecurityEvent } from '../lib/types'
import { AlertRow } from '../components/AlertRow'
import { CameraTile } from '../components/CameraTile'
import { DEFAULT_LAYERS } from '../components/DetectionOverlay'
import { IncidentModal } from '../components/IncidentModal'
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  Readout,
  Segmented,
} from '../components/Primitives'
import { WatchBand } from '../components/WatchBand'

type Window = '6' | '12' | '24'

export function Dashboard({ frames }: { frames: Map<string, DetectionFrame> }) {
  const [hours, setHours] = useState<Window>('12')
  const [selected, setSelected] = useState<SecurityEvent | null>(null)

  const {
    data: stats,
    error: statsError,
    refresh: refreshStats,
  } = usePolling(() => api.statistics({ range: '24h' }), 15_000)
  const { data: cameras, error: camerasError } = usePolling(() => api.cameras(), 8_000)
  const { data: recent, refresh: refreshEvents } = usePolling(
    () => api.events({ limit: 120 }),
    12_000,
  )
  const { data: alerts, refresh: refreshAlerts } = usePolling(
    () => api.alerts({ limit: 8 }),
    12_000,
  )

  const kpis = stats?.kpis
  const liveCameras = useMemo(
    () => (cameras ?? []).filter((c) => c.enabled).slice(0, 4),
    [cameras],
  )

  return (
    <>
      <PageHeader
        eyebrow="Security Operations"
        title="Overview"
        subtitle="Live monitoring across every configured camera, with incidents as they are raised."
        actions={
          <Link className="btn" to="/live">
            Open camera wall
          </Link>
        }
      />

      {statsError && (
        <div style={{ marginBottom: 'var(--s4)' }}>
          <ErrorBanner message={statsError} onRetry={refreshStats} />
        </div>
      )}

      {/* ── KPI strip ────────────────────────────────────────────────────── */}
      <section
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: 'var(--s3)',
          marginBottom: 'var(--s5)',
        }}
      >
        <Kpi
          label="People detected"
          value={kpis ? compactNumber(kpis.people_detected) : '—'}
          hint="Distinct people tracked in the last 24 hours"
        />
        <Kpi
          label="Active cameras"
          value={kpis ? `${kpis.active_cameras}/${kpis.total_cameras}` : '—'}
          tone={
            kpis && kpis.total_cameras > 0 && kpis.active_cameras === 0
              ? 'var(--sev-critical)'
              : kpis && kpis.active_cameras < kpis.total_cameras
                ? 'var(--sev-medium)'
                : 'var(--live)'
          }
          hint="Cameras currently delivering frames"
        />
        <Kpi
          label="PPE violations"
          value={kpis ? compactNumber(kpis.ppe_violations) : '—'}
          tone={kpis?.ppe_violations ? 'var(--sev-high)' : undefined}
          hint="Missing helmet or vest, last 24 hours"
        />
        <Kpi
          label="Intrusions"
          value={kpis ? compactNumber(kpis.intrusions) : '—'}
          tone={kpis?.intrusions ? 'var(--sev-high)' : undefined}
          hint="Restricted-area entries, last 24 hours"
        />
        <Kpi
          label="Suspicious events"
          value={kpis ? compactNumber(kpis.suspicious_events) : '—'}
          hint="Loitering, abnormal movement, possible falls, crowd anomalies"
        />
        <Kpi
          label="Open alerts"
          value={kpis ? compactNumber(kpis.open_alerts) : '—'}
          tone={kpis?.open_alerts ? 'var(--sev-critical)' : 'var(--live)'}
          hint="Awaiting acknowledgement or resolution"
        />
      </section>

      {/* ── Watch Band ───────────────────────────────────────────────────── */}
      <section className="panel" style={{ marginBottom: 'var(--s5)' }}>
        <div className="panel-head">
          <div className="col" style={{ gap: 1 }}>
            <span className="eyebrow">Watch band</span>
            <h2 style={{ fontSize: 'var(--fs-md)' }}>Incident timeline by camera</h2>
          </div>
          <Segmented
            ariaLabel="Timeline window"
            value={hours}
            onChange={setHours}
            options={[
              { value: '6', label: '6h' },
              { value: '12', label: '12h' },
              { value: '24', label: '24h' },
            ]}
          />
        </div>
        <div className="panel-body">
          <WatchBand
            cameras={cameras ?? []}
            events={recent?.items ?? []}
            hours={Number(hours)}
            onSelectEvent={setSelected}
          />
        </div>
      </section>

      {/* ── Live cameras + recent incidents ──────────────────────────────── */}
      <div
        className="dash-split"
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1.35fr) minmax(0, 1fr)',
          gap: 'var(--s4)',
          alignItems: 'start',
        }}
      >
        <section className="panel">
          <div className="panel-head">
            <div className="col" style={{ gap: 1 }}>
              <span className="eyebrow">Live cameras</span>
              <h2 style={{ fontSize: 'var(--fs-md)' }}>
                {liveCameras.length
                  ? `${liveCameras.filter((c) => c.status === 'online').length} of ${liveCameras.length} streaming`
                  : 'No cameras configured'}
              </h2>
            </div>
            <Link className="btn btn-sm" to="/live">
              View all
            </Link>
          </div>
          <div className="panel-body">
            {camerasError && <ErrorBanner message={camerasError} />}
            {liveCameras.length === 0 ? (
              <EmptyState
                title="No cameras yet"
                hint="Add an RTSP stream, a webcam or an uploaded file to start monitoring."
                action={
                  <Link className="btn btn-primary" to="/cameras">
                    Add a camera
                  </Link>
                }
              />
            ) : (
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                  gap: 'var(--s3)',
                }}
              >
                {liveCameras.map((camera) => (
                  <CameraTile
                    key={camera.camera_id}
                    camera={camera}
                    frame={frames.get(camera.camera_id)}
                    layers={DEFAULT_LAYERS}
                    paused={false}
                    compact
                  />
                ))}
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <div className="col" style={{ gap: 1 }}>
              <span className="eyebrow">Active alerts</span>
              <h2 style={{ fontSize: 'var(--fs-md)' }}>
                {alerts?.total ?? 0} awaiting review
              </h2>
            </div>
            <Link className="btn btn-sm" to="/alerts">
              Alert centre
            </Link>
          </div>
          <div className="panel-body">
            {!alerts?.items.length ? (
              <EmptyState
                title="Nothing outstanding"
                hint="New incidents appear here the moment they are detected."
              />
            ) : (
              <div className="col alert-list" style={{ gap: 'var(--s2)' }}>
                {alerts.items.map((event) => (
                  <AlertRow key={event.event_id} event={event} onOpen={setSelected} />
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      {selected && (
        <IncidentModal
          event={selected}
          onClose={() => setSelected(null)}
          onUpdated={() => {
            refreshAlerts()
            refreshEvents()
            refreshStats()
          }}
        />
      )}
    </>
  )
}

function Kpi({
  label,
  value,
  tone,
  hint,
}: {
  label: string
  value: string
  tone?: string
  hint?: string
}) {
  return (
    <div
      className="panel"
      style={{ padding: 'var(--s3) var(--s4)' }}
      title={hint}
    >
      <Readout label={label} value={value} tone={tone} size="lg" />
    </div>
  )
}
