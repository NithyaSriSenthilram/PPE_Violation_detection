/**
 * Camera management.
 *
 * The add form adapts to source type because the three sources need genuinely
 * different inputs — an RTSP URL, a device index, or a filename inside the
 * upload directory — and showing one field labelled "URL" for all three is how
 * operators end up with cameras that never connect.
 */

import { useState } from 'react'
import { api, ApiError } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import { ago, shortStamp, statusTone } from '../lib/format'
import type { Camera, SourceType } from '../lib/types'
import { ZoneEditor } from '../components/ZoneEditor'
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  Readout,
  StatusDot,
} from '../components/Primitives'

const SOURCE_HELP: Record<SourceType, { label: string; placeholder: string; hint: string }> = {
  rtsp: {
    label: 'RTSP URL',
    placeholder: 'rtsp://user:pass@10.0.0.20:554/Streaming/Channels/101',
    hint: 'The camera’s RTSP endpoint. Credentials in the URL are stored as given.',
  },
  http: {
    label: 'HTTP stream URL',
    placeholder: 'http://10.0.0.30/video.mjpg',
    hint: 'An MJPEG or HLS endpoint the server can reach.',
  },
  webcam: {
    label: 'Device index',
    placeholder: '0',
    hint: 'The local capture device number — 0 is the first camera.',
  },
  file: {
    label: 'Filename',
    placeholder: 'site-recording.mp4',
    hint: 'A file inside the server’s upload directory. Useful for replaying footage as a camera.',
  },
}

export function Cameras() {
  const [showForm, setShowForm] = useState(false)
  const [editingZones, setEditingZones] = useState<Camera | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const { data: cameras, refresh } = usePolling(() => api.cameras(), 6_000)

  async function act(id: string, action: 'start' | 'stop' | 'delete' | 'toggle', camera?: Camera) {
    setBusy(id)
    setError(null)
    try {
      if (action === 'start') await api.startCamera(id)
      if (action === 'stop') await api.stopCamera(id)
      if (action === 'toggle' && camera)
        await api.updateCamera(id, { enabled: !camera.enabled })
      if (action === 'delete') {
        // Deleting a camera cascades to its zones and events, so it is worth
        // one confirmation.
        if (
          !window.confirm(
            `Delete "${camera?.name}"? Its zones and recorded incidents are deleted with it.`,
          )
        ) {
          setBusy(null)
          return
        }
        await api.deleteCamera(id)
      }
      await refresh()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'The action failed')
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="Configuration"
        title="Cameras & Zones"
        subtitle="Add video sources and draw the areas you want watched."
        actions={
          <button className="btn btn-primary" onClick={() => setShowForm((v) => !v)}>
            {showForm ? 'Cancel' : 'Add camera'}
          </button>
        }
      />

      {error && (
        <div style={{ marginBottom: 'var(--s3)' }}>
          <ErrorBanner message={error} />
        </div>
      )}

      {showForm && (
        <AddCameraForm
          onCreated={async () => {
            setShowForm(false)
            await refresh()
          }}
          onError={setError}
        />
      )}

      {!cameras?.length ? (
        <div className="panel">
          <EmptyState
            title="No cameras configured"
            hint="Add an RTSP stream, a local webcam, or point at a file in the upload directory to replay footage."
            action={
              <button className="btn btn-primary" onClick={() => setShowForm(true)}>
                Add your first camera
              </button>
            }
          />
        </div>
      ) : (
        <div className="col" style={{ gap: 'var(--s3)' }}>
          {cameras.map((camera) => (
            <section key={camera.camera_id} className="panel">
              <div
                className="row-between"
                style={{ padding: 'var(--s3) var(--s4)', gap: 'var(--s3)', flexWrap: 'wrap' }}
              >
                <div className="row grow" style={{ gap: 'var(--s3)', minWidth: 0 }}>
                  <StatusDot status={camera.status} />
                  <div className="col" style={{ gap: 1, minWidth: 0 }}>
                    <div className="row" style={{ gap: 'var(--s2)' }}>
                      <span className="truncate" style={{ fontSize: 'var(--fs-md)', fontWeight: 600 }}>
                        {camera.name}
                      </span>
                      <span className="chip">{camera.source_type}</span>
                      {!camera.enabled && <span className="chip">disabled</span>}
                      {!camera.analytics_enabled && (
                        <span className="chip" title="Streaming only — no detection rules run">
                          analytics off
                        </span>
                      )}
                    </div>
                    <span className="truncate dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                      {camera.location || 'No location set'} ·{' '}
                      <span className="mono">{camera.stream_url}</span>
                    </span>
                  </div>
                </div>

                <div className="row" style={{ gap: 'var(--s5)' }}>
                  <Readout
                    label="Status"
                    value={camera.status}
                    size="sm"
                    tone={statusTone(camera.status)}
                  />
                  <Readout label="FPS" value={camera.fps.toFixed(1)} size="sm" />
                  <Readout label="People" value={String(camera.people_count)} size="sm" />
                  <Readout label="Zones" value={String(camera.zone_count)} size="sm" />
                  <Readout
                    label="Last frame"
                    value={camera.last_frame_at ? ago(camera.last_frame_at) : '—'}
                    size="sm"
                    title={
                      camera.last_frame_at ? shortStamp(camera.last_frame_at) : 'No frames yet'
                    }
                  />
                </div>

                <div className="row" style={{ gap: 'var(--s2)' }}>
                  <button
                    className="btn btn-sm"
                    onClick={() => setEditingZones(camera)}
                    disabled={busy === camera.camera_id}
                  >
                    Zones
                  </button>
                  {camera.status === 'online' ? (
                    <button
                      className="btn btn-sm"
                      onClick={() => act(camera.camera_id, 'stop', camera)}
                      disabled={busy === camera.camera_id}
                    >
                      Stop
                    </button>
                  ) : (
                    <button
                      className="btn btn-sm"
                      onClick={() => act(camera.camera_id, 'start', camera)}
                      disabled={busy === camera.camera_id || !camera.enabled}
                      title={camera.enabled ? undefined : 'Enable the camera first'}
                    >
                      Start
                    </button>
                  )}
                  <button
                    className="btn btn-sm"
                    onClick={() => act(camera.camera_id, 'toggle', camera)}
                    disabled={busy === camera.camera_id}
                  >
                    {camera.enabled ? 'Disable' : 'Enable'}
                  </button>
                  <button
                    className="btn btn-sm btn-danger"
                    onClick={() => act(camera.camera_id, 'delete', camera)}
                    disabled={busy === camera.camera_id}
                  >
                    Delete
                  </button>
                </div>
              </div>

              {camera.last_error && (
                <div
                  style={{
                    padding: 'var(--s2) var(--s4)',
                    borderTop: '1px solid var(--line-soft)',
                    background: 'var(--sev-critical-wash)',
                    fontSize: 'var(--fs-tiny)',
                    color: 'var(--text-2)',
                  }}
                >
                  <span className="mono" style={{ color: 'var(--error)' }}>
                    FAULT
                  </span>{' '}
                  {camera.last_error}
                </div>
              )}
            </section>
          ))}
        </div>
      )}

      {editingZones && (
        <ZoneEditor
          camera={editingZones}
          onClose={() => {
            setEditingZones(null)
            refresh()
          }}
        />
      )}
    </>
  )
}

function AddCameraForm({
  onCreated,
  onError,
}: {
  onCreated: () => void
  onError: (message: string) => void
}) {
  const [name, setName] = useState('')
  const [location, setLocation] = useState('')
  const [sourceType, setSourceType] = useState<SourceType>('rtsp')
  const [streamUrl, setStreamUrl] = useState('')
  const [profile, setProfile] = useState('thorough')
  const [analytics, setAnalytics] = useState(true)
  const [busy, setBusy] = useState(false)

  const help = SOURCE_HELP[sourceType]

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setBusy(true)
    try {
      await api.createCamera({
        name: name.trim(),
        location: location.trim(),
        source_type: sourceType,
        stream_url: streamUrl.trim(),
        analytics_enabled: analytics,
        settings_json: { profile },
      })
      onCreated()
    } catch (cause) {
      onError(cause instanceof ApiError ? cause.message : 'Could not add the camera')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="panel" onSubmit={submit} style={{ marginBottom: 'var(--s4)' }}>
      <div className="panel-head">
        <span className="eyebrow">New camera</span>
      </div>
      <div
        className="panel-body"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 'var(--s3)',
          alignItems: 'flex-start',
        }}
      >
        <div className="field">
          <label className="eyebrow" htmlFor="c-name">Name</label>
          <input
            id="c-name"
            className="input"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Loading Bay 01"
          />
        </div>

        <div className="field">
          <label className="eyebrow" htmlFor="c-location">Location</label>
          <input
            id="c-location"
            className="input"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="North Yard — Gate A"
          />
        </div>

        <div className="field">
          <label className="eyebrow" htmlFor="c-source">Source type</label>
          <select
            id="c-source"
            className="select"
            value={sourceType}
            onChange={(e) => {
              setSourceType(e.target.value as SourceType)
              setStreamUrl('')
            }}
          >
            <option value="rtsp">RTSP camera</option>
            <option value="http">HTTP / MJPEG stream</option>
            <option value="webcam">Local webcam</option>
            <option value="file">File replay</option>
          </select>
        </div>

        <div className="field" style={{ gridColumn: 'span 2', minWidth: 0 }}>
          <label className="eyebrow" htmlFor="c-url">{help.label}</label>
          <input
            id="c-url"
            className="input mono"
            required
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            placeholder={help.placeholder}
          />
          <span className="dim" style={{ fontSize: 'var(--fs-tiny)' }}>
            {help.hint}
          </span>
        </div>

        <div className="field">
          <label className="eyebrow" htmlFor="c-profile">Detection profile</label>
          <select
            id="c-profile"
            className="select"
            value={profile}
            onChange={(e) => setProfile(e.target.value)}
          >
            <option value="thorough">Thorough — everything</option>
            <option value="standard">Standard — no fall detection</option>
            <option value="security_only">Security only — no PPE</option>
            <option value="ppe_only">PPE only</option>
            <option value="fast">Fast — zones and crowd</option>
          </select>
        </div>

        <div className="col" style={{ gap: 'var(--s3)', justifyContent: 'flex-end' }}>
          <label className="row" style={{ gap: 6, fontSize: 'var(--fs-sm)', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={analytics}
              onChange={(e) => setAnalytics(e.target.checked)}
              style={{ accentColor: 'var(--hivis)' }}
            />
            <span className="dim">Run detection rules</span>
          </label>
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? 'Adding…' : 'Add camera'}
          </button>
        </div>
      </div>
    </form>
  )
}
