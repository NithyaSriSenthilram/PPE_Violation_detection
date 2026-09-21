/**
 * Diagnostics — the page that refuses to flatter the deployment.
 *
 * It reports which inference backend is *actually* running and the verbatim
 * reason each other one is not; whether the Mojo kernels are genuinely in use,
 * with the benchmark that decided each one; and whether PPE assessment comes
 * from a trained model or the colour heuristic. If something is a fallback,
 * this page says so in those words.
 */

import { api } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import { duration } from '../lib/format'
import { ErrorBanner, PageHeader, Readout } from '../components/Primitives'

export function Diagnostics() {
  const { data, error, loading, refresh } = usePolling(() => api.diagnostics(), 10_000)

  if (loading && !data) {
    return (
      <>
        <PageHeader eyebrow="System" title="Diagnostics" />
        <div className="skeleton" style={{ height: 400 }} />
      </>
    )
  }

  if (error || !data) {
    return (
      <>
        <PageHeader eyebrow="System" title="Diagnostics" />
        <ErrorBanner message={error ?? 'No diagnostics available'} onRetry={refresh} />
      </>
    )
  }

  const inference = data.inference as Record<string, unknown>
  const mojo = data.mojo
  const ppe = data.ppe as Record<string, unknown>
  const pipeline = data.pipeline as Record<string, unknown>
  const synthetic = inference.is_synthetic === true
  const heuristicPpe = ppe.method === 'heuristic'

  return (
    <>
      <PageHeader
        eyebrow="System"
        title="Diagnostics"
        subtitle="What is actually running on this machine, and what is falling back."
      />

      {synthetic && (
        <div style={{ marginBottom: 'var(--s4)' }}>
          <ErrorBanner
            message="No real inference backend could load. Detections are synthetic and must not be treated as real security findings."
          />
        </div>
      )}

      <div className="col" style={{ gap: 'var(--s4)' }}>
        {/* ── Runtime ──────────────────────────────────────────────────────── */}
        <Panel eyebrow="Runtime" title="Host and process">
          <Grid>
            <Readout label="Version" value={data.version} size="sm" />
            <Readout label="Environment" value={data.environment} size="sm" />
            <Readout label="Uptime" value={duration(data.uptime_seconds)} size="sm" />
            <Readout label="Python" value={data.python_version} size="sm" />
            <Readout label="Platform" value={String(data.platform.system)} size="sm" />
            <Readout label="Architecture" value={String(data.platform.machine)} size="sm" />
            <Readout label="CPU cores" value={String(data.platform.cpu_count)} size="sm" />
            <Readout
              label="Database"
              value={data.database.connected ? String(data.database.url_scheme) : 'unavailable'}
              size="sm"
              tone={data.database.connected ? 'var(--live)' : 'var(--error)'}
            />
          </Grid>
        </Panel>

        {/* ── Inference backends ───────────────────────────────────────────── */}
        <Panel
          eyebrow="AI inference"
          title={`Active backend: ${String(inference.active_backend)}`}
          note={`Requested: ${String(inference.requested_backend)}`}
        >
          <Grid>
            <Readout
              label="Active"
              value={String(inference.active_backend)}
              size="sm"
              tone={synthetic ? 'var(--sev-critical)' : 'var(--hivis)'}
            />
            <Readout label="Model" value={String(inference.model_path).split('/').pop() ?? '—'} size="sm" />
            <Readout
              label="Model present"
              value={inference.model_present ? 'yes' : 'no'}
              size="sm"
              tone={inference.model_present ? undefined : 'var(--error)'}
            />
            <Readout
              label="Input size"
              value={(inference.input_size as number[])?.join('×') ?? '—'}
              size="sm"
            />
            <Readout
              label="Confidence"
              value={String(inference.confidence_threshold)}
              size="sm"
            />
            <Readout label="NMS IoU" value={String(inference.nms_iou_threshold)} size="sm" />
          </Grid>

          <div className="col" style={{ gap: 'var(--s2)', marginTop: 'var(--s4)' }}>
            <span className="eyebrow">Backend availability</span>
            {data.backends.map((backend) => (
              <div
                key={backend.name}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '110px 84px 1fr',
                  gap: 'var(--s3)',
                  alignItems: 'baseline',
                  padding: 'var(--s2) var(--s3)',
                  borderRadius: 'var(--r-sm)',
                  border: `1px solid ${backend.active ? 'var(--hivis-line)' : 'var(--line-soft)'}`,
                  background: backend.active ? 'var(--hivis-wash)' : 'var(--surface-2)',
                }}
              >
                <span className="mono" style={{ fontSize: 'var(--fs-sm)', fontWeight: 700 }}>
                  {backend.name}
                </span>
                <span
                  className="mono"
                  style={{
                    fontSize: 'var(--fs-micro)',
                    letterSpacing: '0.1em',
                    color: backend.active
                      ? 'var(--hivis)'
                      : backend.available
                        ? 'var(--text-2)'
                        : 'var(--text-4)',
                  }}
                >
                  {backend.active ? 'ACTIVE' : backend.available ? 'AVAILABLE' : 'UNAVAILABLE'}
                </span>
                <span
                  style={{
                    fontSize: 'var(--fs-tiny)',
                    color: 'var(--text-3)',
                    lineHeight: 1.5,
                    wordBreak: 'break-word',
                  }}
                >
                  {backend.reason || detailSummary(backend.detail)}
                </span>
              </div>
            ))}
          </div>
        </Panel>

        {/* ── Mojo ─────────────────────────────────────────────────────────── */}
        <Panel
          eyebrow="Mojo acceleration"
          title={
            mojo.active
              ? 'Active for the kernels that measured faster'
              : mojo.available
                ? 'Library loaded, but numpy won every benchmark'
                : 'Not in use — numpy fallbacks'
          }
          note={mojo.reason || undefined}
        >
          <Grid>
            <Readout label="Mode" value={mojo.mode} size="sm" />
            <Readout
              label="Library"
              value={mojo.available ? 'loaded' : 'not loaded'}
              size="sm"
              tone={mojo.available ? 'var(--live)' : 'var(--text-3)'}
            />
            <Readout label="ABI" value={mojo.abi_version ? `v${mojo.abi_version}` : '—'} size="sm" />
            <Readout
              label="Kernels active"
              value={String(Object.values(mojo.kernels).filter((k) => k.active).length)}
              size="sm"
              tone={mojo.active ? 'var(--hivis)' : undefined}
            />
          </Grid>

          <p
            className="dim"
            style={{ fontSize: 'var(--fs-tiny)', marginTop: 'var(--s3)', lineHeight: 1.6 }}
          >
            Each kernel is checked against its numpy twin for numerical agreement and then
            benchmarked. A Mojo kernel is only used where it actually measured faster on this
            machine — numpy delegates to SIMD C for some of these shapes and genuinely wins.
          </p>

          <div style={{ overflowX: 'auto', marginTop: 'var(--s3)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--fs-tiny)' }}>
              <thead>
                <tr>
                  {['Kernel', 'Using', 'Mojo', 'numpy', 'Speedup', 'Verified'].map((head) => (
                    <th
                      key={head}
                      className="eyebrow"
                      style={{
                        textAlign: 'left',
                        padding: '5px 10px 5px 0',
                        borderBottom: '1px solid var(--line)',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {head}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(mojo.kernels).map(([name, kernel]) => (
                  <tr key={name}>
                    <td className="mono" style={cell}>{name}</td>
                    <td style={cell}>
                      <span
                        className="chip"
                        style={{
                          color: kernel.active ? 'var(--hivis)' : 'var(--text-3)',
                          borderColor: kernel.active ? 'var(--hivis-line)' : 'var(--line-strong)',
                        }}
                      >
                        {kernel.implementation}
                      </span>
                    </td>
                    <td className="mono" style={cell}>
                      {kernel.mojo_ms !== undefined ? `${kernel.mojo_ms.toFixed(3)} ms` : '—'}
                    </td>
                    <td className="mono" style={cell}>
                      {kernel.numpy_ms !== undefined ? `${kernel.numpy_ms.toFixed(3)} ms` : '—'}
                    </td>
                    <td
                      className="mono"
                      style={{
                        ...cell,
                        color:
                          kernel.speedup && kernel.speedup > 1
                            ? 'var(--hivis)'
                            : 'var(--text-3)',
                        fontWeight: 600,
                      }}
                    >
                      {kernel.speedup !== undefined ? `${kernel.speedup.toFixed(2)}×` : '—'}
                    </td>
                    <td className="mono" style={cell}>
                      {kernel.correct === 1 ? 'yes' : kernel.correct === 0 ? 'NO' : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        {/* ── PPE ──────────────────────────────────────────────────────────── */}
        <Panel
          eyebrow="PPE assessment"
          title={
            ppe.method === 'model'
              ? `Trained PPE model — ${String(ppe.model ?? '')}`
              : heuristicPpe
                ? 'Colour heuristic — advisory only'
                : 'Disabled'
          }
        >
          <Grid>
            <Readout
              label="Method"
              value={String(ppe.method_label ?? ppe.method)}
              size="sm"
              tone={heuristicPpe ? 'var(--sev-medium)' : 'var(--text)'}
            />
            <Readout
              label="Model loaded"
              value={ppe.model_loaded ? 'yes' : 'no'}
              size="sm"
              tone={ppe.model_loaded ? 'var(--text)' : 'var(--sev-medium)'}
            />
            <Readout
              label="Inference backend"
              value={String(ppe.backend ?? 'none')}
              size="sm"
            />
            <Readout
              label="Model format"
              value={String(ppe.model_format ?? '—')}
              size="sm"
            />
            <Readout
              label="Confidence threshold"
              value={String(ppe.confidence_threshold ?? '—')}
              size="sm"
            />
            <Readout
              label="Association threshold"
              value={String(ppe.association_threshold ?? '—')}
              size="sm"
            />
            <Readout
              label="Required PPE"
              value={(ppe.required_ppe as string[] | undefined)?.join(', ') || '—'}
              size="sm"
            />
            <Readout
              label="Heuristic permitted"
              value={ppe.heuristic_allowed ? 'yes' : 'no'}
              size="sm"
            />
          </Grid>

          <Readout
            label="Model path"
            value={String(ppe.model_path ?? '—')}
            size="sm"
            tone={ppe.model_present ? undefined : 'var(--sev-medium)'}
          />

          {/* The class map is what makes a third-party model interpretable, so
              it is shown in full rather than summarised — an unmapped class is
              a silent gap in coverage. */}
          {!!(ppe.mapping && Object.keys(ppe.mapping as object).length) && (
            <div style={{ marginTop: 'var(--s3)' }}>
              <span className="eyebrow">Class mapping</span>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 260px))',
                  gap: 'var(--s1) var(--s5)',
                  marginTop: 'var(--s2)',
                  fontSize: 'var(--fs-tiny)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {Object.entries(ppe.mapping as Record<string, string>).map(([name, role]) => (
                  // Left-aligned pairs, not space-between: across a wide panel
                  // the two halves drift so far apart that the reader loses
                  // which role belongs to which class.
                  <div key={name} className="row" style={{ gap: 'var(--s2)' }}>
                    <span style={{ color: 'var(--text-2)' }}>{name}</span>
                    <span style={{ color: 'var(--text-4)' }}>-&gt;</span>
                    <span style={{ color: 'var(--text-3)' }}>{role}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {!!(ppe.unmapped_classes as string[] | undefined)?.length && (
            <p className="dim" style={{ fontSize: 'var(--fs-tiny)', marginTop: 'var(--s2)' }}>
              Unmapped classes (ignored):{' '}
              <span className="mono">{(ppe.unmapped_classes as string[]).join(', ')}</span>
            </p>
          )}

          {!!String(ppe.warning ?? '') && (
            <p
              style={{
                fontSize: 'var(--fs-tiny)',
                marginTop: 'var(--s3)',
                lineHeight: 1.6,
                padding: 'var(--s3)',
                borderRadius: 'var(--r-sm)',
                border: '1px solid var(--sev-medium)',
                background: 'var(--sev-medium-wash)',
                color: 'var(--text-2)',
              }}
            >
              {String(ppe.warning)}
            </p>
          )}

          {(ppe.fallback_chain as { backend: string; reason: string }[] | undefined)?.map(
            (attempt) => (
              <p
                key={attempt.backend}
                className="dim"
                style={{ fontSize: 'var(--fs-tiny)', marginTop: 'var(--s1)' }}
              >
                tried <span className="mono">{attempt.backend}</span>: {attempt.reason}
              </p>
            ),
          )}

          {!!String(ppe.description ?? '') && (
            <p className="dim" style={{ fontSize: 'var(--fs-tiny)', marginTop: 'var(--s2)' }}>
              {String(ppe.description)}
            </p>
          )}
        </Panel>

        {/* ── Pipeline ─────────────────────────────────────────────────────── */}
        <Panel eyebrow="Pipeline" title="Capture and event processing">
          <Grid>
            <Readout label="Cameras registered" value={String(pipeline.cameras_registered ?? 0)} size="sm" />
            <Readout
              label="Cameras online"
              value={String(pipeline.cameras_online ?? 0)}
              size="sm"
              tone={Number(pipeline.cameras_online) > 0 ? 'var(--live)' : undefined}
            />
            <Readout label="Target FPS" value={String(pipeline.target_fps)} size="sm" />
            <Readout
              label="Detect every"
              value={`${pipeline.detect_every_n_frames} frames`}
              size="sm"
            />
            <Readout label="Clip buffer" value={`${pipeline.clip_buffer_seconds}s`} size="sm" />
            <Readout
              label="Buffer memory"
              value={`${pipeline.buffer_memory_mb ?? 0} MB`}
              size="sm"
              title="Total JPEG-encoded rolling buffer across all cameras"
            />
          </Grid>

          {isRecord(pipeline.event_engine) && (
            <>
              <span className="eyebrow" style={{ display: 'block', marginTop: 'var(--s4)' }}>
                Event engine
              </span>
              <Grid>
                <Readout label="Accepted" value={String(pipeline.event_engine.accepted ?? 0)} size="sm" />
                <Readout
                  label="Suppressed"
                  value={String(
                    isRecord(pipeline.event_engine.cooldown)
                      ? (pipeline.event_engine.cooldown.total_suppressed ?? 0)
                      : 0,
                  )}
                  size="sm"
                  title="Duplicate firings collapsed by the cooldown"
                />
                <Readout
                  label="Below confidence"
                  value={String(pipeline.event_engine.rejected_low_confidence ?? 0)}
                  size="sm"
                />
                <Readout
                  label="Cooldown"
                  value={`${pipeline.event_engine.cooldown_seconds}s`}
                  size="sm"
                />
              </Grid>
            </>
          )}
        </Panel>

        {/* ── Config ───────────────────────────────────────────────────────── */}
        <Panel eyebrow="Configuration" title="Effective thresholds">
          <Grid>
            {Object.entries(data.config).map(([key, value]) => (
              <Readout
                key={key}
                label={key.replace(/_/g, ' ')}
                value={Array.isArray(value) ? value.join(', ') || '—' : String(value)}
                size="sm"
              />
            ))}
          </Grid>
        </Panel>
      </div>
    </>
  )
}

const cell: React.CSSProperties = {
  padding: '5px 10px 5px 0',
  borderBottom: '1px solid var(--line-soft)',
  whiteSpace: 'nowrap',
}

function Panel({
  eyebrow,
  title,
  note,
  children,
}: {
  eyebrow: string
  title: string
  note?: string
  children: React.ReactNode
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div className="col" style={{ gap: 1, minWidth: 0 }}>
          <span className="eyebrow">{eyebrow}</span>
          <h2 className="truncate" style={{ fontSize: 'var(--fs-md)' }}>
            {title}
          </h2>
        </div>
        {note && (
          <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
            {note}
          </span>
        )}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  )
}

function Grid({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: 'var(--s3)',
      }}
    >
      {children}
    </div>
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Condense a backend's detail blob into one readable line. */
function detailSummary(detail: Record<string, unknown>): string {
  const parts: string[] = []
  for (const [key, value] of Object.entries(detail)) {
    if (value === undefined || value === null) continue
    if (Array.isArray(value)) {
      if (value.length) parts.push(`${key}: ${value.slice(0, 3).join(', ')}`)
    } else if (typeof value !== 'object') {
      parts.push(`${key}: ${value}`)
    }
  }
  return parts.slice(0, 4).join(' · ') || 'ready'
}
