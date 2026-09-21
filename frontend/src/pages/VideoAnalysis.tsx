/**
 * Video analysis — upload a recording and get incidents out of it.
 *
 * The flow follows the operator's mental model: choose a file, choose how hard
 * to look, watch it work, read the results. Analysis runs server-side on a
 * worker thread, so the page stays responsive and closing the tab does not
 * cancel the job.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../lib/api'
import { usePolling } from '../hooks/usePolling'
import {
  bytes,
  duration,
  eventLabel,
  percent,
  severityColour,
  shortStamp,
  videoTime,
} from '../lib/format'
import type {
  JobProgressMessage,
  SecurityEvent,
  VideoJob,
  VideoJobResult,
} from '../lib/types'
import { AlertRow } from '../components/AlertRow'
import { AnnotatedVideoPlayer } from '../components/AnnotatedVideoPlayer'
import { IncidentModal } from '../components/IncidentModal'
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  Progress,
  Readout,
} from '../components/Primitives'

const TERMINAL = new Set(['COMPLETED', 'FAILED', 'CANCELLED'])

export function VideoAnalysis({
  jobProgress,
}: {
  jobProgress: Map<string, JobProgressMessage>
}) {
  const [file, setFile] = useState<File | null>(null)
  const [profile, setProfile] = useState('standard')
  const [uploading, setUploading] = useState(false)
  const [uploadFraction, setUploadFraction] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [selected, setSelected] = useState<SecurityEvent | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const { data: profiles } = usePolling(() => api.profiles(), 0)
  const { data: jobs, refresh: refreshJobs } = usePolling(() => api.jobs(), 6_000)

  const activeJob = jobs?.find((j) => j.job_id === activeJobId) ?? null
  const runningCount = (jobs ?? []).filter((j) => !TERMINAL.has(j.status)).length

  // A websocket completion should update the list immediately rather than
  // waiting out the poll interval.
  useEffect(() => {
    for (const message of jobProgress.values()) {
      if (TERMINAL.has(message.status)) {
        refreshJobs()
        break
      }
    }
  }, [jobProgress, refreshJobs])

  const pick = useCallback((chosen: File | null) => {
    setError(null)
    setFile(chosen)
  }, [])

  async function start() {
    if (!file) return
    setUploading(true)
    setUploadFraction(0)
    setError(null)
    try {
      const result = await api.uploadVideo(file, profile, setUploadFraction)
      setActiveJobId(result.job_id)
      setFile(null)
      if (inputRef.current) inputRef.current.value = ''
      refreshJobs()
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Upload failed')
    } finally {
      setUploading(false)
      setUploadFraction(0)
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="Forensics"
        title="Video Analysis"
        subtitle="Run the full detection pipeline over a recorded file. Analysis happens on the server — you can leave this page."
      />

      <div
        className="analysis-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 340px) minmax(0, 1fr)',
          gap: 'var(--s4)',
          alignItems: 'start',
        }}
      >
        {/* ── Upload ───────────────────────────────────────────────────────── */}
        <section className="panel">
          <div className="panel-head">
            <span className="eyebrow">New analysis</span>
          </div>
          <div className="panel-body col" style={{ gap: 'var(--s4)' }}>
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                pick(e.dataTransfer.files[0] ?? null)
              }}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
              }}
              style={{
                border: `1px dashed ${dragging ? 'var(--hivis)' : 'var(--line-strong)'}`,
                background: dragging ? 'var(--hivis-wash)' : 'var(--bg-elev)',
                borderRadius: 'var(--r-sm)',
                padding: 'var(--s6) var(--s4)',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'border-color var(--t-fast) var(--ease), background var(--t-fast) var(--ease)',
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm,video/*"
                onChange={(e) => pick(e.target.files?.[0] ?? null)}
                style={{ display: 'none' }}
              />
              {file ? (
                <div className="col" style={{ gap: 4 }}>
                  <span style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}>
                    {file.name}
                  </span>
                  <span className="mono dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                    {bytes(file.size)}
                  </span>
                </div>
              ) : (
                <div className="col" style={{ gap: 4 }}>
                  <span style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}>
                    Drop a video here
                  </span>
                  <span className="dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                    or click to browse · MP4, AVI, MOV, MKV, WebM
                  </span>
                </div>
              )}
            </div>

            <div className="field">
              <label className="eyebrow" htmlFor="profile">
                Detection profile
              </label>
              <select
                id="profile"
                className="select"
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
              >
                {(profiles?.profiles ?? []).map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
              <span className="dim" style={{ fontSize: 'var(--fs-tiny)', lineHeight: 1.5 }}>
                {profiles?.profiles.find((p) => p.name === profile)?.description ?? ''}
              </span>
            </div>

            {error && <ErrorBanner message={error} />}

            {uploading ? (
              <div className="col" style={{ gap: 'var(--s2)' }}>
                <div className="row-between">
                  <span className="eyebrow">Uploading</span>
                  <span className="mono" style={{ fontSize: 'var(--fs-tiny)' }}>
                    {percent(uploadFraction)}
                  </span>
                </div>
                <Progress value={uploadFraction} />
              </div>
            ) : (
              <button className="btn btn-primary" onClick={start} disabled={!file}>
                Start analysis
              </button>
            )}

            {runningCount > 0 && (
              <span className="dim" style={{ fontSize: 'var(--fs-tiny)' }}>
                {runningCount} job{runningCount === 1 ? '' : 's'} in progress.
              </span>
            )}
          </div>
        </section>

        {/* ── Results ──────────────────────────────────────────────────────── */}
        <div className="col" style={{ gap: 'var(--s4)' }}>
          {activeJob && (
            <JobDetail
              job={activeJob}
              live={jobProgress.get(activeJob.job_id)}
              onSelectEvent={setSelected}
              onChanged={refreshJobs}
            />
          )}

          <section className="panel">
            <div className="panel-head">
              <span className="eyebrow">Analysis history</span>
              <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
                {jobs?.length ?? 0} JOB{jobs?.length === 1 ? '' : 'S'}
              </span>
            </div>
            <div className="panel-body">
              {!jobs?.length ? (
                <EmptyState
                  title="No analyses yet"
                  hint="Upload a recording to run person detection, PPE checks and behaviour analysis over it."
                />
              ) : (
                <div className="col" style={{ gap: 'var(--s2)' }}>
                  {jobs.map((job) => (
                    <JobRow
                      key={job.job_id}
                      job={job}
                      live={jobProgress.get(job.job_id)}
                      selected={job.job_id === activeJobId}
                      onOpen={() => setActiveJobId(job.job_id)}
                    />
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>

      {selected && (
        <IncidentModal event={selected} onClose={() => setSelected(null)} />
      )}
    </>
  )
}

/* ── Job row ─────────────────────────────────────────────────────────────── */
function JobRow({
  job,
  live,
  selected,
  onOpen,
}: {
  job: VideoJob
  live?: JobProgressMessage
  selected: boolean
  onOpen: () => void
}) {
  const status = live?.status ?? job.status
  const progress = live?.progress ?? job.progress
  const tone = statusTone(status)

  return (
    <button
      onClick={onOpen}
      className="alert-row"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--s3)',
        width: '100%',
        padding: 'var(--s3)',
        textAlign: 'left',
        borderRadius: 'var(--r-sm)',
        border: `1px solid ${selected ? 'var(--hivis-line)' : 'var(--line-soft)'}`,
        background: selected ? 'var(--hivis-wash)' : 'var(--surface)',
      }}
    >
      <span
        className="tally"
        style={{ background: tone }}
        aria-hidden
      />
      <div className="col grow" style={{ gap: 3, minWidth: 0 }}>
        <span className="truncate" style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}>
          {job.filename}
        </span>
        <div className="row" style={{ gap: 'var(--s3)' }}>
          <span className="mono" style={{ fontSize: 'var(--fs-micro)', color: tone }}>
            {status}
          </span>
          <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
            {job.profile.toUpperCase()}
          </span>
          <span className="mono dim-2" style={{ fontSize: 'var(--fs-micro)' }}>
            {shortStamp(job.created_at)}
          </span>
        </div>
        {!TERMINAL.has(status) && (
          <Progress value={progress} height={3} style={{ marginTop: 2 }} />
        )}
      </div>
      <div className="row" style={{ gap: 'var(--s4)', flexShrink: 0 }}>
        <Readout
          label="Incidents"
          value={String(live?.events_created ?? job.events_created)}
          size="sm"
          tone={(live?.events_created ?? job.events_created) > 0 ? 'var(--sev-high)' : undefined}
        />
        <Readout
          label="People"
          value={String(live?.people_detected ?? job.people_detected)}
          size="sm"
        />
      </div>
    </button>
  )
}

/* ── Job detail ──────────────────────────────────────────────────────────── */
function JobDetail({
  job,
  live,
  onSelectEvent,
  onChanged,
}: {
  job: VideoJob
  live?: JobProgressMessage
  onSelectEvent: (event: SecurityEvent) => void
  onChanged: () => void
}) {
  const status = live?.status ?? job.status
  const progress = live?.progress ?? job.progress
  const done = status === 'COMPLETED'

  // Only fetch results once there is something to fetch.
  const { data: results } = usePolling(
    () => api.events({ job_id: job.job_id, limit: 100 }),
    done ? 0 : 5_000,
    [job.job_id, done],
  )

  // The result document (player URL, encoder metadata, timeline) only exists
  // once the render is finished, so it is fetched on completion, not polled.
  const { data: result } = usePolling(
    () => api.jobResult(job.job_id),
    0,
    [job.job_id, done],
  )
  const [seekTo, setSeekTo] = useState<{ seconds: number; nonce: number } | null>(
    null,
  )

  const summary = (job.summary ?? {}) as Record<string, unknown>
  const byType = (summary.events_by_type ?? {}) as Record<string, number>

  return (
    <section className="panel">
      <div className="panel-head">
        <div className="col" style={{ gap: 1, minWidth: 0 }}>
          <span className="eyebrow">Analysis</span>
          <h2 className="truncate" style={{ fontSize: 'var(--fs-md)' }}>
            {job.filename}
          </h2>
        </div>
        <div className="row" style={{ gap: 'var(--s2)' }}>
          {!TERMINAL.has(status) && (
            <button
              className="btn btn-sm btn-danger"
              onClick={async () => {
                try {
                  await api.cancelJob(job.job_id)
                } finally {
                  onChanged()
                }
              }}
            >
              Cancel
            </button>
          )}
          {done && result?.annotated_ready && result.download_url && (
            <a
              className="btn btn-sm"
              href={api.annotatedDownloadUrl(job.job_id)}
              download
            >
              Download annotated video
            </a>
          )}
        </div>
      </div>

      <div className="panel-body col" style={{ gap: 'var(--s4)' }}>
        {!TERMINAL.has(status) ? (
          <div className="col" style={{ gap: 'var(--s2)' }}>
            <div className="row-between">
              <span className="mono" style={{ fontSize: 'var(--fs-sm)' }}>
                {live?.message ?? job.message}
              </span>
              <span className="mono" style={{ fontSize: 'var(--fs-sm)', fontWeight: 600 }}>
                {percent(progress, 1)}
              </span>
            </div>
            <Progress value={progress} />
            {/* Every figure here is a counter from the worker — nothing is
                interpolated, so a stalled job reads as stalled. */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(104px, 1fr))',
                gap: 'var(--s3)',
                marginTop: 'var(--s2)',
              }}
            >
              <Readout
                label="Frames"
                value={`${live?.processed_frames ?? job.processed_frames} / ${
                  live?.total_frames ?? job.total_frames
                }`}
                size="sm"
              />
              <Readout
                label="Processing"
                value={
                  live?.processing_fps != null ? `${live.processing_fps} fps` : '—'
                }
                size="sm"
              />
              <Readout
                label="Events"
                value={String(live?.events_created ?? job.events_created)}
                size="sm"
                tone={
                  (live?.events_created ?? job.events_created) > 0
                    ? 'var(--sev-high)'
                    : undefined
                }
              />
              <Readout
                label="People"
                value={String(live?.people_detected ?? job.people_detected)}
                size="sm"
              />
              <Readout
                label="PPE violations"
                value={String(live?.ppe_violations ?? 0)}
                size="sm"
                tone={(live?.ppe_violations ?? 0) > 0 ? 'var(--sev-high)' : undefined}
              />
              <Readout
                label="Remaining"
                value={
                  live?.eta_seconds != null ? duration(live.eta_seconds) : '—'
                }
                size="sm"
              />
            </div>
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
              gap: 'var(--s3)',
            }}
          >
            <Readout label="Status" value={status} size="sm" tone={statusTone(status)} />
            <Readout label="Resolution" value={job.resolution} size="sm" />
            <Readout label="Duration" value={duration(job.duration_seconds)} size="sm" />
            <Readout label="Frames" value={String(job.processed_frames)} size="sm" />
            <Readout
              label="Incidents"
              value={String(job.events_created)}
              size="sm"
              tone={job.events_created > 0 ? 'var(--sev-high)' : undefined}
            />
            <Readout label="People" value={String(job.people_detected)} size="sm" />
            {typeof summary.realtime_factor === 'number' && (
              <Readout
                label="Speed"
                value={`${summary.realtime_factor}× realtime`}
                size="sm"
                title="How much faster than playback the analysis ran"
              />
            )}
            {typeof summary.backend === 'string' && (
              <Readout label="Backend" value={summary.backend} size="sm" />
            )}
            {typeof summary.ppe_method === 'string' && (
              <Readout
                label="PPE method"
                value={summary.ppe_method}
                size="sm"
                tone={summary.ppe_method === 'heuristic' ? 'var(--sev-medium)' : undefined}
                title={
                  summary.ppe_method === 'heuristic'
                    ? 'No PPE model installed — PPE state was estimated from colour regions'
                    : undefined
                }
              />
            )}
          </div>
        )}

        {status === 'FAILED' && <ErrorBanner message={job.message} />}

        {Object.keys(byType).length > 0 && (
          <div className="row" style={{ gap: 'var(--s2)', flexWrap: 'wrap' }}>
            {Object.entries(byType).map(([type, count]) => (
              <span key={type} className="chip">
                {eventLabel(type)} <strong style={{ marginLeft: 4 }}>{count}</strong>
              </span>
            ))}
          </div>
        )}

        {done && (
          <AnnotatedResult
            jobId={job.job_id}
            result={result}
            seekTo={seekTo}
            onSeekRequest={setSeekTo}
          />
        )}

        {results?.items.length ? (
          <div className="col alert-list" style={{ gap: 'var(--s2)' }}>
            <span className="eyebrow">Incidents found</span>
            {results.items.map((event) => (
              <div key={event.event_id} className="col" style={{ gap: 2 }}>
                <AlertRow event={event} onOpen={onSelectEvent} showCamera={false} />
                {event.video_timestamp !== null && (
                  <button
                    className="mono dim-2"
                    onClick={() =>
                      setSeekTo({
                        seconds: event.video_timestamp!,
                        nonce: Date.now(),
                      })
                    }
                    title="Jump the annotated video to this moment"
                    style={{
                      alignSelf: 'flex-start',
                      fontSize: 'var(--fs-micro)',
                      marginLeft: 'var(--s3)',
                      padding: '2px 6px',
                      border: '1px solid var(--line-strong)',
                      borderRadius: 'var(--r-xs)',
                      background: 'var(--surface-2)',
                      color: 'var(--text-2)',
                      cursor: 'pointer',
                    }}
                  >
                    ▶ {videoTime(event.video_timestamp)} IN SOURCE
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : done ? (
          <EmptyState
            title="No incidents detected"
            hint="The pipeline found nothing matching the selected profile in this recording."
          />
        ) : null}
      </div>
    </section>
  )
}

/* ── Annotated result ────────────────────────────────────────────────────── */
/**
 * The completed analysis, led by the video.
 *
 * The render is the result — not a link to a file, not a grid of stills — so
 * the player is the first thing on screen once processing finishes. Encoder
 * warnings sit underneath it rather than being swallowed: an operator about to
 * rely on this footage should know if the audio was dropped or the resolution
 * reduced.
 */
function AnnotatedResult({
  jobId,
  result,
  seekTo,
  onSeekRequest,
}: {
  jobId: string
  result: VideoJobResult | null | undefined
  seekTo: { seconds: number; nonce: number } | null
  onSeekRequest: (target: { seconds: number; nonce: number }) => void
}) {
  if (!result) {
    return (
      <div className="col" style={{ gap: 'var(--s2)' }}>
        <span className="eyebrow">Annotated video</span>
        <div
          style={{
            aspectRatio: '16 / 9',
            borderRadius: 'var(--r-sm)',
            background: 'var(--surface-2)',
          }}
          className="skeleton"
        />
      </div>
    )
  }

  if (!result.annotated_ready) {
    return (
      <ErrorBanner
        message={
          result.warnings[0] ??
          'This analysis completed without producing a playable annotated video.'
        }
      />
    )
  }

  return (
    <div className="col" style={{ gap: 'var(--s3)' }}>
      <div className="row-between" style={{ gap: 'var(--s2)', flexWrap: 'wrap' }}>
        <span className="eyebrow">Annotated video — full analysis</span>
        <div className="row" style={{ gap: 'var(--s2)' }}>
          <span className="chip" title="Container and video codec of the render">
            {result.resolution} · {result.codec.toUpperCase()} ·{' '}
            {bytes(result.size_bytes)}
          </span>
          <span
            className="chip"
            title={
              result.has_audio
                ? 'The original audio track was preserved'
                : 'No audio in the render — see the warnings below'
            }
          >
            {result.has_audio ? 'AUDIO' : 'NO AUDIO'}
          </span>
        </div>
      </div>

      <AnnotatedVideoPlayer
        src={api.annotatedVideoUrl(jobId)}
        duration={result.duration_seconds}
        timeline={result.timeline}
        seekTo={seekTo}
      />

      <div className="row" style={{ gap: 'var(--s2)', flexWrap: 'wrap' }}>
        <a className="btn btn-sm" href={api.annotatedDownloadUrl(jobId)} download>
          Download annotated video
        </a>
        {result.original_url && (
          <a
            className="btn btn-sm btn-ghost"
            href={api.originalDownloadUrl(jobId)}
            download
          >
            Download original
          </a>
        )}
      </div>

      {/* Event timeline. Clicking a row seeks the player above. */}
      {result.timeline.length > 0 && (
        <div className="col" style={{ gap: 'var(--s2)' }}>
          <span className="eyebrow">Event timeline</span>
          <div className="col" style={{ gap: 2 }}>
            {result.timeline.map((entry) => (
              <button
                key={entry.event_id}
                onClick={() =>
                  onSeekRequest({
                    seconds: entry.video_timestamp ?? 0,
                    nonce: Date.now(),
                  })
                }
                className="row"
                style={{
                  gap: 'var(--s3)',
                  alignItems: 'center',
                  width: '100%',
                  textAlign: 'left',
                  padding: '6px var(--s3)',
                  border: '1px solid var(--line-soft)',
                  borderLeft: `3px solid ${
                    severityColour[entry.severity] ?? 'var(--sev-medium)'
                  }`,
                  borderRadius: 'var(--r-xs)',
                  background: 'var(--surface)',
                  color: 'var(--text)',
                  cursor: 'pointer',
                  fontSize: 'var(--fs-sm)',
                }}
              >
                <span className="mono" style={{ color: 'var(--text-2)', minWidth: 52 }}>
                  {videoTime(entry.video_timestamp ?? 0)}
                </span>
                <span style={{ flex: 1 }}>{entry.label}</span>
                {entry.person_id != null && (
                  <span className="mono dim" style={{ fontSize: 'var(--fs-micro)' }}>
                    #{String(entry.person_id).padStart(3, '0')}
                  </span>
                )}
                <span className="mono dim" style={{ fontSize: 'var(--fs-micro)' }}>
                  {percent(entry.confidence, 0)}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {result.warnings.length > 0 && (
        <div className="col" style={{ gap: 4 }}>
          {result.warnings.map((warning) => (
            <span
              key={warning}
              style={{
                fontSize: 'var(--fs-tiny)',
                color: 'var(--text-2)',
                padding: 'var(--s2) var(--s3)',
                borderRadius: 'var(--r-xs)',
                border: '1px solid var(--sev-medium)',
                background: 'var(--sev-medium-wash)',
              }}
            >
              {warning}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function statusTone(status: string): string {
  switch (status) {
    case 'COMPLETED':
      return 'var(--live)'
    case 'RUNNING':
      return 'var(--hivis)'
    case 'FAILED':
      return 'var(--error)'
    case 'CANCELLED':
      return 'var(--text-3)'
    default:
      return 'var(--sev-medium)'
  }
}
