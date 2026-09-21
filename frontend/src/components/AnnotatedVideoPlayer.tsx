/**
 * The annotated render, played as video.
 *
 * This is the primary result of an analysis, so it is presented as footage —
 * a real HTML5 player with a real timeline — not as a gallery of stills. The
 * native controls do the work they are good at (scrubbing, volume, fullscreen,
 * picture-in-picture, keyboard access, captions) and this component adds only
 * what they lack: playback speed, and incident markers positioned on the
 * timeline so an operator can see *where* in the footage things happened and
 * jump straight there.
 *
 * Seeking depends on the server answering range requests and on the file
 * carrying its index at the front — both handled in the encoder and the route.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { TimelineEntry } from '../lib/types'
import { severityColour } from '../lib/format'

const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4]

function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00'
  const whole = Math.floor(seconds)
  const m = Math.floor(whole / 60)
  const s = whole % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

export function AnnotatedVideoPlayer({
  src,
  poster,
  duration,
  timeline = [],
  onSelectEvent,
  seekTo,
}: {
  src: string
  poster?: string
  duration: number
  timeline?: TimelineEntry[]
  onSelectEvent?: (entry: TimelineEntry) => void
  /** External seek request — a timeline row being clicked. */
  seekTo?: { seconds: number; nonce: number } | null
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [current, setCurrent] = useState(0)
  const [length, setLength] = useState(duration)
  const [speed, setSpeed] = useState(1)
  const [error, setError] = useState<string | null>(null)

  // Markers need a duration to be positioned against. Prefer the value the
  // browser parsed from the file over the one the API reported: if they ever
  // disagree, the browser's is the one the scrubber uses.
  const total = length > 0 ? length : duration

  const markers = useMemo(
    () =>
      timeline
        .filter((e) => e.video_timestamp != null && total > 0)
        .map((e) => ({ entry: e, left: Math.min(100, (e.video_timestamp! / total) * 100) })),
    [timeline, total],
  )

  const seek = useCallback((seconds: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.max(0, seconds)
    void video.play().catch(() => {
      /* autoplay may be blocked; the user can press play */
    })
  }, [])

  useEffect(() => {
    if (seekTo) seek(seekTo.seconds)
    // `nonce` makes repeat clicks on the same marker re-fire the seek.
  }, [seekTo?.nonce, seekTo?.seconds, seek, seekTo])

  useEffect(() => {
    const video = videoRef.current
    if (video) video.playbackRate = speed
  }, [speed])

  return (
    <div className="col" style={{ gap: 'var(--s2)' }}>
      <div
        style={{
          position: 'relative',
          background: '#050609',
          borderRadius: 'var(--r-sm)',
          border: '1px solid var(--line)',
          overflow: 'hidden',
        }}
      >
        <video
          ref={videoRef}
          src={src}
          poster={poster}
          controls
          playsInline
          preload="metadata"
          onLoadedMetadata={(e) => setLength(e.currentTarget.duration || duration)}
          onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
          onError={() =>
            setError(
              'The browser could not play this file. Check the codec reported ' +
                'in the analysis warnings.',
            )
          }
          style={{ width: '100%', display: 'block', maxHeight: '68vh' }}
        />
        {error && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'grid',
              placeItems: 'center',
              padding: 'var(--s5)',
              background: 'rgba(5,6,9,0.9)',
              color: 'var(--sev-critical)',
              fontSize: 'var(--fs-sm)',
              textAlign: 'center',
            }}
          >
            {error}
          </div>
        )}
      </div>

      {/* Incident markers. The native scrubber cannot show these, and where
          an incident sits in the footage is most of what triage needs. */}
      {markers.length > 0 && (
        <div className="col" style={{ gap: 4 }}>
          <div
            style={{
              position: 'relative',
              height: 22,
              borderRadius: 'var(--r-xs)',
              background: 'var(--surface-2)',
              border: '1px solid var(--line)',
            }}
          >
            {markers.map(({ entry, left }) => (
              <button
                key={entry.event_id}
                onClick={() => {
                  seek(entry.video_timestamp!)
                  onSelectEvent?.(entry)
                }}
                title={`${entry.label} — ${clock(entry.video_timestamp!)}${
                  entry.person_id != null ? ` · person #${entry.person_id}` : ''
                }`}
                aria-label={`Jump to ${entry.label} at ${clock(entry.video_timestamp!)}`}
                style={{
                  position: 'absolute',
                  left: `${left}%`,
                  top: 2,
                  width: 4,
                  height: 16,
                  padding: 0,
                  transform: 'translateX(-2px)',
                  borderRadius: 1,
                  border: 'none',
                  cursor: 'pointer',
                  background: severityColour[entry.severity] ?? 'var(--sev-medium)',
                }}
              />
            ))}
            {/* Playhead, so the markers read against the current position. */}
            {total > 0 && (
              <div
                style={{
                  position: 'absolute',
                  left: `${Math.min(100, (current / total) * 100)}%`,
                  top: 0,
                  bottom: 0,
                  width: 1,
                  background: 'var(--text)',
                  pointerEvents: 'none',
                }}
              />
            )}
          </div>
          <div className="row-between" style={{ fontSize: 'var(--fs-micro)' }}>
            <span className="mono dim">
              {clock(current)} / {clock(total)}
            </span>
            <div className="row" style={{ gap: 4, alignItems: 'center' }}>
              <span className="dim" style={{ fontSize: 'var(--fs-micro)' }}>
                SPEED
              </span>
              {SPEEDS.map((rate) => (
                <button
                  key={rate}
                  onClick={() => setSpeed(rate)}
                  className="chip"
                  style={{
                    cursor: 'pointer',
                    color: rate === speed ? 'var(--bg)' : 'var(--text-2)',
                    background: rate === speed ? 'var(--hivis)' : 'var(--surface-2)',
                    borderColor: rate === speed ? 'var(--hivis)' : 'var(--line-strong)',
                  }}
                >
                  {rate}x
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export { clock as videoClock }
