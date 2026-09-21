/**
 * Polled data fetching.
 *
 * Deliberately small — no data-fetching library for a handful of endpoints.
 * Two behaviours matter here:
 *
 * - Polling pauses when the tab is hidden. A wall-mounted console left open
 *   overnight should not keep hammering the API from a background tab.
 * - `refresh()` is exposed so a WebSocket message can trigger an immediate
 *   re-fetch instead of waiting for the next interval.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../lib/api'

export interface PollResult<T> {
  data: T | null
  error: string | null
  loading: boolean
  refresh: () => void
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs = 5000,
  deps: unknown[] = [],
): PollResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher
  const mounted = useRef(true)
  const inFlight = useRef(false)

  const load = useCallback(async () => {
    // Skip if a previous request is still running — a slow endpoint must not
    // queue up overlapping calls.
    if (inFlight.current) return
    inFlight.current = true
    try {
      const result = await fetcherRef.current()
      if (!mounted.current) return
      setData(result)
      setError(null)
    } catch (cause) {
      if (!mounted.current) return
      setError(
        cause instanceof ApiError ? cause.message : 'Unexpected error loading data',
      )
    } finally {
      inFlight.current = false
      if (mounted.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    setLoading(true)
    void load()

    if (intervalMs <= 0) return () => { mounted.current = false }

    let timer = window.setInterval(() => {
      if (!document.hidden) void load()
    }, intervalMs)

    // Re-fetch immediately when the tab becomes visible again, so the operator
    // never looks at stale data after switching back.
    const onVisible = () => {
      if (!document.hidden) void load()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      mounted.current = false
      window.clearInterval(timer)
      timer = 0
      document.removeEventListener('visibilitychange', onVisible)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, intervalMs, ...deps])

  return { data, error, loading, refresh: load }
}

/** A ticking clock, for the console's live timecode. */
export function useClock(intervalMs = 1000): Date {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}
