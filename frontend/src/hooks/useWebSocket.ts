/**
 * Realtime feed.
 *
 * One shared socket for the whole app: a browser holding several sockets to
 * the same origin wastes connections and multiplies reconnect storms. The hook
 * exposes handlers per message type and reconnects with exponential backoff
 * plus jitter, so a backend restart does not produce a synchronised stampede
 * from every open tab.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { websocketUrl } from '../lib/api'
import type {
  CameraStatusMessage,
  DetectionFrame,
  JobProgressMessage,
  SecurityEvent,
  WsMessage,
} from '../lib/types'

export type ConnectionState = 'connecting' | 'open' | 'closed'

interface Handlers {
  onDetections?: (frame: DetectionFrame) => void
  onEvent?: (event: SecurityEvent) => void
  onCameraStatus?: (status: CameraStatusMessage) => void
  onJobProgress?: (progress: JobProgressMessage) => void
}

const MAX_BACKOFF = 15_000

export function useWebSocket(handlers: Handlers, enabled = true): {
  state: ConnectionState
  lastMessageAt: number | null
} {
  const [state, setState] = useState<ConnectionState>('connecting')
  const [lastMessageAt, setLastMessageAt] = useState<number | null>(null)

  // Handlers change identity every render; a ref keeps the socket from being
  // torn down and rebuilt on each one.
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers

  const socketRef = useRef<WebSocket | null>(null)
  const attemptRef = useRef(0)
  const timerRef = useRef<number | null>(null)
  const closedByUs = useRef(false)

  const connect = useCallback(() => {
    if (!enabled) return
    closedByUs.current = false
    setState('connecting')

    let socket: WebSocket
    try {
      socket = new WebSocket(websocketUrl())
    } catch {
      scheduleReconnect()
      return
    }
    socketRef.current = socket

    socket.onopen = () => {
      attemptRef.current = 0
      setState('open')
    }

    socket.onmessage = (raw) => {
      setLastMessageAt(Date.now())
      let message: WsMessage
      try {
        message = JSON.parse(raw.data as string)
      } catch {
        return
      }
      const h = handlersRef.current
      switch (message.type) {
        case 'detections':
          h.onDetections?.(message.payload)
          break
        case 'event':
          h.onEvent?.(message.payload)
          break
        case 'camera_status':
          h.onCameraStatus?.(message.payload)
          break
        case 'job_progress':
          h.onJobProgress?.(message.payload)
          break
        default:
          break
      }
    }

    socket.onclose = () => {
      setState('closed')
      if (!closedByUs.current) scheduleReconnect()
    }

    socket.onerror = () => {
      // onclose always follows; reconnect is handled there.
    }

    function scheduleReconnect() {
      const attempt = (attemptRef.current += 1)
      const base = Math.min(MAX_BACKOFF, 500 * 2 ** Math.min(attempt, 5))
      // Jitter so many tabs do not all retry on the same tick.
      const delay = base * (0.7 + Math.random() * 0.6)
      timerRef.current = window.setTimeout(connect, delay)
    }
  }, [enabled])

  useEffect(() => {
    if (!enabled) {
      setState('closed')
      return
    }
    connect()
    return () => {
      closedByUs.current = true
      if (timerRef.current) window.clearTimeout(timerRef.current)
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [connect, enabled])

  return { state, lastMessageAt }
}
