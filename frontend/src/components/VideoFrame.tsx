/**
 * A video frame that guarantees the overlay lines up with the picture.
 *
 * Detection coordinates are normalised to the *source* frame, so the rendered
 * image must not be cropped or letterboxed relative to the box the overlay
 * draws into — `object-fit: cover` would silently shift every bounding box,
 * and a `contain` fit inside a differently-shaped container would leave bars
 * that the overlay would treat as picture.
 *
 * This component solves it by sizing its own media box to the stream's
 * *intrinsic* aspect ratio (read from the first decoded frame) and reporting
 * that box's measured pixel size to the caller. The image then fills it
 * exactly, and normalised coordinates map linearly.
 */

import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

interface Props {
  src: string | null
  alt: string
  /** Rendered inside the media box, above the image. */
  children?: (size: { width: number; height: number }) => ReactNode
  /** Shown instead of the image (offline, paused, error). */
  placeholder?: ReactNode
  /** Aspect ratio to use before the first frame decodes. */
  fallbackAspect?: number
  /** `fit` fills the available height (fullscreen); `flow` sizes by aspect. */
  mode?: 'flow' | 'fit'
  /**
   * Change to force the image to remount. An interrupted MJPEG connection
   * never recovers on its own, so re-requesting it is the only way back.
   */
  reloadKey?: number
  onError?: () => void
}

export function VideoFrame({
  src,
  alt,
  children,
  placeholder,
  fallbackAspect = 16 / 9,
  mode = 'flow',
  reloadKey = 0,
  onError,
}: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const [aspect, setAspect] = useState(fallbackAspect)
  const [size, setSize] = useState({ width: 0, height: 0 })

  const measure = useCallback(() => {
    const element = boxRef.current
    if (!element) return
    const rect = element.getBoundingClientRect()
    setSize((current) =>
      Math.abs(current.width - rect.width) < 0.5 &&
      Math.abs(current.height - rect.height) < 0.5
        ? current
        : { width: rect.width, height: rect.height },
    )
  }, [])

  useEffect(() => {
    const element = boxRef.current
    if (!element) return
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    measure()
    return () => observer.disconnect()
  }, [measure])

  return (
    <div
      style={
        mode === 'fit'
          ? {
              position: 'relative',
              flex: 1,
              minHeight: 0,
              display: 'grid',
              placeItems: 'center',
              overflow: 'hidden',
            }
          : { position: 'relative', width: '100%' }
      }
    >
      <div
        ref={boxRef}
        style={{
          position: 'relative',
          aspectRatio: String(aspect),
          background: '#050609',
          overflow: 'hidden',
          // In `fit` mode the box grows to whichever dimension binds first,
          // staying at the true aspect so no bars appear inside it.
          ...(mode === 'fit'
            ? { maxWidth: '100%', maxHeight: '100%', width: 'auto', height: '100%' }
            : { width: '100%' }),
        }}
      >
        {src ? (
          <img
            key={reloadKey}
            src={src}
            alt={alt}
            onLoad={(event) => {
              const image = event.currentTarget
              if (image.naturalWidth > 0 && image.naturalHeight > 0) {
                const ratio = image.naturalWidth / image.naturalHeight
                if (Math.abs(ratio - aspect) > 0.01) setAspect(ratio)
              }
              measure()
            }}
            onError={onError}
            style={{
              display: 'block',
              width: '100%',
              height: '100%',
              // The box already matches the stream's aspect ratio, so `contain`
              // neither crops nor letterboxes — it just fills.
              objectFit: 'contain',
            }}
          />
        ) : (
          placeholder
        )}

        {src && size.width > 0 && children?.(size)}
      </div>
    </div>
  )
}
