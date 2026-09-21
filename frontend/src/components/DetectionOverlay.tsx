/**
 * Detection overlay: SVG drawn over the live frame.
 *
 * Why SVG in the browser rather than boxes burned into the video: the server
 * would have to re-encode every frame, the operator could not toggle layers,
 * and text would blur with the stream. The pipeline sends normalised
 * coordinates and this draws crisp vectors at whatever size the tile is.
 *
 * The viewBox is in **pixel space** (`0 0 width height`), matching the
 * rendered video box, and normalised coordinates are multiplied up. An earlier
 * version used a `0 0 100 100` viewBox with `preserveAspectRatio="none"`,
 * which is a trap: font sizes then land in user units — an 11pt label becomes
 * 11% of the frame — and the non-uniform scale stretches glyphs and corner
 * ticks. Pixel space keeps text at its stated size and geometry undistorted.
 *
 * This requires the caller to pass the video box's true pixel size *and* to
 * ensure the <img> is not cropped: `object-fit: cover` would shift every box
 * relative to what the operator sees.
 */

import { useMemo } from 'react'
import type { DetectionBox, Zone } from '../lib/types'

export interface OverlayLayers {
  boxes: boolean
  ids: boolean
  ppe: boolean
  zones: boolean
  trails: boolean
}

export const DEFAULT_LAYERS: OverlayLayers = {
  boxes: true,
  ids: true,
  ppe: true,
  zones: true,
  trails: false,
}

interface Props {
  boxes: DetectionBox[]
  zones?: Zone[]
  layers?: OverlayLayers
  /** True rendered pixel size of the video box. */
  width: number
  height: number
  onSelectPerson?: (trackId: number) => void
}

/** Label size, tuned so text stays legible on a small tile and does not
 *  balloon in fullscreen. */
function labelScale(width: number): number {
  return Math.max(0.66, Math.min(1.5, width / 640))
}

export function DetectionOverlay({
  boxes,
  zones = [],
  layers = DEFAULT_LAYERS,
  width,
  height,
  onSelectPerson,
}: Props) {
  if (width <= 0 || height <= 0) return null

  const scale = labelScale(width)
  const fontSize = 11 * scale
  const rowHeight = 16 * scale
  const pad = 4 * scale

  const zonePaths = useMemo(
    () =>
      zones
        .filter((z) => z.enabled && z.polygon.length >= 3)
        .map((zone) => ({
          zone,
          points: zone.polygon
            .map(([x, y]) => `${x * width},${y * height}`)
            .join(' '),
          // Anchor the label at the topmost vertex so it does not sit over a
          // person standing in the middle of the zone.
          anchor: zone.polygon.reduce(
            (best, p) => (p[1] < best[1] ? p : best),
            zone.polygon[0],
          ),
        })),
    [zones, width, height],
  )

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
      aria-hidden
    >
      {/* ── Zones ────────────────────────────────────────────────────────── */}
      {layers.zones &&
        zonePaths.map(({ zone, points }) => (
          <polygon
            key={zone.zone_id}
            points={points}
            fill={zone.colour}
            fillOpacity={0.1}
            stroke={zone.colour}
            strokeWidth={1.5}
            strokeDasharray="5 4"
          />
        ))}

      {/* ── People ───────────────────────────────────────────────────────── */}
      {layers.boxes &&
        boxes.map((box, index) => (
          <PersonMark
            key={box.track_id ?? `d${index}`}
            box={box}
            frameWidth={width}
            frameHeight={height}
            fontSize={fontSize}
            rowHeight={rowHeight}
            pad={pad}
            scale={scale}
            showId={layers.ids}
            showPpe={layers.ppe}
            onSelect={onSelectPerson}
          />
        ))}

      {/* Zone labels last, so a box never covers one. */}
      {layers.zones &&
        zonePaths.map(({ zone, anchor }) => (
          <Label
            key={`l-${zone.zone_id}`}
            text={`${zone.name.toUpperCase()} · ${zone.zone_type.toUpperCase()}`}
            x={anchor[0] * width}
            y={Math.max(0, anchor[1] * height - rowHeight - 2)}
            fontSize={fontSize * 0.9}
            height={rowHeight * 0.9}
            pad={pad}
            frameWidth={width}
            fill="rgba(8,9,12,0.86)"
            stroke={zone.colour}
            colour={zone.colour}
          />
        ))}
    </svg>
  )
}

function PersonMark({
  box,
  frameWidth,
  frameHeight,
  fontSize,
  rowHeight,
  pad,
  scale,
  showId,
  showPpe,
  onSelect,
}: {
  box: DetectionBox
  frameWidth: number
  frameHeight: number
  fontSize: number
  rowHeight: number
  pad: number
  scale: number
  showId: boolean
  showPpe: boolean
  onSelect?: (trackId: number) => void
}) {
  const [nx1, ny1, nx2, ny2] = box.bbox
  const x = nx1 * frameWidth
  const y = ny1 * frameHeight
  const w = Math.max(2, (nx2 - nx1) * frameWidth)
  const h = Math.max(2, (ny2 - ny1) * frameHeight)

  // States an operator needs to separate at a glance.
  const colour = box.violation
    ? 'var(--sev-critical)'
    : box.zones.length > 0
      ? 'var(--sev-high)'
      : box.helmet || box.vest
        ? 'var(--hivis)'
        : 'var(--sev-low)'

  // Corner ticks: the viewfinder language. They read as a target at sizes
  // where a 1px rectangle disappears into video noise.
  const tick = Math.min(w * 0.3, h * 0.3, 22 * scale)
  const corners = [
    `M${x},${y + tick} L${x},${y} L${x + tick},${y}`,
    `M${x + w - tick},${y} L${x + w},${y} L${x + w},${y + tick}`,
    `M${x + w},${y + h - tick} L${x + w},${y + h} L${x + w - tick},${y + h}`,
    `M${x + tick},${y + h} L${x},${y + h} L${x},${y + h - tick}`,
  ]

  const idText = `PERSON #${String(box.track_id ?? 0).padStart(3, '0')}  ${Math.round(box.confidence * 100)}%`
  // Flip the header inside the box when the person is at the top of frame.
  const headerY = y - rowHeight - 2 < 0 ? y + 2 : y - rowHeight - 2

  const lines: { text: string; tone: string }[] = []
  if (showPpe && box.ppe_method && box.ppe_method !== 'disabled') {
    const mark = (v: boolean | null) => (v === true ? 'Y' : v === false ? 'N' : '?')
    lines.push({
      text: `HELMET ${mark(box.helmet)}  VEST ${mark(box.vest)}`,
      tone: box.violation ? 'var(--sev-critical)' : 'var(--hivis)',
    })
    // The method rides on its own line rather than as a suffix symbol: a
    // colour estimate must never be shown with the authority of a model
    // detection, and at overlay size a `~` is too easy to miss.
    lines.push({
      text: box.ppe_method === 'heuristic' ? 'HEURISTIC EST.' : 'AI MODEL',
      tone: box.ppe_method === 'heuristic' ? 'var(--sev-medium)' : 'var(--text-3)',
    })
  }
  if (box.speed > 1.4) {
    lines.push({
      text: `${box.speed.toFixed(1)} BH/S`,
      tone: box.speed > 2.2 ? 'var(--sev-high)' : 'var(--text-2)',
    })
  }

  return (
    <g
      style={{
        pointerEvents: onSelect ? 'auto' : 'none',
        cursor: onSelect ? 'pointer' : 'default',
      }}
      onClick={() => box.track_id !== null && onSelect?.(box.track_id)}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        fill="transparent"
        stroke={colour}
        strokeOpacity={0.4}
        strokeWidth={1}
      />
      {corners.map((d) => (
        <path
          key={d}
          d={d}
          fill="none"
          stroke={colour}
          strokeWidth={2}
          strokeLinecap="square"
        />
      ))}

      {showId && (
        <Label
          text={idText}
          x={x}
          y={headerY}
          fontSize={fontSize}
          height={rowHeight}
          pad={pad}
          frameWidth={frameWidth}
          fill={colour}
          colour="#0a0b0d"
          bold
        />
      )}

      {lines.map((line, index) => {
        const lineY = y + h + 2 + index * (rowHeight + 2)
        if (lineY + rowHeight > frameHeight) return null
        return (
          <Label
            key={line.text}
            text={line.text}
            x={x}
            y={lineY}
            fontSize={fontSize * 0.9}
            height={rowHeight * 0.9}
            pad={pad}
            frameWidth={frameWidth}
            fill="rgba(8,9,12,0.86)"
            stroke={line.tone}
            colour={line.tone}
          />
        )
      })}
    </g>
  )
}

/**
 * A text chip in pixel space. Width is estimated from the character count —
 * measuring text in SVG would need a layout pass per frame, and the mono face
 * makes 0.6em per character accurate enough to keep the chip snug.
 */
function Label({
  text,
  x,
  y,
  fontSize,
  height,
  pad,
  frameWidth,
  fill,
  stroke,
  colour,
  bold = false,
}: {
  text: string
  x: number
  y: number
  fontSize: number
  height: number
  pad: number
  frameWidth: number
  fill: string
  stroke?: string
  colour: string
  bold?: boolean
}) {
  const width = text.length * fontSize * 0.6 + pad * 2
  // Keep the chip inside the frame rather than letting it run off the edge.
  const left = Math.max(0, Math.min(x, frameWidth - width))

  return (
    <g>
      <rect
        x={left}
        y={y}
        width={width}
        height={height}
        fill={fill}
        stroke={stroke}
        strokeWidth={stroke ? 1 : 0}
        rx={2}
      />
      <text
        x={left + pad}
        y={y + height * 0.72}
        fontFamily="var(--font-mono)"
        fontSize={fontSize}
        fontWeight={bold ? 700 : 600}
        fill={colour}
        style={{ letterSpacing: '0.04em' }}
      >
        {text}
      </text>
    </g>
  )
}
