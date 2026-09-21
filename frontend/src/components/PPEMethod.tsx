/**
 * How a PPE finding was reached — never hidden, never implied.
 *
 * A trained detector and a colour heuristic are not the same kind of claim, so
 * the interface refuses to render them the same way. The distinction is
 * carried by the words first — `AI MODEL` and `HEURISTIC FALLBACK` are always
 * spelled out, never abbreviated to a symbol — and by emphasis second: the
 * model badge is quiet because it is the expected state, while the heuristic
 * badge is filled amber because it is the exception an operator must notice.
 *
 * Emphasis rather than hue is deliberate. The two states must stay separable
 * for a deuteranopic viewer, so they are never distinguished by a green/amber
 * pair; the amber is the same caution colour used for MEDIUM severity, and the
 * model badge carries no accent hue at all.
 */

export type PPEMethod = 'model' | 'heuristic' | 'disabled' | string

type Presentation = {
  label: string
  colour: string
  border: string
  background: string
  title: string
}

const PRESENTATION: Record<string, Presentation> = {
  model: {
    label: 'AI MODEL',
    colour: 'var(--text-2)',
    border: 'var(--line-strong)',
    background: 'var(--surface-2)',
    title: 'Detected by a trained PPE object-detection model',
  },
  heuristic: {
    label: 'HEURISTIC FALLBACK',
    colour: 'var(--sev-medium)',
    border: 'var(--sev-medium)',
    background: 'rgba(255, 197, 61, 0.12)',
    title:
      'Estimated from HSV colour statistics over the head and torso regions — ' +
      'no trained PPE model is installed. Advisory only: confidence is capped ' +
      'at 0.62 and bright clothing or unusual lighting can fool it.',
  },
  disabled: {
    label: 'PPE DISABLED',
    colour: 'var(--text-3)',
    border: 'var(--line-strong)',
    background: 'transparent',
    title: 'PPE assessment is switched off; this finding predates that change',
  },
}

export function presentationFor(method: PPEMethod | null | undefined): Presentation | null {
  if (!method) return null
  return PRESENTATION[method] ?? null
}

/** Compact chip for dense rows. */
export function PPEMethodBadge({
  method,
  short = false,
}: {
  method: PPEMethod | null | undefined
  short?: boolean
}) {
  const p = presentationFor(method)
  if (!p) return null
  return (
    <span
      className="chip"
      title={p.title}
      style={{
        color: p.colour,
        borderColor: p.border,
        background: p.background,
        fontWeight: method === 'heuristic' ? 700 : 600,
      }}
    >
      {short && method === 'heuristic' ? 'HEURISTIC' : p.label}
    </span>
  )
}

/**
 * The full line for an incident: what is missing, how sure, and by what means.
 * Reads as `Helmet Missing / Confidence: 94% / Method: AI MODEL`.
 */
export function PPEMethodLine({
  method,
  confidence,
  finding,
}: {
  method: PPEMethod | null | undefined
  confidence?: number | null
  finding?: string
}) {
  const p = presentationFor(method)
  if (!p) return null
  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        gap: 'var(--s3)',
        padding: 'var(--s3)',
        borderRadius: 'var(--r-sm)',
        border: `1px solid ${p.border}`,
        background: p.background,
        fontSize: 'var(--fs-sm)',
      }}
    >
      {finding && (
        <span style={{ fontWeight: 700, color: 'var(--text)' }}>{finding}</span>
      )}
      {confidence != null && (
        <span style={{ color: 'var(--text-2)' }}>
          Confidence:{' '}
          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>
            {Math.round(confidence * 100)}%
          </span>
        </span>
      )}
      <span style={{ color: 'var(--text-2)' }}>
        Method:{' '}
        <span style={{ fontFamily: 'var(--font-mono)', color: p.colour, fontWeight: 700 }}>
          {p.label}
        </span>
      </span>
      {method === 'heuristic' && (
        <span
          style={{
            flexBasis: '100%',
            color: 'var(--text-2)',
            fontSize: 'var(--fs-tiny)',
            lineHeight: 1.5,
          }}
        >
          No trained PPE model is installed. This reading came from colour
          statistics over the head and torso regions — treat it as advisory and
          verify before acting. Install a PPE model to get model-grade results.
        </span>
      )}
    </div>
  )
}
