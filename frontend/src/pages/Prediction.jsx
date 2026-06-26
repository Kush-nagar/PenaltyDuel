import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate, Link } from 'react-router-dom'
import { predict } from '../api'

const ZONE_LABELS = {
  high_left: 'HIGH L',
  high_center: 'HIGH C',
  high_right: 'HIGH R',
  low_left: 'LOW L',
  low_center: 'LOW C',
  low_right: 'LOW R',
}

function ZoneCell({ zone, prob, isMax }) {
  const intensity = Math.min(1, prob * 4)
  return (
    <div
      title={`${ZONE_LABELS[zone]}: ${Math.round(prob * 100)}%`}
      style={{
        background: isMax
          ? `rgba(209,33,40,${0.4 + intensity * 0.5})`
          : `rgba(28,58,82,${0.3 + intensity * 0.5})`,
        border: isMax ? '2px solid #d12128' : '1px solid #1C3A52',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 8,
        position: 'relative',
      }}
    >
      <span style={{ fontFamily: '"VT323"', fontSize: 24, color: isMax ? '#F5B731' : '#F0EAD6', lineHeight: 1 }}>
        {Math.round(prob * 100)}%
      </span>
      <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 8, color: '#7A9BB5', letterSpacing: '0.1em', marginTop: 2 }}>
        {ZONE_LABELS[zone]}
      </span>
    </div>
  )
}

function GoalHeatmap({ zones }) {
  const maxProb = Math.max(...zones.map(z => z.prob))
  const topRow = zones.filter(z => z.zone.startsWith('high'))
  const botRow = zones.filter(z => z.zone.startsWith('low'))

  return (
    <div>
      <div style={{ border: '6px solid rgba(240,234,214,0.6)', borderBottom: 'none', padding: 4 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4, marginBottom: 4 }}>
          {topRow.map(z => <ZoneCell key={z.zone} zone={z.zone} prob={z.prob} isMax={z.prob === maxProb} />)}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4 }}>
          {botRow.map(z => <ZoneCell key={z.zone} zone={z.zone} prob={z.prob} isMax={z.prob === maxProb} />)}
        </div>
      </div>
      <div style={{ height: 16, background: '#1A5C34', borderLeft: '6px solid rgba(240,234,214,0.6)', borderRight: '6px solid rgba(240,234,214,0.6)' }} />
    </div>
  )
}

function DiveBars({ dives }) {
  const maxP = Math.max(...dives.map(d => d.prob))
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {dives.map(d => {
        const isMax = d.prob === maxP
        return (
          <div key={d.direction} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, width: 48, textAlign: 'right', color: '#7A9BB5', letterSpacing: '0.1em' }}>
              {d.direction.toUpperCase()}
            </span>
            <div style={{ flex: 1, height: 20, background: '#070C14', border: '2px solid #1C3A52', position: 'relative' }}>
              <div style={{
                height: '100%',
                width: `${d.prob * 100}%`,
                background: isMax ? '#F5B731' : '#1C3A52',
                transition: 'width 0.4s ease',
              }} />
            </div>
            <span style={{ fontFamily: '"VT323"', fontSize: 28, color: isMax ? '#F5B731' : '#7A9BB5', width: 44 }}>
              {Math.round(d.prob * 100)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

function ShapBar({ factor }) {
  const pct = Math.min(factor.pct_impact, 100)
  const isPos = factor.direction === 'positive'
  const isNeg = factor.direction === 'negative'
  const color = isPos ? '#3AB06A' : isNeg ? '#d12128' : '#7A9BB5'
  const sign = isPos ? '+' : isNeg ? '-' : '~'

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#F0EAD6', letterSpacing: '0.05em' }}>
          {factor.label.toUpperCase()}
        </span>
        <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color, fontWeight: 700 }}>
          {sign}{Math.abs(factor.shap_value * 100).toFixed(1)}pp
        </span>
      </div>
      <div style={{ height: 8, background: '#070C14', border: '1px solid #1C3A52' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, transition: 'width 0.4s' }} />
      </div>
    </div>
  )
}

const CONFIDENCE_COLORS = { high: '#3AB06A', medium: '#F5B731', low: '#d12128' }

export default function Prediction() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const shooterName = params.get('shooter') || ''
  const keeperName = params.get('keeper') || ''

  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!shooterName || !keeperName) { setError('Missing shooter or keeper'); setLoading(false); return }
    setLoading(true)
    predict(shooterName, keeperName)
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [shooterName, keeperName])

  if (loading) return (
    <div style={{ minHeight: '100svh', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 24 }}>
      <div style={{ fontFamily: '"Press Start 2P"', fontSize: 14, color: '#d12128' }} className="blink">
        LOADING...
      </div>
      <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#7A9BB5' }}>
        COMPUTING PREDICTION
      </div>
    </div>
  )

  if (error) return (
    <div style={{ minHeight: '100svh', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 24, padding: 24 }}>
      <div style={{ fontFamily: '"Press Start 2P"', fontSize: 14, color: '#d12128' }}>ERROR</div>
      <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 12, color: '#7A9BB5' }}>{error}</div>
      <button className="btn-primary" onClick={() => navigate('/')}>◀ BACK</button>
    </div>
  )

  const { shooter, keeper, zone_distribution, dive_distribution, shap_card, goal_probability, composed_probability } = data
  const goalPct = Math.round(goal_probability * 100)
  const confColor = CONFIDENCE_COLORS[shap_card.confidence]

  return (
    <div style={{ minHeight: '100svh', background: '#070C14' }}>
      {/* VS header */}
      <section style={{
        background: '#070C14',
        borderBottom: '4px solid #1C3A52',
        padding: '24px',
        marginTop: 80,
      }}>
        <div style={{
          maxWidth: 1100,
          margin: '0 auto',
          display: 'grid',
          gridTemplateColumns: '1fr auto 1fr',
          gap: 24,
          alignItems: 'center',
        }}>
          {/* Shooter card */}
          <div style={{
            background: '#0A2238',
            border: '4px solid #d12128',
            padding: 16,
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12, borderBottom: '2px solid #1C3A52', paddingBottom: 12, marginBottom: 12 }}>
              <div style={{ width: 56, height: 56, background: '#070C14', border: '2px solid #1C3A52', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className="material-symbols-outlined" style={{ fontSize: 32, color: '#7A9BB5' }}>person</span>
              </div>
              <div>
                <div style={{ fontFamily: '"Bebas Neue"', fontSize: 28, color: '#F0EAD6', lineHeight: 1 }}>{shooter.name.toUpperCase()}</div>
                <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', letterSpacing: '0.1em' }}>
                  STRIKER · {shooter.preferred_foot ? shooter.preferred_foot.toUpperCase() + ' FOOT' : 'FOOT N/A'}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#7A9BB5' }}>CONV RATE</span>
              <span style={{ fontFamily: '"VT323"', fontSize: 28, color: '#F0EAD6', lineHeight: 1 }}>{Math.round(shooter.conversion_rate * 100)}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#7A9BB5' }}>PENALTIES</span>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#F0EAD6' }}>{shooter.n_penalties}</span>
            </div>
          </div>

          {/* Goal probability */}
          <div style={{ textAlign: 'center', minWidth: 180 }}>
            <div style={{ background: '#0D1520', border: '4px solid #1C3A52', padding: '16px 24px', marginBottom: 12 }}>
              <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, letterSpacing: '0.14em', color: '#7A9BB5', marginBottom: 8 }}>GOAL PROBABILITY</div>
              <div style={{
                fontFamily: '"VT323"',
                fontSize: 80,
                color: '#F5B731',
                lineHeight: 1,
                textShadow: '0 0 20px rgba(245,183,49,0.6)',
              }}>
                {goalPct}%
              </div>
            </div>
            <div style={{
              fontFamily: '"IBM Plex Mono"',
              fontSize: 9,
              letterSpacing: '0.12em',
              color: confColor,
              border: `2px solid ${confColor}`,
              padding: '4px 12px',
              display: 'inline-block',
            }}>
              {shap_card.confidence.toUpperCase()} CONFIDENCE
            </div>
          </div>

          {/* Keeper card */}
          <div style={{
            background: '#0A2238',
            border: '4px solid #F5B731',
            padding: 16,
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12, borderBottom: '2px solid #1C3A52', paddingBottom: 12, marginBottom: 12, flexDirection: 'row-reverse', textAlign: 'right' }}>
              <div style={{ width: 56, height: 56, background: '#070C14', border: '2px solid #1C3A52', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className="material-symbols-outlined" style={{ fontSize: 32, color: '#7A9BB5' }}>sports_soccer</span>
              </div>
              <div>
                <div style={{ fontFamily: '"Bebas Neue"', fontSize: 28, color: '#F0EAD6', lineHeight: 1 }}>{keeper.name.toUpperCase()}</div>
                <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', letterSpacing: '0.1em' }}>
                  KEEPER · {keeper.height_cm ? `${keeper.height_cm}CM` : 'HT N/A'}
                </div>
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', flexDirection: 'row-reverse' }}>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#7A9BB5' }}>SAVE RATE</span>
              <span style={{ fontFamily: '"VT323"', fontSize: 28, color: '#F0EAD6', lineHeight: 1 }}>{Math.round(keeper.save_rate * 100)}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', flexDirection: 'row-reverse', marginTop: 4 }}>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#7A9BB5' }}>FACED</span>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#F0EAD6' }}>{keeper.n_faced}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Low data warning */}
      {shap_card.low_data_warning && (
        <div style={{
          background: 'rgba(209,33,40,0.1)',
          border: '2px solid #d12128',
          padding: '10px 24px',
          maxWidth: 1100,
          margin: '16px auto 0',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}>
          <span className="blink" style={{ fontFamily: '"Press Start 2P"', fontSize: 10, color: '#d12128' }}>⚠</span>
          <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#F0EAD6', letterSpacing: '0.08em' }}>
            {shap_card.low_data_note}
          </span>
        </div>
      )}

      {/* 3-panel analysis grid */}
      <section style={{ maxWidth: 1100, margin: '24px auto', padding: '0 24px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20 }}>

        {/* Panel 1: Shot placement heatmap */}
        <div style={{ background: '#0A2238', border: '4px solid #1C3A52', padding: 20 }}>
          <h3 style={{ fontFamily: '"Bebas Neue"', fontSize: 24, color: '#F0EAD6', marginBottom: 16, borderBottom: '2px solid #1C3A52', paddingBottom: 8, letterSpacing: '0.05em' }}>
            SHOT PLACEMENT
          </h3>
          <GoalHeatmap zones={zone_distribution} />
          <p style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', marginTop: 12, textAlign: 'center', letterSpacing: '0.1em' }}>
            FROM KEEPER'S POV · BRIGHTER = MORE LIKELY
          </p>
        </div>

        {/* Panel 2: Keeper dive tendency */}
        <div style={{ background: '#0A2238', border: '4px solid #1C3A52', padding: 20 }}>
          <h3 style={{ fontFamily: '"Bebas Neue"', fontSize: 24, color: '#F0EAD6', marginBottom: 16, borderBottom: '2px solid #1C3A52', paddingBottom: 8, letterSpacing: '0.05em' }}>
            DIVE TENDENCY
          </h3>
          <DiveBars dives={dive_distribution} />
          <div style={{ marginTop: 20, borderTop: '1px solid #1C3A52', paddingTop: 12 }}>
            <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', marginBottom: 4, letterSpacing: '0.1em' }}>COMPOSED P(GOAL)</div>
            <div style={{ fontFamily: '"VT323"', fontSize: 40, color: '#F5B731', lineHeight: 1 }}>
              {Math.round(composed_probability * 100)}%
            </div>
            <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, color: '#7A9BB5', marginTop: 4 }}>
              Zone × Dive decomposition
            </div>
          </div>
        </div>

        {/* Panel 3: SHAP terminal */}
        <div style={{ background: '#070C14', border: '4px solid #F0EAD6', padding: 20, boxShadow: '4px 4px 0 rgba(240,234,214,0.1)' }}>
          <h3 style={{ fontFamily: '"Bebas Neue"', fontSize: 24, color: '#F0EAD6', marginBottom: 4, borderBottom: '2px solid #1C3A52', paddingBottom: 8, letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 20 }}>terminal</span>
            ANALYSIS
          </h3>

          {/* Summaries */}
          <div style={{ marginBottom: 16 }}>
            <p style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', marginBottom: 8, lineHeight: 1.6 }}>
              <span style={{ color: '#d12128' }}>▶ SHOOTER: </span>{shap_card.shooter_summary}
            </p>
            <p style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', lineHeight: 1.6 }}>
              <span style={{ color: '#F5B731' }}>◀ KEEPER: </span>{shap_card.keeper_summary}
            </p>
          </div>

          <div style={{ borderTop: '1px solid #1C3A52', paddingTop: 12, marginBottom: 12 }}>
            <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, color: '#7A9BB5', letterSpacing: '0.1em', marginBottom: 10 }}>
              FACTOR ATTRIBUTION (SHAP)
            </div>
            {shap_card.factors.map(f => <ShapBar key={f.feature} factor={f} />)}
          </div>

          {shap_card.top_factors.length > 0 && (
            <div style={{ borderTop: '1px dashed #1C3A52', paddingTop: 10 }}>
              <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, color: '#7A9BB5', letterSpacing: '0.1em', marginBottom: 6 }}>
                TOP DRIVERS
              </div>
              {shap_card.top_factors.map((tf, i) => (
                <div key={i} style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#F0EAD6', marginBottom: 4 }}>
                  <span style={{ color: '#d12128' }}>{i + 1}.</span> {tf}
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Back button */}
      <div style={{ textAlign: 'center', padding: '24px 0 48px' }}>
        <Link to="/" style={{ textDecoration: 'none' }}>
          <button className="btn-primary">◀ NEW PREDICTION</button>
        </Link>
      </div>
    </div>
  )
}
