import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PlayerSearch from '../components/PlayerSearch'
import { getShooters, getKeepers } from '../api'

const TICKER_ITEMS = [
  'MESSI — 0.82 CONV',
  'RONALDO — 0.85 CONV',
  'BUFFON — 0.22 SAVE',
  'PENALTY PREDICTOR v0.1',
  '1477 PENALTIES ANALYSED',
  'STATSBOMB OPEN DATA',
  'KAGGLE DIVE DATASET',
  '772 SHOOTERS · 379 KEEPERS',
  '21 COMPETITIONS',
]

export default function Home() {
  const [shooter, setShooter] = useState(null)
  const [keeper, setKeeper] = useState(null)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    if (!shooter || !keeper) {
      setError('SELECT BOTH SHOOTER AND KEEPER')
      return
    }
    navigate(`/predict?shooter=${encodeURIComponent(shooter.name)}&keeper=${encodeURIComponent(keeper.name)}`)
  }

  return (
    <div style={{ minHeight: '100svh', background: '#070C14' }}>
      {/* Ticker bar */}
      <div className="ticker-wrapper" style={{ marginTop: 80, padding: '8px 0' }}>
        <div className="ticker-content" style={{
          fontFamily: '"Press Start 2P"',
          fontSize: 10,
          color: '#F5B731',
          letterSpacing: '0.1em',
        }}>
          {Array(3).fill(TICKER_ITEMS).flat().map((item, i) => (
            <span key={i} style={{ marginRight: 48 }}>
              <span style={{ color: '#d12128' }}>▶ </span>{item}
            </span>
          ))}
        </div>
      </div>

      {/* Hero section */}
      <section style={{
        width: '100%',
        minHeight: 420,
        background: '#070C14',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        borderBottom: '4px solid #1C3A52',
        overflow: 'hidden',
        padding: '48px 0 0',
      }}>
        {/* Pixel football pitch */}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0, height: 120,
          background: '#1A5C34',
          boxShadow: 'inset 0 20px 40px rgba(0,0,0,0.5)',
        }}>
          <div style={{
            position: 'absolute', inset: 0, opacity: 0.15,
            backgroundImage: 'radial-gradient(#F0EAD6 1px, transparent 1px)',
            backgroundSize: '8px 8px',
          }} />
        </div>

        {/* Goal frame */}
        <div style={{
          position: 'absolute', bottom: 80, left: '50%', transform: 'translateX(-50%)',
          width: 280, height: 140,
          border: '6px solid rgba(240,234,214,0.85)',
          display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)',
          gridTemplateRows: 'repeat(3, 1fr)',
          gap: 4, padding: 4,
          background: 'rgba(13,21,32,0.6)',
          zIndex: 5,
        }}>
          {Array(18).fill(0).map((_, i) => (
            <div key={i} style={{ border: '1px solid rgba(255,255,255,0.15)' }} />
          ))}
        </div>

        {/* Hero text */}
        <div style={{ position: 'relative', zIndex: 10, textAlign: 'center', padding: '0 24px 160px' }}>
          <div className="pixel-shadow-text" style={{
            fontFamily: '"Press Start 2P"',
            fontSize: 'clamp(16px, 3vw, 28px)',
            color: '#F0EAD6',
            lineHeight: 1.6,
            marginBottom: 16,
          }}>
            PENALTY
          </div>
          <div className="pixel-shadow-text" style={{
            fontFamily: '"Press Start 2P"',
            fontSize: 'clamp(16px, 3vw, 28px)',
            color: '#d12128',
            lineHeight: 1.6,
            marginBottom: 24,
          }}>
            PREDICTOR
          </div>
          <p style={{
            fontFamily: '"IBM Plex Mono"',
            fontSize: 12,
            color: '#7A9BB5',
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
          }}>
            Hierarchical Explainable Penalty Outcome Model
          </p>
        </div>
      </section>

      {/* Pitch divider */}
      <div className="pitch-divider" />

      {/* Matchup builder */}
      <section style={{ maxWidth: 900, margin: '0 auto', padding: '0 24px 64px' }}>
        <h2 style={{
          fontFamily: '"Bebas Neue"',
          fontSize: 40,
          color: '#F0EAD6',
          marginBottom: 8,
          letterSpacing: '0.05em',
        }}>
          BUILD YOUR MATCHUP
        </h2>
        <p style={{
          fontFamily: '"IBM Plex Mono"',
          fontSize: 12,
          color: '#7A9BB5',
          marginBottom: 32,
          letterSpacing: '0.1em',
        }}>
          SELECT A SHOOTER AND A GOALKEEPER — WE'LL DO THE REST
        </p>

        <form onSubmit={handleSubmit}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr auto 1fr',
            gap: 24,
            alignItems: 'start',
            marginBottom: 32,
          }}>
            {/* Shooter panel */}
            <div className="pixel-card" style={{ padding: 24 }}>
              <div style={{
                fontFamily: '"IBM Plex Mono"',
                fontSize: 10,
                letterSpacing: '0.14em',
                color: '#d12128',
                fontWeight: 700,
                marginBottom: 16,
              }}>
                ▶ SHOOTER
              </div>
              <PlayerSearch
                label="Shooter"
                fetchFn={getShooters}
                selected={shooter}
                onSelect={setShooter}
              />
              {shooter && (
                <div style={{ marginTop: 16, borderTop: '1px solid #1C3A52', paddingTop: 12 }}>
                  <div style={{ fontFamily: '"Bebas Neue"', fontSize: 28, color: '#F0EAD6', letterSpacing: '0.05em' }}>
                    {shooter.name.toUpperCase()}
                  </div>
                  <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', marginTop: 4 }}>
                    {shooter.n_penalties ?? '?'} PENALTIES RECORDED
                  </div>
                </div>
              )}
            </div>

            {/* VS */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '40px 8px 0',
            }}>
              <span style={{
                fontFamily: '"Press Start 2P"',
                fontSize: 20,
                color: '#F5B731',
                textShadow: '2px 2px 0 #d12128',
              }}>VS</span>
            </div>

            {/* Keeper panel */}
            <div className="pixel-card" style={{ padding: 24 }}>
              <div style={{
                fontFamily: '"IBM Plex Mono"',
                fontSize: 10,
                letterSpacing: '0.14em',
                color: '#F5B731',
                fontWeight: 700,
                marginBottom: 16,
              }}>
                ◀ KEEPER
              </div>
              <PlayerSearch
                label="Keeper"
                fetchFn={getKeepers}
                selected={keeper}
                onSelect={setKeeper}
              />
              {keeper && (
                <div style={{ marginTop: 16, borderTop: '1px solid #1C3A52', paddingTop: 12 }}>
                  <div style={{ fontFamily: '"Bebas Neue"', fontSize: 28, color: '#F0EAD6', letterSpacing: '0.05em' }}>
                    {keeper.name.toUpperCase()}
                  </div>
                  <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', marginTop: 4 }}>
                    {keeper.n_faced ?? '?'} PENALTIES FACED
                  </div>
                </div>
              )}
            </div>
          </div>

          {error && (
            <p style={{
              fontFamily: '"IBM Plex Mono"',
              fontSize: 12,
              color: '#d12128',
              marginBottom: 16,
              textAlign: 'center',
              letterSpacing: '0.1em',
            }}>
              ⚠ {error}
            </p>
          )}

          <div style={{ textAlign: 'center' }}>
            <button
              type="submit"
              className="btn-primary"
              style={{ fontSize: 14, padding: '16px 48px', letterSpacing: '0.08em' }}
              disabled={!shooter || !keeper}
            >
              ▶ PREDICT OUTCOME
            </button>
          </div>
        </form>

        {/* Stats bar */}
        <div style={{
          marginTop: 48,
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 16,
          borderTop: '2px solid #1C3A52',
          paddingTop: 32,
        }}>
          {[
            ['1,477', 'PENALTIES'],
            ['772', 'SHOOTERS'],
            ['379', 'KEEPERS'],
            ['73.9%', 'AVG CONV RATE'],
          ].map(([val, lab]) => (
            <div key={lab} style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: '"VT323"', fontSize: 48, color: '#F5B731', lineHeight: 1 }}>{val}</div>
              <div style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, color: '#7A9BB5', letterSpacing: '0.14em', marginTop: 4 }}>{lab}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
