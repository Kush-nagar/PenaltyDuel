import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getShooters, getKeepers } from '../api'

function PlayerRow({ player, type, onClick }) {
  const countKey = type === 'shooter' ? 'n_penalties' : 'n_faced'
  const countLabel = type === 'shooter' ? 'PENALTIES' : 'FACED'
  const rateKey = type === 'shooter' ? 'conv_rate_shrunk' : 'save_rate_shrunk'

  return (
    <button
      onClick={() => onClick(player)}
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr auto auto',
        gap: 16,
        alignItems: 'center',
        width: '100%',
        padding: '10px 16px',
        background: 'transparent',
        border: 'none',
        borderBottom: '1px solid #1C3A52',
        color: '#F0EAD6',
        textAlign: 'left',
        cursor: 'pointer',
        transition: 'background 0.1s',
      }}
      onMouseEnter={e => e.currentTarget.style.background = '#0A2238'}
      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
    >
      <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 13, fontWeight: 600 }}>{player.name}</span>
      <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, color: '#7A9BB5', textAlign: 'right' }}>
        {player[countKey]} {countLabel}
      </span>
      <span style={{ fontFamily: '"VT323"', fontSize: 28, color: '#F5B731', width: 50, textAlign: 'right', lineHeight: 1 }}>
        {player[rateKey] !== undefined ? `${Math.round(player[rateKey] * 100)}%` : '—'}
      </span>
    </button>
  )
}

export default function Players() {
  const [tab, setTab] = useState('shooter')
  const [query, setQuery] = useState('')
  const [players, setPlayers] = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  async function load(q = '') {
    setLoading(true)
    try {
      const data = tab === 'shooter' ? await getShooters(q) : await getKeepers(q)
      setPlayers(data)
    } catch { setPlayers([]) }
    setLoading(false)
  }

  useEffect(() => { load() }, [tab])

  useEffect(() => {
    const t = setTimeout(() => load(query), 300)
    return () => clearTimeout(t)
  }, [query])

  function handlePlayerClick(player) {
    if (tab === 'shooter') {
      navigate(`/predict?shooter=${encodeURIComponent(player.name)}&keeper=`)
    } else {
      navigate(`/?keeper=${encodeURIComponent(player.name)}`)
    }
  }

  const isShooter = tab === 'shooter'

  return (
    <div style={{ minHeight: '100svh', background: '#070C14', paddingTop: 80 }}>
      {/* Header */}
      <section style={{ borderBottom: '4px solid #1C3A52', padding: '24px' }}>
        <div style={{ maxWidth: 900, margin: '0 auto' }}>
          <h1 style={{ fontFamily: '"Bebas Neue"', fontSize: 48, color: '#F0EAD6', letterSpacing: '0.05em', marginBottom: 4 }}>
            PLAYER EXPLORER
          </h1>
          <p style={{ fontFamily: '"IBM Plex Mono"', fontSize: 11, color: '#7A9BB5', letterSpacing: '0.1em' }}>
            BROWSE 772 SHOOTERS + 379 KEEPERS FROM THE STATSBOMB DATABASE
          </p>
        </div>
      </section>

      <div style={{ maxWidth: 900, margin: '0 auto', padding: '24px' }}>
        {/* Tab switcher */}
        <div style={{ display: 'flex', marginBottom: 24, borderBottom: '2px solid #1C3A52' }}>
          {[['shooter', 'SHOOTERS (772)'], ['keeper', 'KEEPERS (379)']].map(([val, label]) => (
            <button
              key={val}
              onClick={() => { setTab(val); setQuery('') }}
              style={{
                fontFamily: '"IBM Plex Mono"',
                fontSize: 11,
                letterSpacing: '0.12em',
                fontWeight: 700,
                padding: '10px 24px',
                background: tab === val ? '#d12128' : 'transparent',
                color: tab === val ? '#F0EAD6' : '#7A9BB5',
                border: 'none',
                borderBottom: tab === val ? '2px solid #d12128' : '2px solid transparent',
                marginBottom: -2,
                cursor: 'pointer',
                transition: 'all 0.1s',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Search */}
        <div style={{ position: 'relative', marginBottom: 20 }}>
          <span className="material-symbols-outlined" style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', fontSize: 18, color: '#7A9BB5', pointerEvents: 'none' }}>search</span>
          <input
            className="terminal-input"
            style={{ paddingLeft: 36 }}
            placeholder={`SEARCH ${isShooter ? 'SHOOTERS' : 'KEEPERS'}...`}
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>

        {/* Column headers */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr auto auto',
          gap: 16,
          padding: '6px 16px',
          borderBottom: '2px solid #d12128',
          marginBottom: 4,
        }}>
          <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, letterSpacing: '0.14em', color: '#7A9BB5' }}>PLAYER</span>
          <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, letterSpacing: '0.14em', color: '#7A9BB5' }}>{isShooter ? 'PENALTIES' : 'FACED'}</span>
          <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, letterSpacing: '0.14em', color: '#7A9BB5' }}>{isShooter ? 'CONV %' : 'SAVE %'}</span>
        </div>

        {/* Player list */}
        <div style={{ background: '#0A2238', border: '2px solid #1C3A52', minHeight: 400 }}>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
              <span style={{ fontFamily: '"Press Start 2P"', fontSize: 12, color: '#d12128' }} className="blink">LOADING...</span>
            </div>
          ) : players.length === 0 ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
              <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 12, color: '#7A9BB5' }}>NO RESULTS FOUND</span>
            </div>
          ) : (
            players.map((p, i) => (
              <PlayerRow key={p.id || i} player={p} type={tab} onClick={handlePlayerClick} />
            ))
          )}
        </div>

        <p style={{ fontFamily: '"IBM Plex Mono"', fontSize: 9, color: '#7A9BB5', marginTop: 12, letterSpacing: '0.1em' }}>
          {players.length} RESULTS · CLICK PLAYER TO START PREDICTION
        </p>
      </div>
    </div>
  )
}
