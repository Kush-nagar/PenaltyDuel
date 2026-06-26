import { useState, useEffect, useRef } from 'react'

export default function PlayerSearch({ label, fetchFn, onSelect, selected }) {
  const [query, setQuery] = useState(selected?.name || '')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const debounceRef = useRef(null)
  const wrapperRef = useRef(null)

  useEffect(() => {
    if (selected) setQuery(selected.name)
  }, [selected])

  useEffect(() => {
    clearTimeout(debounceRef.current)
    if (!query || query === selected?.name) { setResults([]); setOpen(false); return }
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const data = await fetchFn(query)
        setResults(data.slice(0, 8))
        setOpen(true)
      } catch { setResults([]) }
      setLoading(false)
    }, 250)
    return () => clearTimeout(debounceRef.current)
  }, [query])

  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  function handleSelect(p) {
    setQuery(p.name)
    setOpen(false)
    setResults([])
    onSelect(p)
  }

  return (
    <div ref={wrapperRef} style={{ position: 'relative' }}>
      <label style={{
        display: 'block',
        fontFamily: '"IBM Plex Mono"',
        fontSize: 10,
        letterSpacing: '0.14em',
        color: '#7A9BB5',
        marginBottom: 6,
        fontWeight: 700,
      }}>
        {label}
      </label>
      <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
        <span className="material-symbols-outlined" style={{
          position: 'absolute', left: 10, fontSize: 18, color: '#7A9BB5', pointerEvents: 'none'
        }}>search</span>
        <input
          className="terminal-input"
          style={{ paddingLeft: 36 }}
          placeholder={`SEARCH ${label.toUpperCase()}...`}
          value={query}
          onChange={e => { setQuery(e.target.value); if (!e.target.value) { onSelect(null) } }}
          onFocus={() => { if (results.length) setOpen(true) }}
        />
        {loading && (
          <span className="material-symbols-outlined blink" style={{
            position: 'absolute', right: 10, fontSize: 16, color: '#d12128'
          }}>hourglass_empty</span>
        )}
      </div>
      {open && results.length > 0 && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          background: '#0A2238',
          border: '2px solid #d12128',
          zIndex: 100,
          maxHeight: 240,
          overflowY: 'auto',
        }}>
          {results.map((p, i) => {
            const isShooter = 'n_penalties' in p
            const countKey = isShooter ? 'n_penalties' : 'n_faced'
            const countLabel = isShooter ? 'PK' : 'FACED'
            return (
              <button
                key={p.id || i}
                onClick={() => handleSelect(p)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  width: '100%',
                  padding: '8px 12px',
                  background: 'transparent',
                  border: 'none',
                  borderBottom: '1px solid #1C3A52',
                  color: '#F0EAD6',
                  fontFamily: '"IBM Plex Mono"',
                  fontSize: 12,
                  cursor: 'pointer',
                  textAlign: 'left',
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#0D1520'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <span>{p.name}</span>
                <span style={{ color: '#7A9BB5', fontSize: 10 }}>
                  {p[countKey]} {countLabel}
                </span>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
