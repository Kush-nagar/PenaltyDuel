import { Link, useLocation } from 'react-router-dom'

const LINKS = [
  { to: '/', label: 'HOME' },
  { to: '/predict', label: 'PREDICT' },
  { to: '/players', label: 'PLAYERS' },
]

export default function TopNav() {
  const { pathname } = useLocation()

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 flex justify-between items-center px-6 py-4"
      style={{ background: '#070C14', borderBottom: '4px solid #1C3A52' }}
    >
      <div className="flex items-center gap-4">
        <div className="flex flex-col items-start leading-tight">
          <span style={{ fontFamily: '"IBM Plex Mono"', fontSize: 10, letterSpacing: '0.14em', color: '#d12128', fontWeight: 700 }}>
            STATSBOMB
          </span>
          <span style={{ fontFamily: '"Press Start 2P"', fontSize: 14, color: '#F0EAD6', lineHeight: 1.4 }}>
            PENALTY
          </span>
          <span style={{ fontFamily: '"Press Start 2P"', fontSize: 14, color: '#F0EAD6', lineHeight: 1.4 }}>
            PREDICTOR
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        {LINKS.map(({ to, label }) => {
          const active = pathname === to || (to !== '/' && pathname.startsWith(to))
          return (
            <Link
              key={to}
              to={to}
              style={{
                fontFamily: '"IBM Plex Mono"',
                fontSize: 11,
                fontWeight: active ? 700 : 400,
                letterSpacing: '0.14em',
                color: active ? '#F0EAD6' : '#7A9BB5',
                textDecoration: 'none',
                borderBottom: active ? '2px solid #d12128' : '2px solid transparent',
                paddingBottom: 2,
                transition: 'color 0.1s',
              }}
            >
              {label}
            </Link>
          )
        })}
        <span style={{
          fontFamily: '"IBM Plex Mono"',
          fontSize: 10,
          letterSpacing: '0.12em',
          color: '#7A9BB5',
          border: '1px solid #1C3A52',
          padding: '2px 8px',
        }}>
          StatsBomb + Kaggle
        </span>
      </div>
    </nav>
  )
}
