const BASE = '/api'

export async function getShooters(q = '') {
  const url = q ? `${BASE}/players/shooters?q=${encodeURIComponent(q)}` : `${BASE}/players/shooters`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`Shooters fetch failed: ${r.status}`)
  return r.json()
}

export async function getKeepers(q = '') {
  const url = q ? `${BASE}/players/keepers?q=${encodeURIComponent(q)}` : `${BASE}/players/keepers`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`Keepers fetch failed: ${r.status}`)
  return r.json()
}

export async function predict(shooterName, keeperName) {
  const r = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ shooter_name: shooterName, keeper_name: keeperName }),
  })
  if (!r.ok) {
    const err = await r.json().catch(() => ({}))
    throw new Error(err.detail || `Predict failed: ${r.status}`)
  }
  return r.json()
}

export async function healthCheck() {
  const r = await fetch(`${BASE}/health`)
  return r.json()
}
