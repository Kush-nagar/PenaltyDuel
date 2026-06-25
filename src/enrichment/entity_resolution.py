"""
Entity resolution: link StatsBomb players to FBref and Transfermarkt identities.

The core problem (see the enrichment spec): StatsBomb, FBref and Transfermarkt
each use their own ID namespaces that never overlap.  The only shared signal is
the player's name, and names are inconsistent (diacritics, abbreviations,
spelling variants).  So we:

  1. Normalise every name with the SAME ``normalize_name`` function.
  2. Match on the normalised name as the FIRST signal only.
  3. NEVER auto-accept a name-only match — require a SECOND corroborating
     signal (nationality match, team-season overlap, or position consistency)
     before granting "high"/"medium" confidence.
  4. Fall back to ``rapidfuzz.token_sort_ratio`` (threshold 90) only to
     *propose* fuzzy matches — these are flagged ``needs_review`` and never
     auto-accepted.

Public API (called by pipelines/enrich_players.py):
    normalize_name(name)
    nationality_matches(a, b)
    fuzzy_name_match(query, candidates, threshold)
    resolve_player(...)            -> ResolvedMatch
    build_player_pool(...)         -> pd.DataFrame
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

# Threshold for fuzzy name matching (token_sort_ratio).  Matches below this are
# not even proposed.  Matches at/above this are proposed but ALWAYS flagged for
# human review — never auto-accepted.
FUZZY_THRESHOLD = 90

# Confidence ordering used when collapsing several candidate signals into the
# single best match for a player.
_CONFIDENCE_RANK = {"unresolved": 0, "low": 1, "medium": 2, "high": 3}

# A tiny, deliberately conservative country-name -> alias set so that StatsBomb
# full names ("England") line up with the variants the external sources emit.
# Only used to make ``nationality_matches`` a touch more forgiving; missing
# entries simply fall through to the normalised-string / fuzzy comparison.
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "england": {"england", "eng", "english"},
    "scotland": {"scotland", "sco", "scottish"},
    "wales": {"wales", "wal", "welsh"},
    "northern ireland": {"northern ireland", "nir"},
    "republic of ireland": {"republic of ireland", "ireland", "irl"},
    "united states": {"united states", "usa", "united states of america", "us"},
    "south korea": {"south korea", "korea republic", "kor"},
    "ivory coast": {"ivory coast", "cote d'ivoire", "civ"},
    "netherlands": {"netherlands", "holland", "ned", "nld"},
    "germany": {"germany", "ger", "deu"},
    "spain": {"spain", "esp", "spanish"},
    "france": {"france", "fra", "french"},
    "brazil": {"brazil", "bra", "brasil"},
    "portugal": {"portugal", "por", "prt"},
    "argentina": {"argentina", "arg"},
}


@dataclass
class ResolvedMatch:
    """The outcome of trying to link one StatsBomb player to one source."""

    fbref_player_id: Optional[str] = None
    tm_player_url: Optional[str] = None
    match_method: str = "unresolved"
    match_confidence: str = "unresolved"
    notes: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        # Per spec: fuzzy proposals and total misses go to the human queue.
        return self.match_method in {"fuzzy_proposed", "unresolved"}

    @property
    def is_usable(self) -> bool:
        # Only high/medium confidence matches feed the attribute tables.
        return self.match_confidence in {"high", "medium"}


def normalize_name(name: Optional[str]) -> Optional[str]:
    """Lowercase, strip diacritics with unidecode, collapse whitespace.

    This MUST be the only name-normalisation used across the whole enrichment
    pipeline so that join keys line up across StatsBomb / FBref / Transfermarkt.
    Mirrors ``src.ingestion.statsbomb._normalize_name`` exactly.
    """
    if not name:
        return None
    return re.sub(r"\s+", " ", unidecode(str(name)).lower().strip())


def _nationality_tokens(value: Optional[str]) -> set[str]:
    """Reduce a nationality string to a comparable token set.

    Handles FBref's ``"eng ENG"`` style, plain country names, and small alias
    groups.  Returns an empty set for missing values so callers treat "unknown"
    as "no signal" rather than "mismatch".
    """
    norm = normalize_name(value)
    if not norm:
        return set()
    tokens = {t for t in norm.split() if t}
    tokens.add(norm)
    # Fold in any known aliases (keyed by the full normalised string and by
    # each individual token).
    for key, aliases in _COUNTRY_ALIASES.items():
        if norm in aliases or norm == key or tokens & aliases:
            tokens |= aliases
            tokens.add(key)
    return tokens


def nationality_matches(a: Optional[str], b: Optional[str]) -> bool:
    """True if two nationality strings plausibly refer to the same country.

    Conservative: returns False when either side is missing (no signal != match).
    """
    ta, tb = _nationality_tokens(a), _nationality_tokens(b)
    if not ta or not tb:
        return False
    if ta & tb:
        return True
    # Last resort: fuzzy on the full normalised strings (handles spelling drift).
    na, nb = normalize_name(a), normalize_name(b)
    return bool(na and nb and fuzz.token_sort_ratio(na, nb) >= 90)


def teams_overlap(
    statsbomb_teams: Iterable[Optional[str]],
    source_team: Optional[str],
) -> bool:
    """True if a source squad name matches any StatsBomb team for the player."""
    src = normalize_name(source_team)
    if not src:
        return False
    sb = {normalize_name(t) for t in statsbomb_teams if t}
    sb.discard(None)
    if src in sb:
        return True
    # Allow fuzzy club-name drift (e.g. "Manchester Utd" vs "Manchester United").
    return any(fuzz.token_sort_ratio(src, t) >= 90 for t in sb if t)


def fuzzy_name_match(
    query: Optional[str],
    candidates: Iterable[str],
    threshold: int = FUZZY_THRESHOLD,
) -> Optional[tuple[str, float]]:
    """Return (best_candidate, score) at/above ``threshold`` or None.

    Uses ``token_sort_ratio`` per spec.  Only ever used to *propose* a match.
    """
    q = normalize_name(query)
    cand = [c for c in candidates if c]
    if not q or not cand:
        return None
    best = process.extractOne(q, cand, scorer=fuzz.token_sort_ratio)
    if best is None:
        return None
    name, score, _ = best
    if score >= threshold:
        return name, float(score)
    return None


def better_of(a: ResolvedMatch, b: ResolvedMatch) -> ResolvedMatch:
    """Pick the higher-confidence of two matches (ties keep ``a``)."""
    return a if _CONFIDENCE_RANK[a.match_confidence] >= _CONFIDENCE_RANK[b.match_confidence] else b


def build_player_pool(
    lineup_ref: pd.DataFrame,
    penalties: pd.DataFrame,
) -> pd.DataFrame:
    """Build the deduplicated scrape/resolution queue (one row per unique player).

    The queue is scoped to the players we actually enrich — the SHOOTERS and
    KEEPERS that appear in the penalty table — not every player in every
    penalty-match lineup (that would be thousands of needless Transfermarkt
    scrapes).  The lineup reference is still used, but only to *enrich* those
    target players with extra nationality / team / GK signals for resolution.

    Returns columns:
        player_name_normalized, statsbomb_player_id, player_name,
        nationality, is_goalkeeper, teams (set[str])
    """
    rows: dict[str, dict] = {}

    target_names = set(penalties["shooter_name_normalized"].dropna()) | set(
        penalties["keeper_name_normalized"].dropna()
    )

    def _ingest(name_norm, pid, name, nat, is_gk, team):
        if not name_norm or name_norm not in target_names:
            return
        rec = rows.setdefault(
            name_norm,
            {
                "player_name_normalized": name_norm,
                "statsbomb_player_id": pid,
                "player_name": name,
                "nationality": nat,
                "is_goalkeeper": bool(is_gk),
                "teams": set(),
            },
        )
        # Keep the first non-null id/name/nationality we saw; OR the GK flag.
        if pd.isna(rec["statsbomb_player_id"]) and pid is not None:
            rec["statsbomb_player_id"] = pid
        if not rec["nationality"] and nat:
            rec["nationality"] = nat
        rec["is_goalkeeper"] = rec["is_goalkeeper"] or bool(is_gk)
        if team:
            rec["teams"].add(team)

    for _, r in lineup_ref.iterrows():
        _ingest(
            r.get("player_name_normalized"),
            r.get("player_id"),
            r.get("player_name"),
            r.get("country_name"),
            r.get("played_goalkeeper", False),
            r.get("team_name"),
        )

    # Shooters from the penalty table (covers any shooter missing from lineups).
    for _, r in penalties.iterrows():
        _ingest(
            r.get("shooter_name_normalized"),
            r.get("shooter_id"),
            r.get("shooter_name"),
            None,
            False,
            r.get("shooter_team_name"),
        )
        _ingest(
            r.get("keeper_name_normalized"),
            r.get("keeper_id"),
            r.get("keeper_name"),
            r.get("keeper_nationality"),
            True,
            None,
        )

    pool = pd.DataFrame(list(rows.values()))
    if not pool.empty:
        pool["statsbomb_player_id"] = pool["statsbomb_player_id"].astype("Int64")
    return pool.reset_index(drop=True)


def resolve_fbref(
    player: pd.Series,
    fbref_index: dict[str, list[dict]],
) -> ResolvedMatch:
    """Resolve one StatsBomb player against the FBref player index.

    ``fbref_index`` maps normalised name -> list of {fbref_player_id, squads,
    nation, is_goalkeeper}.  We accept an FBref id only with a SECOND signal:
    team-season overlap (preferred) or nationality match.  A goalkeeper is never
    matched to an outfield FBref entry.
    """
    name = player["player_name_normalized"]
    sb_teams = player.get("teams") or set()
    sb_nat = player.get("nationality")
    is_gk = bool(player.get("is_goalkeeper", False))

    candidates = fbref_index.get(name, [])
    # Position guardrail: drop GK<->outfield cross matches when FBref tells us.
    candidates = [
        c for c in candidates
        if c.get("is_goalkeeper") is None or c["is_goalkeeper"] == is_gk
    ]
    if not candidates:
        # Try a fuzzy proposal (never auto-accepted).
        hit = fuzzy_name_match(name, list(fbref_index.keys()))
        if hit:
            cand_name, score = hit
            fb = fbref_index[cand_name][0]
            return ResolvedMatch(
                fbref_player_id=fb.get("fbref_player_id"),
                match_method="fuzzy_proposed",
                match_confidence="low",
                notes=[f"fbref fuzzy '{cand_name}' score={score:.0f}"],
            )
        return ResolvedMatch()

    for c in candidates:
        if any(teams_overlap(sb_teams, sq) for sq in c.get("squads", [])):
            return ResolvedMatch(
                fbref_player_id=c.get("fbref_player_id"),
                match_method="exact_name_plus_team_season",
                match_confidence="high",
                notes=["fbref team-season overlap"],
            )
    for c in candidates:
        if nationality_matches(sb_nat, c.get("nation")):
            return ResolvedMatch(
                fbref_player_id=c.get("fbref_player_id"),
                match_method="exact_name_plus_nationality",
                match_confidence="high",
                notes=["fbref nationality match"],
            )
    # Exact normalised name but no corroboration -> low, not usable.
    return ResolvedMatch(
        fbref_player_id=candidates[0].get("fbref_player_id"),
        match_method="exact_name",
        match_confidence="low",
        notes=["fbref name-only (no second signal)"],
    )


def resolve_transfermarkt(
    player: pd.Series,
    tm_candidates: list[dict],
) -> ResolvedMatch:
    """Resolve one StatsBomb player against Transfermarkt search candidates.

    ``tm_candidates`` is the parsed output for this player's name search, each a
    dict: {name, url, tm_id, nationality, position_is_goalkeeper, ...}.  TM search
    is name-based, so a hit is at best a name match; we still require a SECOND
    signal (nationality, or GK-position consistency) for high/medium confidence.
    """
    name = player["player_name_normalized"]
    sb_nat = player.get("nationality")
    is_gk = bool(player.get("is_goalkeeper", False))

    if not tm_candidates:
        return ResolvedMatch()

    exact = [c for c in tm_candidates if normalize_name(c.get("name")) == name]
    pool = exact or tm_candidates

    # 1) name + nationality (strongest available from TM).
    for c in pool:
        if nationality_matches(sb_nat, c.get("nationality")):
            return ResolvedMatch(
                tm_player_url=c.get("url"),
                match_method="exact_name_plus_nationality",
                match_confidence="high",
                notes=["tm nationality match"],
            )

    # 2) exact name + GK-position consistency -> medium (weak but corroborating).
    for c in exact:
        gk = c.get("position_is_goalkeeper")
        if gk is not None and gk == is_gk:
            return ResolvedMatch(
                tm_player_url=c.get("url"),
                match_method="exact_name_plus_nationality",
                match_confidence="medium",
                notes=["tm exact name + position consistency"],
            )

    # 3) exact normalised name, no corroboration -> low (not usable).
    if exact:
        return ResolvedMatch(
            tm_player_url=exact[0].get("url"),
            match_method="exact_name",
            match_confidence="low",
            notes=["tm name-only (no second signal)"],
        )

    # 4) only fuzzy name hits -> propose for review.
    hit = fuzzy_name_match(name, [normalize_name(c.get("name")) for c in pool])
    if hit:
        cand_name, score = hit
        match = next(
            (c for c in pool if normalize_name(c.get("name")) == cand_name),
            pool[0],
        )
        return ResolvedMatch(
            tm_player_url=match.get("url"),
            match_method="fuzzy_proposed",
            match_confidence="low",
            notes=[f"tm fuzzy '{cand_name}' score={score:.0f}"],
        )

    return ResolvedMatch()
