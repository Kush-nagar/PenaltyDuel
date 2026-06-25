"""
Pipeline: enrich StatsBomb penalties with FBref + Transfermarkt player attributes.

Run from the project root:
    python pipelines/enrich_players.py

Optional flags (for smoke-testing / partial runs — the default is a full run):
    --limit-players N   Only resolve the first N unique players (others -> unresolved)
    --skip-fbref        Skip the FBref (SoccerData) pull
    --skip-tm           Skip the Transfermarkt (ScraperFC) scrape
    --lookback N        Seasons of FBref history before the earliest (default 3)

Steps:
  1  Load the StatsBomb penalty table + player lineup reference
  2  Build the unique player scrape queue (deduped names + nationalities)
  3  Pull FBref career penalty stats via SoccerData (all relevant seasons)
  4  Pull player attributes from Transfermarkt via ScraperFC (cache-first)
  5  Entity resolution -> name_map.csv + unresolved.csv
  6  Build player_attributes.parquet
  7  Build player_career_penalty_stats.parquet
  8  Join everything back to the penalty table -> penalties_enriched.parquet
  9  Print coverage / match-rate / unresolved summary

NOTE: first full run can take 30-60 min (mostly Transfermarkt rate limiting).
Every Transfermarkt response is cached, so re-runs are fast and resumable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# UTF-8 on Windows (matches build_dataset.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.enrichment.entity_resolution import (  # noqa: E402
    ResolvedMatch,
    better_of,
    build_player_pool,
    resolve_fbref,
    resolve_transfermarkt,
)
from src.ingestion.fbref import (  # noqa: E402
    build_fbref_index,
    fetch_fbref_career_penalty_stats,
    fetch_fbref_foot,
    statsbomb_seasons_to_fbref,
)
from src.ingestion.transfermarkt import TransfermarktClient  # noqa: E402

ENRICHMENT_VERSION = "fbref_tm_v1"

STATSBOMB_DIR = ROOT / "outputs" / "statsbomb"
PENALTIES_FILE = STATSBOMB_DIR / "penalties_statsbomb_clean.parquet"
LINEUPS_FILE = STATSBOMB_DIR / "lineups_statsbomb_penalty_matches.parquet"

ENRICH_DIR = ROOT / "outputs" / "enrichment"
CACHE_FBREF_DIR = ENRICH_DIR / "cache" / "fbref"
CACHE_TM_DIR = ENRICH_DIR / "cache" / "tm"
RESOLUTION_DIR = ENRICH_DIR / "entity_resolution"

SEP = "=" * 70


def _ensure_dirs() -> None:
    for d in (CACHE_FBREF_DIR, CACHE_TM_DIR, RESOLUTION_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Step 1 ──────────────────────────────────────────────────────────────────────

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(SEP)
    print("STEP 1 — Loading StatsBomb penalty table + lineup reference")
    print(SEP)
    if not PENALTIES_FILE.is_file():
        raise FileNotFoundError(f"Missing {PENALTIES_FILE}. Run pipelines/build_dataset.py first.")
    penalties = pd.read_parquet(PENALTIES_FILE)
    lineups = (
        pd.read_parquet(LINEUPS_FILE)
        if LINEUPS_FILE.is_file()
        else pd.DataFrame(
            columns=[
                "player_id", "player_name", "player_name_normalized",
                "country_name", "team_name", "played_goalkeeper",
            ]
        )
    )
    print(f"  penalties: {len(penalties):,} rows")
    print(f"  lineups:   {len(lineups):,} rows")
    return penalties, lineups


# ── Step 2 ──────────────────────────────────────────────────────────────────────

def build_queue(penalties: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}\nSTEP 2 — Building unique player scrape queue\n{SEP}")
    pool = build_player_pool(lineups, penalties)
    n_gk = int(pool["is_goalkeeper"].sum())
    print(f"  unique players: {len(pool):,}  (goalkeepers: {n_gk:,})")
    return pool


# ── Step 3 ──────────────────────────────────────────────────────────────────────

# Per-season subprocess limits.  SoccerData's headless browser is crash-prone, so
# we cap how long any single season may take and how many times we retry it.
FBREF_SEASON_TIMEOUT_S = 90   # was 240; seasons fail fast when FBref blocks
FBREF_SEASON_ATTEMPTS = 1     # was 2; no retry — cache means re-runs are free


def pull_fbref(penalties: pd.DataFrame, lookback: int, skip: bool, offline: bool = False):
    print(f"\n{SEP}\nSTEP 3 — FBref career penalty stats (SoccerData)\n{SEP}")
    from src.ingestion.fbref import _empty_career_df, season_part_paths

    empty_foot = pd.DataFrame(columns=["player_name_normalized", "preferred_foot"])
    part_dir = CACHE_FBREF_DIR / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    if skip or offline:
        # --skip-fbref or --offline: read existing parts, no new fetches.
        career = _concat_parts(part_dir, "_career_", _empty_career_df())
        foot = _concat_parts(part_dir, "_foot_", empty_foot)
        label = "offline" if offline else "--skip-fbref"
        print(f"  {label}: loaded {len(career):,} career rows / {len(foot):,} foot rows from cache")
        return career, foot

    domestic, intl = statsbomb_seasons_to_fbref(
        penalties["season_name"].dropna().unique(), lookback_years=lookback
    )
    print(f"  domestic seasons: {domestic}")
    print(f"  international seasons: {intl}")

    jobs = [("domestic", s) for s in domestic] + [("international", s) for s in intl]

    done, failed = 0, 0
    for group, season in jobs:
        career_path, _ = season_part_paths(part_dir, group, season)
        if career_path.exists():  # resume: already fetched in a previous run
            done += 1
            continue
        if _run_fbref_season(group, season, part_dir):
            done += 1
        else:
            failed += 1

    career = _concat_parts(part_dir, "_career_", _empty_career_df())
    foot = _concat_parts(part_dir, "_foot_", empty_foot)
    print(
        f"  FBref seasons: {done} ok / {failed} failed  |  "
        f"career rows: {len(career):,}  foot rows: {len(foot):,}"
    )
    return career, foot


def _run_fbref_season(group: str, season: str, part_dir: Path) -> bool:
    """Fetch one FBref season in an isolated subprocess (browser-crash safe)."""
    import subprocess

    from src.ingestion.fbref import season_part_paths

    cmd = [
        sys.executable, "-m", "src.ingestion.fbref",
        "--group", group, "--season", season,
        "--cache-dir", str(CACHE_FBREF_DIR), "--out-dir", str(part_dir),
    ]
    career_path, _ = season_part_paths(part_dir, group, season)
    log_path = part_dir / f"_log_{group}_{season.replace('/', '-')}.txt"
    for attempt in range(1, FBREF_SEASON_ATTEMPTS + 1):
        try:
            # Redirect to file — avoids pipe-buffer deadlock when SoccerData
            # produces verbose output that fills the capture buffer.
            with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.run(
                    cmd, cwd=str(ROOT), stdout=log, stderr=log,
                    timeout=FBREF_SEASON_TIMEOUT_S,
                )
        except subprocess.TimeoutExpired:
            print(f"    ! {group} {season}: timeout (attempt {attempt})")
            continue
        if proc.returncode == 0 and career_path.exists():
            return True
        print(f"    ! {group} {season}: exit {proc.returncode} (attempt {attempt})")
    return False


def _concat_parts(part_dir: Path, prefix: str, empty: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for p in sorted(part_dir.glob(f"{prefix}*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if len(df):
            frames.append(df)
    if not frames:
        return empty
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


# ── Steps 4 + 5 ─────────────────────────────────────────────────────────────────

def resolve_all(
    pool: pd.DataFrame,
    fbref_index: dict,
    tm_client: TransfermarktClient | None,
    limit_players: int,
) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Run FBref + Transfermarkt resolution for every unique player.

    Returns the name_map DataFrame and a dict of normalised name -> the chosen
    Transfermarkt candidate attr dicts (so Step 6 can read foot/dob/height).
    """
    print(f"\n{SEP}\nSTEP 4 + 5 — Transfermarkt scrape + entity resolution\n{SEP}")
    rows: list[dict] = []
    tm_attrs: dict[str, list[dict]] = {}

    total = len(pool)
    work = pool.head(limit_players) if limit_players and limit_players > 0 else pool
    if limit_players and limit_players > 0:
        print(f"  --limit-players={limit_players}: resolving {len(work)}/{total}")

    work_ids = set(work.index)
    for i, (idx, player) in enumerate(pool.iterrows(), start=1):
        name = player.get("player_name")
        norm = player.get("player_name_normalized")

        fb_match = resolve_fbref(player, fbref_index)

        if tm_client is not None and idx in work_ids:
            candidates = tm_client.resolve_player_attributes(
                name or norm or "",
                norm or "",
                target_nationality=player.get("nationality"),
                target_is_goalkeeper=bool(player.get("is_goalkeeper", False)),
            )
            tm_attrs[norm] = candidates
            tm_match = resolve_transfermarkt(player, candidates)
        else:
            tm_match = ResolvedMatch()

        combined = better_of(tm_match, fb_match)
        rows.append(
            {
                "player_name_normalized": norm,
                "statsbomb_player_id": player.get("statsbomb_player_id"),
                "fbref_player_id": fb_match.fbref_player_id if fb_match.is_usable else "",
                "tm_player_url": tm_match.tm_player_url if tm_match.is_usable else "",
                "match_method": combined.match_method,
                "match_confidence": combined.match_confidence,
                "needs_review": combined.needs_review,
                # carried internally for Step 6 (dropped before write)
                "_tm_usable_url": tm_match.tm_player_url if tm_match.is_usable else None,
                "_fbref_usable": fb_match.is_usable,
            }
        )

        if tm_client is not None and i % 25 == 0:
            print(f"  ... resolved {i}/{total}")

    name_map = pd.DataFrame(rows)
    print(f"  resolved {len(name_map):,} players")
    return name_map, tm_attrs


def write_name_map(name_map: pd.DataFrame) -> None:
    public_cols = [
        "player_name_normalized", "statsbomb_player_id", "fbref_player_id",
        "tm_player_url", "match_method", "match_confidence", "needs_review",
    ]
    public = name_map[public_cols].copy()
    public.to_csv(RESOLUTION_DIR / "name_map.csv", index=False)

    unresolved = public[public["needs_review"]].copy()
    unresolved.to_csv(RESOLUTION_DIR / "unresolved.csv", index=False)
    print(f"  wrote name_map.csv ({len(public)}) + unresolved.csv ({len(unresolved)})")


# ── Step 6 ──────────────────────────────────────────────────────────────────────

def build_player_attributes(
    pool: pd.DataFrame,
    name_map: pd.DataFrame,
    tm_attrs: dict[str, list[dict]],
    fbref_foot: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\n{SEP}\nSTEP 6 — Building player_attributes\n{SEP}")
    fbref_foot_lookup = (
        fbref_foot.set_index("player_name_normalized")["preferred_foot"].to_dict()
        if not fbref_foot.empty
        else {}
    )
    nm = name_map.set_index("player_name_normalized")

    records: list[dict] = []
    for _, p in pool.iterrows():
        norm = p["player_name_normalized"]
        row = nm.loc[norm] if norm in nm.index else None

        chosen = None
        if row is not None and row.get("_tm_usable_url"):
            url = row["_tm_usable_url"]
            chosen = next(
                (c for c in tm_attrs.get(norm, []) if c.get("url") == url),
                None,
            )

        preferred_foot = chosen.get("preferred_foot") if chosen else None
        foot_source = "transfermarkt" if preferred_foot else None

        if not preferred_foot:
            fb_usable = bool(row.get("_fbref_usable")) if row is not None else False
            fb_foot = fbref_foot_lookup.get(norm)
            if fb_usable and fb_foot:
                preferred_foot = fb_foot
                foot_source = "fbref_misc"

        if not foot_source:
            foot_source = "unresolved"

        records.append(
            {
                "player_name_normalized": norm,
                "statsbomb_player_id": p.get("statsbomb_player_id"),
                "preferred_foot": preferred_foot,
                "date_of_birth": chosen.get("date_of_birth") if chosen else None,
                "height_cm": chosen.get("height_cm") if chosen else None,
                "nationality": (chosen.get("nationality") if chosen else None) or p.get("nationality"),
                "foot_source": foot_source,
                "tm_player_url": (row.get("tm_player_url") if row is not None else "") or None,
                "fbref_player_id": (row.get("fbref_player_id") if row is not None else "") or None,
            }
        )

    attrs = pd.DataFrame(records)
    attrs.to_parquet(ENRICH_DIR / "player_attributes.parquet", index=False)
    attrs.to_csv(ENRICH_DIR / "player_attributes.csv", index=False)
    resolved_feet = attrs["preferred_foot"].notna().sum()
    print(f"  player_attributes: {len(attrs):,} rows  |  foot resolved: {resolved_feet:,}")
    return attrs


# ── Step 7 ──────────────────────────────────────────────────────────────────────

def build_career_stats(career: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}\nSTEP 7 — Building player_career_penalty_stats\n{SEP}")
    cols = [
        "player_name_normalized", "fbref_player_id", "season_name",
        "squad", "competition", "pk_attempted", "pk_scored", "minutes_played",
    ]
    out = career.reindex(columns=cols) if not career.empty else pd.DataFrame(columns=cols)
    out.to_parquet(ENRICH_DIR / "player_career_penalty_stats.parquet", index=False)
    out.to_csv(ENRICH_DIR / "player_career_penalty_stats.csv", index=False)
    print(f"  player_career_penalty_stats: {len(out):,} rows")
    return out


# ── Step 8 ──────────────────────────────────────────────────────────────────────

def join_to_penalties(penalties: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}\nSTEP 8 — Joining enrichment back to penalties\n{SEP}")
    a = attrs.set_index("player_name_normalized")
    foot = a["preferred_foot"].to_dict()
    dob = a["date_of_birth"].to_dict()
    height = a["height_cm"].to_dict()
    source = a["foot_source"].to_dict()

    out = penalties.copy()  # never mutate the source file; only add columns

    s_norm = out["shooter_name_normalized"]
    out["shooter_preferred_foot"] = s_norm.map(foot)
    out["shooter_dob"] = s_norm.map(dob)
    out["shooter_height_cm"] = s_norm.map(height)
    out["shooter_foot_source"] = s_norm.map(source).fillna("unresolved")

    k_norm = out["keeper_name_normalized"]
    out["keeper_preferred_foot"] = k_norm.map(foot)
    out["keeper_height_cm"] = k_norm.map(height)
    out["keeper_foot_source"] = k_norm.map(source).fillna("unresolved")

    out["shooter_preferred_foot_is_missing"] = out["shooter_preferred_foot"].isna()
    out["keeper_preferred_foot_is_missing"] = out["keeper_preferred_foot"].isna()
    out["enrichment_version"] = ENRICHMENT_VERSION

    assert len(out) == len(penalties), "row count changed — enrichment must never drop rows"
    out.to_parquet(ENRICH_DIR / "penalties_enriched.parquet", index=False)
    out.to_csv(ENRICH_DIR / "penalties_enriched.csv", index=False)
    print(f"  penalties_enriched: {len(out):,} rows (== source {len(penalties):,})")
    return out


# ── Step 9 ──────────────────────────────────────────────────────────────────────

def print_summary(penalties, pool, attrs, name_map, enriched) -> None:
    print(f"\n{SEP}\nSTEP 9 — Summary\n{SEP}")

    shooters = set(penalties["shooter_name_normalized"].dropna())
    keepers = set(penalties["keeper_name_normalized"].dropna())
    foot_by_name = attrs.set_index("player_name_normalized")["preferred_foot"].to_dict()

    def pct_resolved(names: set) -> tuple[int, int]:
        resolved = sum(1 for n in names if pd.notna(foot_by_name.get(n)))
        return resolved, len(names)

    s_res, s_tot = pct_resolved(shooters)
    k_res, k_tot = pct_resolved(keepers)
    print(f"  shooters with preferred foot: {s_res}/{s_tot}  ({_safe_pct(s_res, s_tot)})")
    print(f"  keepers  with preferred foot: {k_res}/{k_tot}  ({_safe_pct(k_res, k_tot)})")

    print(f"\n  unique players in pool: {len(pool):,}")
    print(f"  players needing review:  {int(name_map['needs_review'].sum()):,}")

    print("\n  match_method breakdown:")
    for method, cnt in name_map["match_method"].value_counts().items():
        print(f"    {str(method):<32s} {cnt:>5d}")

    print("\n  match_confidence breakdown:")
    for conf, cnt in name_map["match_confidence"].value_counts().items():
        print(f"    {str(conf):<32s} {cnt:>5d}")

    print("\n  foot_source breakdown (player_attributes):")
    for src, cnt in attrs["foot_source"].value_counts().items():
        print(f"    {str(src):<32s} {cnt:>5d}")

    print("\n  penalties_enriched coverage:")
    print(f"    shooter foot present: {(~enriched['shooter_preferred_foot_is_missing']).sum():,}/{len(enriched):,}")
    print(f"    keeper foot present:  {(~enriched['keeper_preferred_foot_is_missing']).sum():,}/{len(enriched):,}")
    print(SEP)
    print("\nDone. Enrichment outputs are in outputs/enrichment/")


def _safe_pct(num: int, den: int) -> str:
    return f"{(num / den * 100):.1f}%" if den else "n/a"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich penalties with FBref + Transfermarkt.")
    parser.add_argument("--limit-players", type=int, default=0)
    parser.add_argument("--skip-fbref", action="store_true")
    parser.add_argument("--skip-tm", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Use only cached data; skip all network calls for both FBref and TM")
    parser.add_argument("--lookback", type=int, default=3)
    args = parser.parse_args()

    offline = args.offline
    _ensure_dirs()

    penalties, lineups = load_inputs()
    pool = build_queue(penalties, lineups)
    career, fbref_foot = pull_fbref(penalties, args.lookback, args.skip_fbref, offline=offline)
    fbref_index = build_fbref_index(career)

    if args.skip_tm:
        print(f"\n{SEP}\nSTEP 4 — Transfermarkt\n{SEP}\n  --skip-tm set; skipping.")
        tm_client = None
    else:
        tm_client = TransfermarktClient(CACHE_TM_DIR, offline=offline)
        if offline:
            print(f"\n{SEP}\nSTEP 4 — Transfermarkt\n{SEP}\n  offline mode: using cache only (no new scrapes)")

    name_map, tm_attrs = resolve_all(pool, fbref_index, tm_client, args.limit_players)
    write_name_map(name_map)

    attrs = build_player_attributes(pool, name_map, tm_attrs, fbref_foot)
    career_out = build_career_stats(career)
    enriched = join_to_penalties(penalties, attrs)
    print_summary(penalties, pool, attrs, name_map, enriched)


if __name__ == "__main__":
    main()
