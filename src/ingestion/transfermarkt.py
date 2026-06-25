"""
Transfermarkt ingestion via ScraperFC.

We use ScraperFC EXCLUSIVELY for Transfermarkt (SoccerData's TM support is
weaker).  Transfermarkt is the PRIMARY source for ``preferred_foot`` and also
supplies date_of_birth, height and nationality.

ScraperFC's ``Transfermarkt.scrape_player(url)`` needs a profile URL, and there
is NO name-search method, so we build the search ourselves against
``/schnellsuche/ergebnis/schnellsuche?query=NAME``, parse the candidate profile
links, then hand the chosen URL to ScraperFC.

Critical behaviour (per spec):
  * Cache EVERY raw response before parsing.  Check the cache before any network
    call so an interrupted run never re-scrapes a player it already has.
  * Rate limit: ``time.sleep(MIN_REQUEST_DELAY_S)`` (>= 3s) between live requests.
  * On any network/parse error: log, return all-None for that player, continue.
  * Never invent a foot value — unknown stays None.

Public API:
    TransfermarktClient(cache_dir)
        .resolve_player_attributes(name, normalized_name) -> dict
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.enrichment.entity_resolution import normalize_name

BASE_URL = "https://www.transfermarkt.com"
SEARCH_URL = BASE_URL + "/schnellsuche/ergebnis/schnellsuche?query={query}"
PROFILE_RE = re.compile(r"/profil/spieler/(\d+)")
# ScraperFC's scrape_player does NOT expose preferred foot, so we parse the one
# field it lacks straight from the profile HTML info-table:
#   Foot:</span> <span class="info-table__content--bold">right</span>
FOOT_RE = re.compile(
    r"Foot:\s*</span>\s*<span[^>]*>\s*([A-Za-z][A-Za-z\- ]*?)\s*</span>",
    re.IGNORECASE,
)
MIN_REQUEST_DELAY_S = 3.0  # spec: never below 3s between Transfermarkt requests
MAX_PROFILE_SCRAPES = 2    # disambiguation budget per player (keeps runtime sane)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class TransfermarktClient:
    """Thin, cache-first wrapper over ScraperFC's Transfermarkt scraper."""

    def __init__(self, cache_dir: Path, verbose: bool = True, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.offline = offline  # when True, skip all network calls; return [] for cache misses
        self._last_request_ts = 0.0
        self._session = self._build_session()
        self._tm = None  # ScraperFC client, lazily constructed
        self._logged_profile_cols = False

    # ── infrastructure ──────────────────────────────────────────────────────

    def _build_session(self):
        try:
            import cloudscraper  # bundled with ScraperFC; defeats Cloudflare

            sess = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
        except Exception:  # noqa: BLE001
            import requests

            sess = requests.Session()
        sess.headers.update({"User-Agent": _USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
        return sess

    @property
    def tm(self):
        if self._tm is None:
            from ScraperFC import Transfermarkt

            self._tm = Transfermarkt()
        return self._tm

    def _throttle(self):
        elapsed = time.time() - self._last_request_ts
        if elapsed < MIN_REQUEST_DELAY_S:
            time.sleep(MIN_REQUEST_DELAY_S - elapsed)
        self._last_request_ts = time.time()

    # ── caching helpers ─────────────────────────────────────────────────────

    def _cache_path(self, kind: str, key: str) -> Path:
        safe = re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_")[:120] or "blank"
        return self.cache_dir / f"{kind}_{safe}.json"

    @staticmethod
    def _read_cache(path: Path) -> Optional[dict]:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    @staticmethod
    def _write_cache(path: Path, payload: dict) -> None:
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ── search ──────────────────────────────────────────────────────────────

    def _search_candidates(self, name: str, normalized_name: str) -> list[dict]:
        """Return candidate {name, url, tm_id} dicts for a name query (cache-first)."""
        cache_path = self._cache_path("search", normalized_name)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached.get("candidates", [])

        if self.offline:
            return []  # no network; treat as no candidates without sleeping or writing cache

        candidates: list[dict] = []
        try:
            self._throttle()
            url = SEARCH_URL.format(query=requests_quote(name))
            resp = self._session.get(url, timeout=30)
            html = resp.text if resp.status_code == 200 else ""
            candidates = _parse_search_html(html)
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"      ! TM search failed for '{name}': {exc}")
            candidates = []

        self._write_cache(cache_path, {"query": name, "candidates": candidates})
        return candidates

    # ── profile ─────────────────────────────────────────────────────────────

    def _scrape_profile(self, url: str, tm_id: str) -> dict:
        """Scrape (cache-first) a single TM profile into a parsed attr dict."""
        cache_path = self._cache_path("profile", tm_id)
        cached = self._read_cache(cache_path)
        if cached is not None:
            # Backfill foot for caches written before the HTML foot parser existed.
            if not cached.get("_foot_checked"):
                if not cached.get("preferred_foot") and not self.offline:
                    cached["preferred_foot"] = self._scrape_foot(url)
                cached["_foot_checked"] = True
                self._write_cache(cache_path, cached)
            return cached

        if self.offline:
            return {"tm_id": tm_id, "url": url, "_foot_checked": True}

        parsed: dict = {"tm_id": tm_id, "url": url, "_foot_checked": True}
        try:
            self._throttle()
            df = self.tm.scrape_player(url)
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                if self.verbose and not self._logged_profile_cols:
                    print(f"      TM profile columns: {list(df.columns)[:20]}")
                    self._logged_profile_cols = True
                parsed.update(_parse_profile_df(df))
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"      ! TM profile scrape failed for {url}: {exc}")

        # ScraperFC omits preferred foot — parse it from the profile HTML.
        if not parsed.get("preferred_foot"):
            parsed["preferred_foot"] = self._scrape_foot(url)

        self._write_cache(cache_path, parsed)
        return parsed

    def _scrape_foot(self, url: str) -> Optional[str]:
        """Fetch the profile page and parse the preferred foot from its HTML."""
        try:
            self._throttle()
            resp = self._session.get(url, timeout=30)
            if resp.status_code != 200:
                return None
            m = FOOT_RE.search(resp.text)
            return _parse_foot(m.group(1)) if m else None
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"      ! TM foot fetch failed for {url}: {exc}")
            return None

    # ── public ──────────────────────────────────────────────────────────────

    def resolve_player_attributes(
        self,
        name: str,
        normalized_name: str,
        target_nationality: Optional[str] = None,
        target_is_goalkeeper: Optional[bool] = None,
    ) -> list[dict]:
        """Return enriched candidate dicts for one player (for entity resolution).

        Each candidate carries: name, url, tm_id, preferred_foot, date_of_birth,
        height_cm, nationality, position_is_goalkeeper.  Returns ``[]`` when the
        search found nothing.  Never raises.

        Scraping is LAZY: candidates are scraped one at a time and we stop as soon
        as one corroborates the StatsBomb player (nationality match, or GK-position
        consistency).  This keeps the common case to a single profile request.
        ``MAX_PROFILE_SCRAPES`` caps the disambiguation budget.
        """
        from src.enrichment.entity_resolution import nationality_matches

        candidates = self._search_candidates(name, normalized_name)
        if not candidates:
            return []

        exact = [c for c in candidates if normalize_name(c.get("name")) == normalized_name]
        ordered = (exact or candidates)[:MAX_PROFILE_SCRAPES]

        enriched: list[dict] = []
        for c in ordered:
            tm_id = str(c.get("tm_id") or "")
            url = c.get("url")
            if not url or not tm_id:
                continue
            attrs = self._scrape_profile(url, tm_id)
            merged = {**c, **attrs, "name": c.get("name") or attrs.get("name"), "url": url}
            enriched.append(merged)
            # Early exit once a candidate corroborates the StatsBomb identity.
            if nationality_matches(target_nationality, merged.get("nationality")):
                break
            gk = merged.get("position_is_goalkeeper")
            if target_is_goalkeeper is not None and gk is not None and gk == target_is_goalkeeper:
                break
        return enriched


# ── module-level parsers (pure, easy to reason about) ──────────────────────────

def requests_quote(value: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus(value)


def _parse_search_html(html: str) -> list[dict]:
    """Extract player profile candidates from a Transfermarkt search results page."""
    if not html:
        return []
    candidates: dict[str, dict] = {}
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=PROFILE_RE):
            href = a.get("href", "")
            m = PROFILE_RE.search(href)
            if not m:
                continue
            tm_id = m.group(1)
            text = (a.get("title") or a.get_text() or "").strip()
            full_url = href if href.startswith("http") else BASE_URL + href
            if tm_id not in candidates and text:
                candidates[tm_id] = {"name": text, "url": full_url, "tm_id": tm_id}
    except Exception:  # noqa: BLE001 — fall back to a raw regex sweep below
        pass

    if not candidates:
        for m in PROFILE_RE.finditer(html):
            tm_id = m.group(1)
            candidates.setdefault(tm_id, {"name": None, "url": None, "tm_id": tm_id})
    return list(candidates.values())


def _parse_profile_df(df: pd.DataFrame) -> dict:
    """Pull foot/dob/height/nationality/position from a scrape_player 1-row DF.

    Column names vary by ScraperFC version, so we locate fields by fuzzy header
    matching and parse values defensively.  Anything we can't confirm -> None.
    """
    row = df.iloc[0]
    cols = {str(c).strip().lower(): c for c in df.columns}

    def pick(*patterns: str):
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            for lc, original in cols.items():
                if rx.search(lc):
                    val = row[original]
                    if isinstance(val, (list, tuple)):
                        val = val[0] if len(val) else None
                    if val is not None and not (isinstance(val, float) and pd.isna(val)):
                        return val
        return None

    return {
        "name": _coerce_str(pick(r"^name$", r"player.?name", r"full.?name")),
        "preferred_foot": _parse_foot(pick(r"foot")),
        "date_of_birth": _parse_dob(pick(r"date.?of.?birth", r"\bdob\b", r"born", r"birth")),
        "height_cm": _parse_height(pick(r"height", r"size")),
        "nationality": _parse_nationality(pick(r"citizenship", r"nationalit", r"country")),
        "position_is_goalkeeper": _parse_is_gk(pick(r"position", r"main.?position")),
    }


def _coerce_str(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip() or None


def _parse_foot(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    v = str(value).strip().lower()
    if "both" in v:
        return "both"
    has_left, has_right = "left" in v, "right" in v
    if has_left and has_right:
        return "both"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return None


def _parse_dob(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\(.*?\)", "", str(value)).strip()  # drop "(age 36)" suffixes
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            ts = pd.to_datetime(text, format=fmt, errors="coerce")
            if pd.notna(ts):
                return ts.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            continue
    try:
        ts = pd.to_datetime(text, dayfirst=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    return ts.strftime("%Y-%m-%d") if pd.notna(ts) else None


def _parse_height(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*m", text)
    if m:
        return int(round(float(m.group(1)) * 100))
    m = re.search(r"(\d{2,3})\s*cm", text)
    if m:
        return int(m.group(1))
    digits = re.sub(r"[^\d.]", "", text)
    try:
        num = float(digits)
    except ValueError:
        return None
    if num < 3:          # metres, e.g. 1.85
        return int(round(num * 100))
    if 100 <= num <= 250:  # already centimetres
        return int(num)
    return None


def _parse_nationality(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if value is None:
        return None
    return str(value).split("/")[0].strip() or None


def _parse_is_gk(value) -> Optional[bool]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return "keeper" in str(value).lower() or "goalkeeper" in str(value).lower()
