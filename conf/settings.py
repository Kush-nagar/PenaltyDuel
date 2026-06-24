"""
Central configuration for the penalty predictor pipeline.
All paths, pitch constants, and shared vocabularies live here.
Change DATA_DIR below if your data folder is somewhere other than the project root.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# ROOT is the project root (the folder that contains data/, src/, conf/, etc.)
ROOT = Path(__file__).parent.parent

DATA_DIR    = ROOT / "data"
COMPS_FILE  = DATA_DIR / "competitions.json"
MATCHES_DIR = DATA_DIR / "matches"
EVENTS_DIR  = DATA_DIR / "events"
LINEUPS_DIR = DATA_DIR / "lineups"

OUTPUTS_DIR = ROOT / "outputs" / "statsbomb"

# ── StatsBomb pitch coordinate constants ──────────────────────────────────────
# StatsBomb uses a 120 × 80 pitch (yards). Attacking direction is left → right
# (x increases toward 120). The goal is centered at y = 40.
#
# Goal posts (approximate, keeper's perspective):
#   left post  ≈ y = 36
#   right post ≈ y = 44
#   total width = 8 yards
#
# We divide the goal mouth into three equal columns:
#   left:   end_y < 38.67   (keeper's left  = shooter's natural right)
#   center: 38.67 ≤ end_y ≤ 41.33
#   right:  end_y > 41.33   (keeper's right = shooter's natural left)
#
# ⚠️  VALIDATE these constants in Step 0 output and the outputs/statsbomb summary.
#     The pipeline prints a zone distribution — if "low_left" and "low_right"
#     are very unequal, or if almost everything is "center", the boundaries
#     may need adjustment.  Raw (x, y) coords are preserved in the parquet output
#     so you can re-bin without re-running the full pipeline.
GOAL_Y_LEFT_POST  = 36.0
GOAL_Y_RIGHT_POST = 44.0
GOAL_Y_CENTER_LOW  = 38.67   # left edge of centre column
GOAL_Y_CENTER_HIGH = 41.33   # right edge of centre column

# Shots logged with end_location_z >= this threshold are classified "high".
# Shots with z missing (not all StatsBomb events log z) default to "low".
SHOT_Z_HIGH_THRESHOLD = 2.0

# ── Domain vocabularies ────────────────────────────────────────────────────────
GOALKEEPER_POSITION_NAME = "Goalkeeper"

# Maps StatsBomb's fine-grained shot_outcome_name → the 3-class target label.
# outcome_3 values: "scored" | "saved" | "missed"
# Rows where the outcome isn't in this map will get outcome_3 = "unknown" —
# the validation step will warn you if that happens.
OUTCOME_3_MAP: dict[str, str] = {
    "Goal":             "scored",
    "Saved":            "saved",
    "Saved Off Target": "saved",
    "Off T":            "missed",
    "Post":             "missed",
    "Wayward":          "missed",
    "Blocked":          "missed",
}