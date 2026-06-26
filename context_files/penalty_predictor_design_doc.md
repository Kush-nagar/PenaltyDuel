# Penalty Predictor — UI Design Document (Retro Edition)
### For use with Google Stitch

---

## Design Vision

**Theme:** Retro 16-bit football game meets vintage sports poster.  
**Mood reference:** The pixel art website in the reference image (dark background, floodlit scene, warm cream content sections, bold pixel typography) combined with the American retro color palette — deep navy `#01344F`, punch red `#D12128`, warm cream `#FAE3AC`.  
**Signature element:** The home page hero is a full-width pixel art penalty scene — stadium floodlights, a goalkeeper crouched on the line, a pixel soccer ball mid-flight — rendered as inline SVG/CSS pixel art. The matchup results page uses a retro video game "VS SCREEN" header in the style of classic football games like Sensible Soccer and ISS Pro.  
**Dark mode:** All pages are dark mode. The cream `#FAE3AC` appears only as a text/accent colour on dark backgrounds, never as a background itself.

---

## Design Tokens

### Color Palette

| Token | Hex | Usage |
|---|---|---|
| `color-bg-void` | `#070C14` | Deepest background, behind hero |
| `color-bg-primary` | `#0D1520` | Main page background |
| `color-bg-surface` | `#01344F` | Cards, panels (the deep navy from ref image) |
| `color-bg-surface-raised` | `#0A2238` | Hover states, nested cards |
| `color-accent-red` | `#D12128` | Primary CTA, buttons, alerts (from ref palette) |
| `color-accent-red-dark` | `#8B0E13` | Red hover, pressed state |
| `color-accent-cream` | `#FAE3AC` | Primary text on dark, highlight labels (from ref palette) |
| `color-accent-cream-muted` | `#3D2E10` | Cream tint backgrounds |
| `color-accent-amber` | `#F5B731` | Data highlights, probability numbers, heatmap hot |
| `color-text-primary` | `#F0EAD6` | Body text on dark backgrounds |
| `color-text-secondary` | `#7A9BB5` | Captions, labels, metadata |
| `color-goal-green` | `#3AB06A` | Goal outcome, positive indicators |
| `color-save-red` | `#D12128` | Saved/missed (reuses accent red) |
| `color-pitch-green` | `#1A5C34` | Pixel pitch elements, decorative only |
| `color-border-pixel` | `#D12128` | Outer card border (red, pixel style) |
| `color-border-inner` | `#1C3A52` | Inner card dividers |
| `color-heatmap-cold` | `#01344F` | Heatmap cold zone |
| `color-heatmap-hot` | `#F5B731` | Heatmap hot zone |
| `color-scanline` | `rgba(0,0,0,0.08)` | CRT scanline overlay (repeating gradient) |

### Typography

| Role | Typeface | Weight | Usage |
|---|---|---|---|
| Pixel display | Press Start 2P | 400 | Key numbers, score displays, short ALL-CAPS labels — use very sparingly (this font is expensive to read at length) |
| Retro headline | Bebas Neue | 400 | Section titles, player names in headers, H1/H2 |
| Body / UI | IBM Plex Mono | 400, 600 | All body copy, labels, captions, form fields — monospace gives a retro terminal feel |
| Stats / data | VT323 | 400 | Large probability numbers on the scoreboard display, scoreline-style stats |

Import from Google Fonts:
```
Press+Start+2P
Bebas+Neue
IBM+Plex+Mono:wght@400;600
VT323
```

**Type scale:**
- `px-hero`: Press Start 2P, 16–24px — only for the main logo and scoreboard numbers
- `h1`: Bebas Neue 64px desktop / 40px mobile
- `h2`: Bebas Neue 40px
- `h3`: Bebas Neue 28px
- `body`: IBM Plex Mono 14px, line-height 1.7
- `label`: IBM Plex Mono 600, 11px, letter-spacing 0.14em, uppercase
- `data-lg`: VT323 56px
- `data-sm`: VT323 32px

### Pixel Border System

Cards in this design use a CSS `box-shadow`-based pixel border technique to simulate 8-bit UI chrome — thick outer border in `color-accent-red`, thinner inner inset in `color-bg-surface-raised`:

```css
box-shadow:
  0 0 0 3px #D12128,    /* pixel outer border */
  0 0 0 6px #070C14,    /* dark gap */
  0 0 0 9px #01344F;    /* navy frame */
```

Use this on: player cards, the three analysis panels, the matchup builder card, the VS header.  
Do NOT use on: nav bar, small chips/pills, inline text elements.

### Pixel Corner Accents

All major cards have small pixel-art corner brackets rendered as `::before` / `::after` pseudo-elements in `color-accent-red`. These are 8×8px L-shaped marks at each corner (top-left, top-right, bottom-left, bottom-right). They look like the corner brackets in retro game UI boxes.

### Scanline Overlay

All hero sections and the VS screen header have a full-size `::after` overlay using:
```css
background: repeating-linear-gradient(
  0deg,
  transparent,
  transparent 2px,
  rgba(0,0,0,0.07) 2px,
  rgba(0,0,0,0.07) 4px
);
pointer-events: none;
```
This adds a subtle CRT monitor scanline effect that reinforces the retro screen aesthetic.

### Spacing

Base unit `8px`. Scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 / 96px.

### Border Radius

**This is a pixel art design — no border-radius anywhere except pills.**  
- Cards, buttons, inputs: `0px` (hard square corners = pixel aesthetic)
- Chips/tags only: `2px` (near-square)

---

## Pixel Art Soccer Elements Library

The following pixel art components appear across the site. All are SVG-based, rendered inline so they scale without blur. Suggest to Stitch to use either inline SVG or CSS pixel art (scaled up `image-rendering: pixelated` sprites).

### PX-01: Pixel Soccer Ball
8×8 base (scaled to 32×32 or 64×64 display). Classic black-and-white hexagon panel pattern. Used in: nav logo, section dividers, loading states.

### PX-02: Pixel Goal (Front View)
A goal viewed from the front — white goalposts, crossbar, net rendered as a dot grid. Width 96px at base size. Used in: hero scene, heatmap background.

### PX-03: Pixel Goalkeeper Silhouette
16×24 base. A simplified goalkeeper figure in diving pose, arms extended. Colour: `color-text-secondary`. Used in: keeper dive panel, hero scene.

### PX-04: Pixel Striker Silhouette
16×24 base. Figure in kicking stance. Used in: hero scene, player cards.

### PX-05: Pixel Pitch Line Divider
Full-width divider that looks like the centre line of a football pitch — a dashed line in `color-pitch-green` with a centre circle arc on one side. Used between every major page section.

### PX-06: Pixel Stadium Floodlights
Two tower structures (left and right) with glowing pixel lights at the top. `color-accent-cream` lights with a soft glow `box-shadow`. Used in the hero and VS screen backgrounds only.

### PX-07: Pixel Scoreboard
Retro LED scoreboard panel with HOME | SCORE | AWAY layout. Numbers in VT323, background `#111`, border `color-accent-red` pixel-style. Used in the prediction results header.

### PX-08: Pixel Grass Tile
A repeating 16×8 pixel tile in two shades of `color-pitch-green` simulating alternating mow stripes. Used as the ground plane in the hero scene.

---

## Global Layout

**Max content width:** `1280px`, centered, `padding: 0 24px`.  
**Grid:** 12-column, `24px` gap.  
**Navigation height:** `56px`.

### Global Navigation Bar

**Background:** `color-bg-void`, full bleed, bottom border `3px solid color-accent-red`.

**Left — Logo:**  
Inline PX-01 soccer ball pixel art (32×32) + wordmark.  
Wordmark: `PENALTY` in Press Start 2P, `10px`, `color-accent-cream` — then on the next line `PREDICTOR` same style.  
(Two-line stacked wordmark, total height ~40px. This is the one place Press Start 2P is used in the nav.)

**Center — Nav Links:**  
`HOME` | `PLAYERS` | `ABOUT`  
IBM Plex Mono 600, `12px`, uppercase, letter-spacing `0.16em`, `color-text-secondary`.  
Active state: `color-accent-cream` with a 2px bottom border in `color-accent-red`.  
Hover: `color-accent-cream`, no border.

**Right — Data Source Badge:**  
Pill-shaped (2px radius) background `color-bg-surface`, border `1px color-border-inner`.  
Text: `StatsBomb + Kaggle`, IBM Plex Mono 400, 11px, `color-text-secondary`.  
A tiny PX-01 pixel ball precedes the text.

---

## Page 1: Home — Matchup Builder

**Route:** `/`

---

### Section 1.1 — Pixel Art Hero Scene

**Height:** `520px` desktop / `360px` mobile.  
**Background:** `color-bg-void` with the PX-08 pixel grass tile along the bottom 80px.

**Scene layout (described for SVG rendering):**

```
┌────────────────────────────────────────────────────────────┐
│  ★         ★    ★           ★      ★        ★             │  ← Stars (small pixel squares)
│                                                            │
│  [PX-06 Floodlight]                  [PX-06 Floodlight]   │
│        |                                      |            │
│                                                            │
│              ┌──────── PX-02 GOAL ────────┐               │
│              │  · · · · · · · · · · · · · │               │  ← Net as dot grid
│              │  · · · [PX-03 KEEPER] · · ·│               │
│              │  · · · · · · · · · · · · · │               │
│              └────────────────────────────┘               │
│         ⚽ ←── PX-01 ball mid-flight                       │
│  [PX-04 Striker, back view, mid-kick]                      │
│░░░░░░░░░░░░░ PX-08 PIXEL GRASS ░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────────────────────────────────────────────────────────┘
```

The goalkeeper (PX-03) is shown diving to one side, slightly off-center, hinting at the tension of the moment. The ball is between the striker and goal, pixel-art motion blur trail behind it (3–4 lighter pixels trailing left).

Scanline overlay (CSS `::after`) over the entire scene.

**Centered on top of the scene, vertically placed in upper third:**

Eyebrow text: `PRESS START` — Press Start 2P, `10px`, `color-accent-red`, letter-spacing `0.2em`. Blinks at 1s interval (CSS animation, `prefers-reduced-motion` off for this one since it's purely decorative).

Main headline: `WHO SCORES?` — Bebas Neue, `96px` desktop / `64px` mobile, `color-accent-cream`. Hard drop shadow: `4px 4px 0px #D12128` (pixel shadow, no blur radius).

Sub-headline: `Penalty matchup predictor` — IBM Plex Mono 400, `14px`, `color-text-secondary`.

---

### Section 1.2 — Matchup Builder Card

**Background:** `color-bg-primary`, full bleed, `padding: 64px 0`.

**Centered card:**  
Max-width `600px`, background `color-bg-surface`, pixel border system (see design tokens), padding `32px`.

Pixel corner accents on all four corners.

**Card header (inside card, top):**  
Row of three pixel elements: `[PX-04 Striker 24px]  ——  [PX-01 Ball 20px]  ——  [PX-03 Keeper 24px]`  
Centered, margin-bottom `24px`.

**Shooter field:**  
Label: `> SHOOTER_` — IBM Plex Mono 600, `11px`, `color-accent-red`, uppercase. The `>` prefix and `_` suffix are part of the label — they look like a terminal cursor prompt.  
Input: Full width, `height: 48px`, background `#070C14`, border `2px solid color-border-inner`, `color: color-accent-cream`, IBM Plex Mono 400, `14px`. Placeholder: `color-text-secondary`, text `Enter player name...`.  
Focus: border becomes `2px solid color-accent-red`, no border-radius.  
Autocomplete dropdown: `color-bg-void`, border `2px solid color-accent-red`. Each row: player name in IBM Plex Mono 14px `color-accent-cream`, country + club in `color-text-secondary` 12px.

**VS Divider:**  
A full-width pixel pitch line (PX-05, mini version, height 24px) with `VS` in Press Start 2P, `12px`, `color-accent-cream`, centered over it. Margin `20px 0`.

**Goalkeeper field:** Same as shooter field, label `> GOALKEEPER_`.

**Predict button:**  
Full width, `height: 52px`. Background `color-accent-red`. Text: `[ PREDICT OUTCOME ]` — Press Start 2P, `10px`, `color-accent-cream`. Hard pixel shadow: `4px 4px 0px #8B0E13`.  
Hover: background `color-accent-red-dark`, shadow shifts to `2px 2px 0px #8B0E13` (pressed-in effect).  
Loading state: Text becomes `[ LOADING... ]` with a blinking `_` cursor.

---

### Section 1.3 — Recent Matchups Ticker

**Background:** `color-bg-void`, `padding: 20px 0`. Full bleed.  
**Style:** Looks like a live scoreboard ticker — a single horizontal scrolling row, auto-scrolling left continuously (CSS animation, pauses on hover).

**Ticker label (left, fixed, not scrolling):**  
`◀ RECENT MATCHUPS` — IBM Plex Mono 600, `11px`, `color-accent-red`, uppercase. Bordered right with `2px solid color-accent-red`.

**Ticker items (scrolling):**  
Each item: `[Shooter]  vs  [Goalkeeper]  ·  XX%`  
IBM Plex Mono 400, `13px`, `color-accent-cream`. Percentage in VT323, `22px`, `color-accent-amber`.  
Items separated by a pixel soccer ball `⚽` (or PX-01 at 14px).  
Click on an item: navigate to that prediction result.

---

### Section 1.4 — Pitch Line Divider

PX-05 full-width pitch line divider. Height `48px`. This appears between every major section on every page.

---

### Section 1.5 — How It Works

`padding: 64px 0`. Background `color-bg-primary`.

**Section header:**  
Eyebrow: `// MATCH ANALYSIS ENGINE` — IBM Plex Mono 600, `11px`, `color-accent-red`, uppercase.  
Title: `HOW IT WORKS` — Bebas Neue, `48px`, `color-accent-cream`.

Three equal cards side by side (desktop), stacked (mobile). Each card uses the pixel border system.

**Card 1 — Shooter Profile**  
Pixel icon: PX-04 (Striker) in `color-accent-red`, 48px, centered at top.  
Title: `PLACEMENT MODEL` — Bebas Neue, `24px`, `color-accent-cream`.  
Body: IBM Plex Mono 400, `13px`, `color-text-secondary`, line-height 1.7.  
Copy: `Maps where each shooter aims — bottom-left, top-right, centre — weighted by kick history and Bayesian shrinkage toward league priors.`

**Card 2 — Keeper Profile**  
Pixel icon: PX-03 (Keeper diving) in `color-accent-amber`, 48px.  
Title: `DIVE TENDENCY`  
Copy: `Profiles which direction each goalkeeper commits to and how often they anticipate the shooter's pattern correctly.`

**Card 3 — Resolution**  
Pixel icon: PX-07 (Scoreboard panel) at 48px.  
Title: `OUTCOME PROBABILITY`  
Copy: `A resolution table combines both profiles into one calibrated goal probability with a plain-language breakdown of the matchup dynamics.`

---

## Page 2: Prediction Results

**Route:** `/predict?shooter={id}&goalkeeper={id}`

---

### Section 2.1 — VS Screen Header

**This is the page's signature moment.** It should feel like booting up a classic football video game and seeing the team/player select screen.

**Background:** `color-bg-void` with PX-05 pitch line repeated horizontally as a background pattern (very subtle, `opacity: 0.15`). PX-06 floodlight towers on far left and far right edges. Scanline overlay.

**Height:** `280px` desktop.

**Layout: three columns**

```
[ SHOOTER CARD ]      [ SCOREBOARD ]      [ GOALKEEPER CARD ]
```

**Shooter Card (left):**  
Background `color-bg-surface`, pixel border in `color-accent-red`. Padding `16px`.  
Player photo: `64px × 64px` square (no border radius — pixel art style), `image-rendering: pixelated` if possible.  
Name: Bebas Neue, `32px`, `color-accent-cream`.  
Club: IBM Plex Mono 400, `11px`, `color-text-secondary`, uppercase.  
Stat: `XX pens · XX% conv.` — VT323, `22px`, `color-accent-amber`.

**Central Scoreboard (center — PX-07):**  
Background `#0A0A0A`, pixel border system.  
Top row: `GOAL PROBABILITY` — IBM Plex Mono 600, `10px`, `color-text-secondary`, uppercase.  
Big number: VT323, `80px`, `color-accent-amber`. Example: `73%`.  
Below number: two chips side by side:
- `⚽ GOAL` chip: background `rgba(58,176,106,0.15)`, border `2px solid color-goal-green`, text `color-goal-green`, IBM Plex Mono 600, 11px.
- `🧤 SAVED` chip: same style in red.

Below chips: `PRESS ↓ FOR ANALYSIS` — Press Start 2P, `7px`, `color-text-secondary`. Blinking chevron `▼`.

**Goalkeeper Card (right):** Mirror of shooter card, text right-aligned, border in `color-accent-amber` (to differentiate sides).

---

### Section 2.2 — Pitch Line Divider (PX-05)

---

### Section 2.3 — Three-Panel Analysis Grid

`padding: 48px 0`. Background `color-bg-primary`.  
Three equal columns desktop, stacked on mobile. Each panel uses the pixel border system.

---

#### Panel A: Shot Placement Heatmap

**Panel label:** `> SHOOTER PLACEMENT` — IBM Plex Mono 600, `11px`, `color-accent-red`, uppercase, margin-bottom `16px`.

**The goal-mouth heatmap (keeper's perspective):**

Render a pixel art goal viewed from behind the keeper — the keeper looks out at the shooter.  

- Goalposts and crossbar: `color-accent-cream`, `4px` thick (square caps, no round joins).  
- Net behind: PX-02 dot grid pattern in `color-border-inner`.  
- **3×3 zone grid:** Each zone has a colored fill interpolating from `color-heatmap-cold` (rare) to `color-heatmap-hot` (frequent). Zone borders are `1px` `color-bg-void` (gives a grid-cell pixel look).  
- Zone probability label: VT323, `24px`, `color-accent-cream`, centered in each zone.

**Goal visual atmosphere:** Two small PX-06 floodlight towers flank the top-left and top-right corners of the goal (purely decorative, 12px tall).

**Below goal — stat row:**  
Two columns, IBM Plex Mono 400, `12px`:  
`PREFERRED FOOT:` [value — `color-accent-cream`]  
`TOP ZONE:` [zone — `color-accent-amber`]

**Legend below stats:**  
Gradient strip 160px wide, `8px` tall, `cold → hot`. Labels: `RARE` and `FREQUENT` in IBM Plex Mono 400, `10px`, `color-text-secondary`.

---

#### Panel B: Keeper Dive Tendency

**Panel label:** `> KEEPER DIVE` — IBM Plex Mono 600, `11px`, `color-accent-amber`, uppercase.

**Visual — PX-03 keeper figure:**  
The pixel keeper silhouette (32×48px) centered in a mini pitch-green rectangle (the goal line area). No colour fill — just the outline in `color-text-secondary`.

**Three dive direction bars:**

```
LEFT    [██████████░░░░░░░░░░]  38%
CENTRE  [████░░░░░░░░░░░░░░░░]  18%
RIGHT   [████████████░░░░░░░░]  44%
```

Bar style — like a retro game health bar:  
- Label: IBM Plex Mono 600, `11px`, `color-text-secondary`, uppercase, fixed width `52px`.  
- Track: `height: 12px`, background `#0A0A0A`, border `1px solid color-border-inner`, border-radius `0` (square pixel bars).  
- Fill: `color-accent-amber`. Segmented — rendered as a series of `10px` wide filled rectangles with `2px` gaps between them (pixel-block style rather than smooth fill).  
- Percentage: VT323, `22px`, `color-accent-amber`, right-aligned.

**Dominant direction chip:**  
`◀ DIVES RIGHT` — IBM Plex Mono 600, `10px`, pixel border in `color-accent-amber`, background `color-accent-cream-muted`.

**Low-data warning (if < 15 dives observed):**  
`[!] LIMITED DATA · PRIOR-WEIGHTED` — IBM Plex Mono 400, `11px`, `color-accent-red`. Displayed in a 1px dashed red border inset, no background fill.

---

#### Panel C: Match Analysis

**Panel label:** `> MATCH ANALYSIS` — IBM Plex Mono 600, `11px`, `color-accent-cream`, uppercase.

**Card border:** Pixel border system, outer border in `color-accent-cream` (not red — makes this card stand out as the key insight).

**Body text:**  
IBM Plex Mono 400, `14px`, `color-text-primary`, line-height `1.8`.  
Populated from API's plain-language output. Render as a paragraph. No bullets.  
The text has a subtle `>` character prepended (terminal style) — `color-accent-red`.

Example:
```
> Rodrigo favours bottom-left with his right
  foot — his top zone accounts for 41% of
  historical kicks. Ederson tends to dive
  right under pressure, leaving that zone
  exposed. Model gives a 73% goal
  probability driven by the zone mismatch.
```

**Confidence bar:**  
Label: `MODEL CONFIDENCE` — IBM Plex Mono 600, `11px`, `color-text-secondary`, uppercase.  
Same segmented bar style as Panel B but fill in `color-goal-green` (high confidence) or `color-accent-red` (low confidence).  
Tooltip text: `Based on XX penalties observed for this shooter / XX for this goalkeeper.`

---

### Section 2.4 — Pitch Line Divider

---

### Section 2.5 — Historical Context (Expandable)

Collapsed by default.

**Toggle row:**  
`[ ▶ SHOW PENALTY HISTORY ]` — IBM Plex Mono 600, `12px`, `color-accent-red`, full-width, background `color-bg-surface`, pixel border, `height: 44px`, centered text.  
When expanded, chevron rotates to `▼` and text becomes `[ ▼ HIDE PENALTY HISTORY ]`.

**Expanded state — penalty log table:**

| DATE | COMPETITION | GOALKEEPER | ZONE | DIVE | RESULT |
|---|---|---|---|---|---|
| Feb 2024 | EPL | Raya | Bot-L | Right | ✅ GOAL |
| Nov 2023 | UCL | Oblak | Top-R | Right | ✅ GOAL |
| Aug 2023 | EPL | Flekken | Bot-L | Left | ❌ SAVED |

Table style:  
- No border-radius.  
- Header row: background `color-bg-void`, IBM Plex Mono 600, `11px`, uppercase, `color-text-secondary`.  
- Body rows: IBM Plex Mono 400, `13px`, `color-text-primary`. Alternating row backgrounds: `color-bg-surface` and `color-bg-surface-raised`.  
- Row hover: border-left `3px solid color-accent-red`.  
- ✅ in `color-goal-green`, ❌ in `color-accent-red`.  
- Outer table border: pixel border system.

---

### Section 2.6 — New Matchup CTA

`padding: 48px 0`. Background `color-bg-primary`, centered.

A single row: PX-01 pixel ball (24px) + copy `WANT TO TEST A DIFFERENT DUEL?` in Bebas Neue, `32px`, `color-accent-cream`.  
Below: `[ NEW MATCHUP ]` button — same red full-width button style as home page, but `width: auto`, `padding: 14px 40px`.

---

## Page 3: Player Explorer

**Route:** `/players`

---

### Section 3.1 — Page Header

`padding: 48px 0 32px`. Background `color-bg-primary`.

**Eyebrow:** `// SQUAD DATABASE` — IBM Plex Mono 600, `11px`, `color-accent-red`, uppercase.  
**H1:** `PLAYER EXPLORER` — Bebas Neue, `64px`, `color-accent-cream`. Hard pixel shadow `4px 4px 0px #D12128`.  
**Body:** `Browse penalty profiles for all players in the dataset.` — IBM Plex Mono 400, `14px`, `color-text-secondary`.

---

### Section 3.2 — Filter Bar

Sticky below nav on scroll. `padding: 16px 0`. Background `color-bg-void`, bottom border `3px solid color-accent-red`.

Controls row:

1. **Search input:** `width: 260px`, same terminal-style input as home page (dark bg, red focus border), placeholder `SEARCH PLAYER...`.

2. **Role filter — pixel toggle:**  
Three connected buttons: `ALL` | `SHOOTERS` | `KEEPERS`.  
Active: background `color-accent-red`, text `color-accent-cream`, IBM Plex Mono 600.  
Inactive: background `color-bg-surface`, text `color-text-secondary`.  
Borders: `2px solid color-accent-red`. Border-radius `0`.

3. **League dropdown:** Same input style. Options: `ALL LEAGUES`, `EPL`, `SERIE A`, `PRIMEIRA LIGA`, `WORLD CUP`.

4. **Sort dropdown:** Options: `NAME A–Z`, `MOST PENALTIES`, `HIGHEST CONV.`.

---

### Section 3.3 — Player Card Grid

4 columns desktop / 2 tablet / 1 mobile. Gap `20px`. Background `color-bg-primary`, `padding: 32px 0`.

**Player Card:**  
Background `color-bg-surface`, pixel border system, padding `20px`.  
Hover: outer pixel border changes from red to `color-accent-amber`, slight `translateY(-2px)` (snap, not smooth — 0ms transition for pixel feel).

**Card layout:**

```
┌──────────────────────────────┐ ← pixel border
│ [PHOTO 56px]  NAME           │
│               CLUB · LEAGUE  │
│                              │
│  [ROLE CHIP]                 │
│                              │
│  PENALTIES  32 / 40          │
│  CONV. RATE  80%             │
│                              │
│  [ PREDICT VS THEM ▶ ]       │
└──────────────────────────────┘
```

- **Photo:** `56×56px` square, `image-rendering: pixelated`, `border: 2px solid color-border-inner`.
- **Name:** Bebas Neue, `22px`, `color-accent-cream`.
- **Club · League:** IBM Plex Mono 400, `11px`, `color-text-secondary`, uppercase.
- **Role chip:** Square pill (2px radius), `border: 2px solid`, IBM Plex Mono 600, `10px`, uppercase.  
  `SHOOTER`: border + text `color-goal-green`.  
  `GOALKEEPER`: border + text `color-accent-amber`.
- **Stats:** IBM Plex Mono 400, `11px`, `color-text-secondary`. Value: VT323, `22px`, `color-accent-cream`.
- **CTA button:** Full width, `height: 40px`. Background transparent, border `2px solid color-accent-red`, text `color-accent-red`, IBM Plex Mono 600, `11px`, uppercase. Hover: background `color-accent-red`, text `color-accent-cream`.

---

## Empty & Error States

### No Results
Centered layout:  
PX-01 pixel ball in `color-text-secondary` at `64px`.  
Below: `NO PLAYERS FOUND.` — Bebas Neue, `28px`, `color-accent-cream`.  
Below: `ADJUST YOUR SEARCH OR CLEAR FILTERS.` — IBM Plex Mono 400, `13px`, `color-text-secondary`.

### Prediction Loading
All three panels show skeleton loaders: `color-bg-surface-raised` rectangles with a blinking opacity animation (`opacity: 1 → 0.4 → 1`, `800ms` loop). The VS screen scoreboard shows `??%` in `color-text-secondary`.

### Low-Data Warning Banner
Displayed above the three panels.  
Background `color-accent-cream-muted`, border `2px dashed color-accent-red`, padding `12px 16px`.  
`[!] LIMITED DATA FOR [NAME] — PREDICTIONS USE LEAGUE-WIDE PRIORS.` — IBM Plex Mono 400, `12px`, `color-accent-cream`.

### API Error
Replaces results grid. Centered:  
`ERROR 404` — Press Start 2P, `18px`, `color-accent-red`, blinking.  
`PREDICTION UNAVAILABLE.` — Bebas Neue, `32px`, `color-accent-cream`.  
`Try again or select different players.` — IBM Plex Mono 400, `13px`, `color-text-secondary`.  
Red CTA button: `[ TRY AGAIN ]`.

---

## Responsive Behaviour

### Desktop (≥1024px)
- Analysis panels: 3 equal columns.
- Player grid: 4 columns.
- VS screen: 3-column layout.

### Tablet (768–1023px)
- Analysis panels: stacked single column.
- Player grid: 2 columns.
- VS screen: probability scoreboard goes below the two player cards, full width.

### Mobile (<768px)
- Nav: hamburger icon (pixel art 3-line icon, `color-accent-cream`) → slide-down overlay `color-bg-void`.
- All pixel art hero: `height: 280px`, headline Bebas Neue `48px`.
- Matchup builder: full bleed, 16px padding.
- VS screen probability: `font-size: 56px`.
- All grids: single column.
- Ticker: auto-scrolls, touch-draggable.

---

## Motion & Animation

**Philosophy:** Motion should feel like a retro game booting up, not a modern animation library. Prefer snap/jump transitions over easing curves. Limited motion, high impact.

- **VS screen load:** The two player cards slide in from the left and right edges simultaneously with `translateX`, `200ms` no easing (linear). Then the scoreboard number counts up from `00%` to final value, VT323 style, `600ms`, stepping in whole-number increments (`steps(N)` CSS timing, not smooth).

- **Heatmap reveal:** Zones appear one at a time in a scan pattern (left-to-right, top-to-bottom), each snapping to full opacity instantly with `0ms` transition, `50ms` stagger. Looks like a dot-matrix printer revealing the image.

- **Dive bars:** Fill from 0 to final width using `steps(10)` CSS animation — 10 discrete jumps, `400ms`. Segmented pixel blocks appear one at a time.

- **`PRESS START` blink:** `opacity: 0 ↔ 1` on a `1s` `step-start` animation. Only on the home hero.

- **Ticker scroll:** `linear`, `infinite`, `30s` cycle. Pauses on `hover`.

- **`prefers-reduced-motion`:** All transitions become `0ms`. Counting animations are disabled (show final value immediately). Blink animations are removed.

---

## Content Conventions

- All labels: uppercase.
- Percentages: whole numbers with `%` — never decimals.
- Zone names: `BOT-L`, `BOT-C`, `BOT-R`, `MID-L`, `MID-C`, `MID-R`, `TOP-L`, `TOP-C`, `TOP-R` — abbreviated for pixel-display fit.
- Outcomes: `GOAL` and `SAVED` — never `win/loss`.
- Empty states: always give a clear action. Never just say "nothing here."
- API explanation text: render as a block, terminal-prefix `>` style, no bullets.

---

## What This Design Doc Does Not Cover

- Authentication / user accounts.
- Video clip integration (future phase).
- Admin or data management UI.
- Third-party embed widgets.

---

*End of design document. All hex values, font names, spacing values, pixel element specs, and copy are implementation-ready.*
