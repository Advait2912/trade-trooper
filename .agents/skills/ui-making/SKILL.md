# ui-making

Claude-aligned UI patterns for Streamlit dashboards in this repo.
Apply when building or redesigning `web/streamlit_app.py` and its page modules.

## Palette

| Token | Hex |
|-------|-----|
| background | `#0B1220` |
| card background | `#111827` |
| accent (primary) | `#22D3EE` |
| success | `#22C55E` |
| danger | `#EF4444` |
| warning | `#F59E0B` |
| muted | `#9CA3AF` |
| text | `#E5E7EB` |

## Components

- **KPI tile**: large number + label + optional delta arrow. `web.ui.theme.kpi_tile`.
- **Card**: bordered container with subtle background. `web.ui.theme.card`.
- **Status pill**: colored pill for running/stopped/error. `web.ui.theme.status_pill`.
- **Section header**: emoji icon + title + optional subtitle. `web.ui.theme.section`.

## Layout

- Streamlit `layout="wide"`.
- Max 8 columns; mobile stacks to one column.
- 24-px gutters between cards.
- One accent color (`#22D3EE`) for interactive elements; success/danger only for semantic states.

## Typography

- Headings for hierarchy (h1 title, h3 sections).
- `st.code` / monospace for numbers.
- Muted text for secondary info.

## Chart rules

- All charts live in `web.ui.charts` and share one plotly template with the palette above.
- Titles short and imperative: "Account Equity", "Realized P&L per Trade".
- No chart is taller than ~420px unless it carries more than two series.

## Page structure

1. `section()` header with tab-appropriate emoji.
2. Status banner (runner running/stopped) where relevant.
3. KPI tiles row.
4. Charts in columns.
5. Tables last.

## Paths

- All paths relative to repo root via `web.ui.data.repo_root()`.
- Never use absolute paths; never `.absolute()` in new code.