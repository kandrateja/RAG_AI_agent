# DIMMS Automation

This project automates regulatory deficiency analysis using Claude on Vertex AI. It contains:

- A **theme extraction pipeline** that processes deficiency data and outputs two theme columns.
- A **countermeasures generator** that creates regulatory guidance for selected themes.
- Utility scripts for **cost estimation** and **goal tracking** Excel sheets.

---

## What This Project Does

For a deficiency dataset (`deficiency_data.xlsx`), the main pipeline creates:

- `theme_API_predefined`:
  - Runs only for rows where `VERTICAL == API`.
  - Uses predefined Sub Categories from `APIthemes.xlsx`.
- `theme_model_knowledge`:
  - Runs for all rows.
  - Uses a two-pass, data-driven process:
    - Phase 1: discover a master list of themes from non-API descriptions.
    - Phase 2: classify each row into exactly one discovered theme.

Final output keeps all original columns and appends the two theme columns.

---

## Project Structure

- `main.py` - End-to-end orchestration (data load, BigQuery fetch, theme discovery, classification, export)
- `config.py` - Environment-driven configuration
- `llm_service.py` - Claude Vertex client wrappers + resilient response parsing
- `checkpoint_manager.py` - Resume support for long runs
- `countermeasures.py` - Theme-level countermeasures/guidance generation
- `test_countermeasures.py` - Live test harness for countermeasures generation
- `.env.example` - Example runtime configuration

---

## Prerequisites

- Python 3.10+
- Access to Google Cloud Vertex AI
- Anthropic Vertex support (`anthropic[vertex]` package)
- Service account / ADC setup that can access Vertex AI

Optional:
- BigQuery access (if you want scheduled data refresh from BigQuery)

---

## Installation

### Option A: UV (recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option B: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration

Copy env template:

```bash
cp .env.example .env
```

Important variables in `.env`:

- **Core data**
  - `DATA_FILE_PATH` (default: `data/deficiency_data.xlsx`)
  - `API_THEMES_PATH` (default: `data/APIthemes.xlsx`)
- **Claude on Vertex**
  - `VERTEX_PROJECT_ID` (required)
  - `VERTEX_REGION` (default: `us-east5`)
  - `ANTHROPIC_MODEL` (default: `claude-opus-4@20250514`)
  - `ANTHROPIC_MAX_TOKENS`
  - `ANTHROPIC_TIMEOUT`
- **Pipeline performance**
  - `THEME_BATCH_SIZE`
  - `THEME_API_BATCH_SIZE`
  - `THEME_SEMAPHORE_LIMIT`
  - `THEME_DELAY_INITIAL`
  - `THEME_DELAY_MAX`
- **Testing and sampling**
  - `THEME_ROW_LIMIT`
  - `THEME_TOP_N`
  - `THEME_BOTTOM_N`
  - `DRY_RUN`
- **Optional BigQuery refresh**
  - `BIGQUERY_CREDENTIALS_PATH`
  - `BIGQUERY_PROJECT_ID`
  - `BIGQUERY_TABLE`
  - `BIGQUERY_FETCH_DAYS` (example: `5,20`)
- **Countermeasures**
  - `COUNTERMEASURE_MODEL`
  - `COUNTERMEASURE_MAX_TOKENS`

---

## Input Files

### 1) Deficiency data

Expected in `data/deficiency_data.xlsx` (or `DATA_FILE_PATH`), with key columns such as:

- `VERTICAL`
- `PRODUCTNAME`
- `DEFICIENCYDESCRIPTION`

### 2) API themes file

Expected in `data/APIthemes.xlsx` (or `API_THEMES_PATH`) with columns:

- `Category`
- `Sub Category`
- `Key word identifier`

---

## How the Theme Pipeline Works (Step by Step)

1. **Startup and output folder creation**
   - Creates timestamped output under `output_folder/output_for_<timestamp>_clusters_FINAL_DATA/`.

2. **Optional BigQuery refresh**
   - If BigQuery is configured, fetches `BIGQUERY_TABLE` only on configured days.
   - Fetch happens only once per calendar day.
   - If fresh data is fetched, checkpoints are cleared to force a full recompute.

3. **Data loading and optional sampling**
   - Loads Excel data.
   - Supports row limiting and top/bottom sampling for fast validation runs.

4. **Vertical split**
   - API rows (`VERTICAL == API`)
   - Non-API rows

5. **Phase 1 (Theme Discovery for non-API)**
   - Sends all non-API descriptions in batches.
   - Collects candidate themes.
   - Consolidates and deduplicates into a final master theme list (data-driven count).
   - Saves discovered themes into checkpoint.

6. **Phase 2A (API predefined classification)**
   - Classifies API rows against allowed `Sub Category` values only.

7. **Phase 2B (Model-knowledge classification)**
   - Classifies all rows into exactly one theme from discovered master theme list.

8. **Cleanup pass**
   - Reprocesses rows where `theme_model_knowledge` is missing/NA.

9. **Final export**
   - Writes `final_output_theme.xlsx` with:
     - `Results` sheet: original data + `theme_API_predefined` + `theme_model_knowledge`
     - `ThemeList` sheet: discovered master themes

---

## Code Build Flow (Developer Step-by-Step)

This section explains exactly how the code is structured and executed, so a new user/developer can trace the full flow with confidence.

### 1) Entry point and orchestration (`main.py`)

The application starts in `main()`:

- Creates output directory with timestamp.
- Creates a `CheckpointManager` instance.
- Optionally refreshes data from BigQuery (`fetch_bigquery_to_data_folder`).
- Loads data (`load_data`) and prepares `row_id`.
- Applies optional sampling / limits from env.
- Splits dataset into API and non-API subsets.
- Runs async pipelines (`run_pipelines`) using `asyncio.run`.
- Merges API + model-knowledge outputs back into original dataframe.
- Performs cleanup pass for NA themes.
- Writes final Excel output.

Why this design:

- Keep one clear orchestration method for business flow.
- Keep heavy logic delegated to specialized functions.

### 2) Configuration layer (`config.py`)

`config.py` centralizes all runtime controls:

- File paths (`DATA_FILE_PATH`, `API_THEMES_PATH`)
- Model/runtime (`ANTHROPIC_MODEL`, `VERTEX_PROJECT_ID`, region, token limits)
- Performance (`THEME_BATCH_SIZE`, semaphore, delays)
- Retry and parse controls
- Test toggles (`DRY_RUN`, row limits)
- BigQuery schedule and source details

Why this design:

- No hardcoding in logic files.
- Production tuning happens via `.env`, not code changes.

### 3) Data ingestion and refresh strategy (`main.py`)

There are two data paths:

- **Primary:** local excel file (`data/deficiency_data.xlsx`)
- **Optional scheduled refresh:** BigQuery fetch on configured day(s)

BigQuery flow:

- Check if today is in `BIGQUERY_FETCH_DAYS`.
- Check `.last_bigquery_fetch` to avoid duplicate same-day fetch.
- Load credentials and query full table.
- Save to `data/deficiency_data.xlsx`.
- If refreshed, clear checkpoint to avoid stale cached results.

Why this design:

- Ensures fresh data only when intended.
- Avoids unnecessary expensive reprocessing.

### 4) Checkpoint/resume model (`checkpoint_manager.py`)

Long-running classification is resumable through:

- `verticals.<name>.completed_batches`
- `batch_results` for each completed batch
- `discovered_themes` from phase 1

Used by both API and model-knowledge phases to:

- Skip completed batches
- Reuse cached outputs
- Continue from interruption point

Why this design:

- Protects long runs from restart loss.
- Enables incremental reliability at batch level.

### 5) LLM abstraction and robust parsing (`llm_service.py`)

`llm_service.py` wraps model interactions:

- Builds async/sync Anthropic Vertex clients.
- Sends Claude messages with common request format.
- Parses responses with layered fallbacks:
  1. strict Pydantic parse
  2. JSON block extraction parse
  3. schema-flex fallback parse
  4. regex extraction from truncated responses

Why this design:

- LLM output can be malformed/truncated.
- Multi-stage parsing prevents full-batch failure.

### 6) Prompt architecture in `main.py`

Code contains separate prompt contracts for each task:

- `PROMPT_API`: closed-set mapping to allowed API sub-categories.
- `PROMPT_THEME_DISCOVERY`: data-driven theme discovery for non-API.
- `PROMPT_CLASSIFY_MODEL_KNOWLEDGE`: one-theme-per-row closed-set assignment.

Why this design:

- Different business tasks need different output contracts.
- Keeps classification deterministic and auditable.

### 7) Phase 1 theme discovery (`discover_themes`)

How phase 1 is built:

- Build `FormattedDescription` from vertical/product/description.
- Chunk non-API rows into discovery batches.
- Run concurrent discovery tasks (`_run_one_discovery_batch`).
- Collect all candidate themes.
- Deduplicate and run consolidation prompt for final theme list.
- Save final themes to checkpoint.

Why this design:

- Use all non-API information for better coverage.
- Consolidation removes duplicates and near-duplicates.

### 8) Phase 2A API classification (`theme_extraction_batch_n_api`)

How API path is built:

- Load allowed sub-categories + keyword identifiers from `APIthemes.xlsx`.
- Build strict context list (“output only allowed values”).
- Process in concurrent batches with retries.
- Save each batch to checkpoint immediately.
- Normalize output to exact sub-category strings.

Why this design:

- API vertical must align with predefined taxonomy.
- Immediate batch persistence improves resiliency.

### 9) Phase 2B model-knowledge classification (`theme_extraction_batch`)

How non-API/all-row theme assignment is built:

- Use discovered theme list as closed set.
- Compact response format: JSON array of theme names only.
- Process chunk-by-chunk with concurrent task groups.
- Save each batch output immediately in checkpoint.
- Pad missing outputs with `NA` defensively.

Why this design:

- Compact output reduces token/latency overhead.
- Closed set guarantees consistency of final theme catalog.

### 10) Rate limit and retry strategy

Both classification paths include:

- Retry on rate-limit/timeouts/server errors.
- Exponential backoff.
- Adaptive delay tuning:
  - increase delay on 429
  - decrease delay on success

Why this design:

- Stabilizes large-scale production runs under variable API load.

### 11) Cleanup pass for NA rows

After merge, code identifies `NA` / blank model-knowledge themes:

- Rebuilds formatted text for only failed rows.
- Reclassifies in smaller cleanup batches.
- Rewrites only those rows in final dataframe.

Why this design:

- Improves completeness without re-running full dataset.
- Keeps throughput high and recovery targeted.

### 12) Output assembly and final write

Final write flow:

- Merge API theme results by `row_id`.
- Merge model-knowledge themes by `row_id`.
- Clean illegal Excel characters.
- Save `Results` and `ThemeList` sheets.

Why this design:

- Preserve original source columns exactly.
- Add AI insights as additive columns for downstream users.

### 13) Countermeasures module design (`countermeasures.py`)

Separate from main pipeline, this module:

- Accepts `theme_name` + full `deficiency_data` text.
- Uses async Claude call with retries.
- Produces one markdown string under `countermeasures`.
- Enforces four response sections for regulatory actionability.

Why this design:

- Designed for on-demand deep response after theme click.
- Returns a single clean output payload for UI/API integration.

### 14) Utility script architecture

- `generate_cost_sheet.py`: formula-driven cost planning workbook
- `generate_goals_sheet.py`: project goals/achievement reporting workbook

Why this design:

- Keep operational/reporting tools independent from NLP pipeline runtime.

### 15) End-to-end lifecycle summary

At runtime, the system lifecycle is:

1. Configure -> 2) Fetch/Load data -> 3) Discover themes -> 4) Classify API
5. Classify model-knowledge -> 6) Cleanup NA -> 7) Export workbook -> 8) Reuse checkpoint next run

This is the core implementation blueprint of how the codebase is built.

---

## Checkpointing and Resume

Long runs are resumable via `checkpoints/checkpoint.json`.

Tracked state includes:

- discovered themes
- completed batches per vertical
- cached batch classification results

Behavior:

- On restart, completed batches are skipped.
- If data is refreshed from BigQuery, checkpoint is cleared automatically.

---

## Run Commands

### Full pipeline

```bash
python3 main.py
```

### Quick test (sample only)

```bash
THEME_TOP_N=50 THEME_BOTTOM_N=50 python3 main.py
```

### Dry run (no LLM call)

```bash
DRY_RUN=1 THEME_ROW_LIMIT=5 python3 main.py
```

---

## Countermeasures Generator

`countermeasures.py` generates a structured markdown guidance document for a given theme and its deficiency descriptions.

### What it returns

A dictionary:

```python
{"countermeasures": "<markdown document>"}
```

### Output sections

- Countermeasures (CAPA)
- Pre-Submission Checkpoints
- Recommended Response to FDA
- Proactive Risk Mitigation

### Usage pattern

Call:

```python
await generate_countermeasures(theme_name, deficiency_data)
```

Where:

- `theme_name` is the selected theme
- `deficiency_data` is a single combined text blob (product names + deficiency descriptions)

### Test file

Use `test_countermeasures.py` to run a live test against configured model and project.

---

## Utility Excel Generators

### Cost calculator

```bash
python3 generate_cost_sheet.py
```

Creates: `claude_cost_calculator.xlsx`

Contains:

- Theme pipeline fixed monthly cost
- Chatbot cost model
- Countermeasures / generic-response / ICH-guidelines / learnings cost model
- Combined monthly/yearly projections for multiple user counts

### Goals/Achievements sheet

```bash
python3 generate_goals_sheet.py
```

Creates: `goals_achievements.xlsx`

---

## Performance Notes

- Parallel async processing is controlled by `THEME_SEMAPHORE_LIMIT`.
- Larger batches reduce round-trips but can increase response size risk.
- Retry/backoff handles rate limits and transient API failures.
- Keep `ANTHROPIC_MAX_TOKENS` high enough for batch responses.

---

## Common Troubleshooting

- **`VERTEX_PROJECT_ID is required`**
  - Set `VERTEX_PROJECT_ID` in `.env`.

- **BigQuery fetch skipped**
  - Check fetch day and credentials path.
  - Verify `BIGQUERY_FETCH_DAYS`, `BIGQUERY_CREDENTIALS_PATH`, and table/project values.

- **Slow runs**
  - Tune `THEME_BATCH_SIZE`, `THEME_API_BATCH_SIZE`, `THEME_SEMAPHORE_LIMIT`.
  - Use sampling (`THEME_TOP_N`, `THEME_BOTTOM_N`) for quick validation.

- **Unexpected empty/NA themes**
  - Cleanup pass handles residual NAs.
  - Re-run with checkpoint enabled; completed good batches are reused.

---

## Summary

This repository is a production-focused, resumable FDA deficiency intelligence workflow:

- Scheduled data refresh from BigQuery (optional)
- Two-pass, data-driven theme generation/classification
- API + non-API path handling
- Countermeasures document generation for selected themes
- Built-in operational utilities (cost and goals Excel automation)
