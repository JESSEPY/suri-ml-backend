# Training Data Upload & Retraining Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give admins a browser UI to upload new soil-test data, retrain the fertilizer model, review its metrics against the live model, and promote or roll back — with no CLI, git, or CI step.

**Architecture:** `suri-ml-backend` (FastAPI on a single EC2 instance) gains an upload validator that reuses the existing deterministic derivation functions, a background retrain job, a Supabase-backed model-version registry, and an in-process hot-swap of the loaded model. `suri`'s Next.js admin dashboard gets a new tab that proxies to those endpoints through an authenticated server route. `/predict` keeps serving live traffic throughout retraining — training only ever writes a *new* artifact directory and never touches the loaded model until an admin promotes it.

**Tech Stack:** Python 3.12, FastAPI, scikit-learn/XGBoost, joblib, supabase-py, pandas, openpyxl, pytest; Next.js (App Router), React, TanStack Query, shadcn/ui, sonner.

**Spec:** `suri-ml-backend/docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md`

## Global Constraints

- **Single uvicorn worker only.** Hot-swap is in-process memory. `deploy/suri-ml.service`'s `ExecStart` must never gain `--workers N` (N>1) or `--reload`, or promotion would update one worker's memory while others keep serving the old model. This is enforced by a comment in the unit file (Task 9).
- **`/predict` is never paused, blocked, or maintenance-moded during retraining.** Decided in the spec (§4, §7, §10).
- **Every derived field is computed by existing functions, never re-implemented:** `ml/agronomy.nutrient_requirements`, `ml/agronomy.total_nitrogen_from_om`, `ml/remarks.derive_{ph,om,p,k}_remark`, `ml/recipes.plan_applications`, `data/nutrient_recommendation_guide.{k_cmol_to_ppm,target_unit_for_crop,resolve_variety,resolve_cultivation_method,resolve_age_band}`.
- **The accepted-crop list is read from `data.nutrient_recommendation_guide.CROPS`.** Never hardcode a crop list in new code (spec §1 forward-compatibility).
- **No golden-record (SCC-417) check.** The spec's §4 mentions one; that was carried over from the retired synthetic-data era and is **wrong for this codebase**. `tests/test_train.py::test_no_golden_test_key` actively asserts `"golden_test" not in metadata`, and the Sorghum/SCC-417 fixture was deliberately retired when the BSWM guide-table lookup replaced the baseline+formula approach. The retrain review gate is therefore: per-target MAE/R², the `dummy_mae` baseline comparison already present in `metrics.overall`, and the new-vs-current diff. **Task 0 corrects the spec text.**
- **Artifact directory naming is `%Y-%m-%d-%H%M%S` UTC**, produced by `ml/train.py::train()`. Never construct a version string anywhere else.
- **`ml/artifacts/` paths are relative to the process CWD**, which systemd pins via `WorkingDirectory=/opt/suri-ml-backend`. Pass `ARTIFACT_ROOT` explicitly rather than assuming CWD in new modules.
- Commit after every task. Run `python -m pytest` from `suri-ml-backend/` before each commit.

---

### Task 0: Correct the spec's retired-golden-test reference

**Files:**
- Modify: `docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md`

**Interfaces:**
- Consumes: nothing
- Produces: nothing (documentation only)

- [ ] **Step 1: Remove the golden-record bullet from spec §4**

In `docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md`, find this line under "**`GET /retrain/{job_id}`**":

```markdown
- The SCC-417 golden-record check (§10.3 of the original spec), pass/fail + delta.
```

Replace it with:

```markdown
- (No golden-record check: the SCC-417/Sorghum fixture was retired when the BSWM
  guide-table lookup replaced the old baseline+formula approach —
  `tests/test_train.py::test_no_golden_test_key` asserts its absence. The dummy
  baseline plus the vs-current diff are the review gate.)
```

- [ ] **Step 2: Remove the matching mention from spec §7's safety gates**

Find, in the "**Retrain safety gates**" area of §7 (Error Handling & Safety) or §4, any remaining sentence naming a golden-record check and delete it. If none remains after Step 1, skip.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md
git commit -m "docs: drop retired SCC-417 golden check from pipeline spec"
```

---

### Task 1: Paginate the Supabase training-data read

**Why this is first:** `ml/train.py::_load_frame()` currently calls `.select("*")` with no range. PostgREST caps a single response at 1000 rows by default. The real corpus is ~3,076 rows, so a Supabase-sourced retrain — which is exactly what this pipeline triggers — would silently train on a third of the data and report plausible-looking metrics. The README only ever documents `--csv`, so this path has likely never run at full size.

**Files:**
- Modify: `ml/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `data.ingest_to_supabase.{TABLE, get_client}`
- Produces: `ml.train.fetch_all_rows(client, table: str, page_size: int = 1000) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`:

```python
class TestFetchAllRowsPaginates:
    """PostgREST caps one response at 1000 rows; the real corpus is larger.
    A single .select("*") would silently return a truncated corpus and train
    on it without any error surfacing."""

    def test_keeps_requesting_until_a_short_page_arrives(self):
        from ml.train import fetch_all_rows

        pages = [
            [{"laboratory_code": f"A{i}"} for i in range(1000)],
            [{"laboratory_code": f"B{i}"} for i in range(1000)],
            [{"laboratory_code": f"C{i}"} for i in range(76)],
        ]
        requested_ranges = []

        class FakeQuery:
            def __init__(self, page):
                self._page = page

            def range(self, start, end):
                requested_ranges.append((start, end))
                return self

            def execute(self):
                return type("R", (), {"data": self._page})()

        class FakeTable:
            def __init__(self):
                self._calls = 0

            def select(self, _columns):
                page = pages[self._calls]
                self._calls += 1
                return FakeQuery(page)

        class FakeClient:
            def __init__(self):
                self._table = FakeTable()

            def table(self, _name):
                return self._table

        rows = fetch_all_rows(FakeClient(), "ml_soil_training_data")

        assert len(rows) == 2076
        assert requested_ranges == [(0, 999), (1000, 1999), (2000, 2999)]

    def test_single_short_page_stops_immediately(self):
        from ml.train import fetch_all_rows

        class FakeQuery:
            def range(self, start, end):
                return self

            def execute(self):
                return type("R", (), {"data": [{"laboratory_code": "A"}]})()

        class FakeClient:
            def table(self, _name):
                return type("T", (), {"select": lambda _self, _c: FakeQuery()})()

        assert len(fetch_all_rows(FakeClient(), "t")) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_train.py::TestFetchAllRowsPaginates -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_all_rows'`

- [ ] **Step 3: Implement pagination**

In `ml/train.py`, add above `_load_frame`:

```python
SUPABASE_PAGE_SIZE = 1000


def fetch_all_rows(client, table: str, page_size: int = SUPABASE_PAGE_SIZE) -> list[dict]:
    """PostgREST returns at most 1000 rows per response regardless of how many
    match, and does so without any error or truncation signal. A bare
    .select("*") therefore silently caps the training corpus -- read it in
    explicit ranges until a short page proves the end was reached."""
    rows: list[dict] = []
    start = 0
    while True:
        page = client.table(table).select("*").range(start, start + page_size - 1).execute().data
        rows.extend(page)
        if len(page) < page_size:
            return rows
        start += page_size
```

Then replace the Supabase branch of `_load_frame`:

```python
    from data.ingest_to_supabase import TABLE, get_client
    return pd.DataFrame(fetch_all_rows(get_client(), TABLE))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS (all tests in the file, including the pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add ml/train.py tests/test_train.py
git commit -m "fix: paginate the Supabase training read past PostgREST's 1000-row cap"
```

---

### Task 2: Ingest already-parsed records (not just a CSV path)

**Why:** `data/ingest_to_supabase.ingest()` takes a filesystem path and reads it with pandas. Uploaded data arrives as in-memory dicts that have already been validated and derived, so it needs a records-level entry point. The batching and NaN-to-None conversion are reused, not rewritten.

**Files:**
- Modify: `data/ingest_to_supabase.py`
- Test: `tests/test_ingest_to_supabase.py`

**Interfaces:**
- Consumes: `data.ingest_to_supabase.{chunked, get_client, TABLE, DEFAULT_BATCH_SIZE}`
- Produces: `data.ingest_to_supabase.ingest_records(records: list[dict], batch_size: int = DEFAULT_BATCH_SIZE, client=None) -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ingest_to_supabase.py`:

```python
class TestIngestRecords:
    def test_upserts_every_record_in_batches_on_the_composite_key(self):
        from data.ingest_to_supabase import ingest_records

        captured = []

        class FakeTable:
            def upsert(self, batch, on_conflict):
                captured.append((len(batch), on_conflict))
                return self

            def execute(self):
                return None

        class FakeClient:
            def table(self, _name):
                return FakeTable()

        records = [{"laboratory_code": f"L{i}", "data_source": "lab"} for i in range(7)]
        total = ingest_records(records, batch_size=3, client=FakeClient())

        assert total == 7
        assert [size for size, _ in captured] == [3, 3, 1]
        assert {conflict for _, conflict in captured} == {"laboratory_code,data_source"}

    def test_empty_input_writes_nothing(self):
        from data.ingest_to_supabase import ingest_records

        class ExplodingClient:
            def table(self, _name):
                raise AssertionError("must not touch Supabase for an empty batch")

        assert ingest_records([], client=ExplodingClient()) == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ingest_to_supabase.py::TestIngestRecords -v`
Expected: FAIL with `ImportError: cannot import name 'ingest_records'`

- [ ] **Step 3: Implement it**

In `data/ingest_to_supabase.py`, add after `frame_to_records`:

```python
def ingest_records(
    records: list[dict], batch_size: int = DEFAULT_BATCH_SIZE, client: Client | None = None
) -> int:
    """Upsert already-built rows (the admin-upload path, where rows were
    validated and derived in memory rather than read from a CSV on disk).
    Same idempotent composite-key upsert as ingest(); the `client` parameter
    exists so tests can inject a fake instead of reaching Supabase."""
    if not records:
        return 0

    client = client or get_client()
    total = 0
    for batch in chunked(records, batch_size):
        client.table(TABLE).upsert(batch, on_conflict="laboratory_code,data_source").execute()
        total += len(batch)
    return total
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ingest_to_supabase.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data/ingest_to_supabase.py tests/test_ingest_to_supabase.py
git commit -m "feat: add ingest_records() for the in-memory upload path"
```

---

### Task 3: Upload template parser and validator

**Files:**
- Create: `data/training_upload.py`
- Modify: `.gitignore`
- Test: `tests/test_training_upload.py`

**Interfaces:**
- Consumes: `data.nutrient_recommendation_guide.{CROPS, k_cmol_to_ppm, target_unit_for_crop, resolve_variety, resolve_cultivation_method, resolve_age_band}`, `ml.agronomy.{nutrient_requirements, total_nitrogen_from_om}`, `ml.recipes.plan_applications`, `ml.remarks.{derive_ph_remark, derive_om_remark, derive_p_remark, derive_k_remark}`
- Produces:
  - `data.training_upload.TEMPLATE_COLUMNS: list[str]`
  - `data.training_upload.REQUIRED_COLUMNS: list[str]`
  - `data.training_upload.parse_upload(content: bytes, filename: str) -> list[dict]`
  - `data.training_upload.validate_rows(raw_rows: list[dict], existing_codes: set[str]) -> ValidationResult`
  - `data.training_upload.ValidationResult` — frozen dataclass with `.accepted: list[dict]` (each `{"row_number": int, "laboratory_code": str, "raw": dict, "derived": dict}`) and `.rejected: list[dict]` (each `{"row_number": int, "laboratory_code": str | None, "reason": str}`)
  - `data.training_upload.build_training_row(raw: dict) -> dict` — the 28-column `ml_soil_training_data` row

**Note on `.gitignore`:** the repo ignores `*.csv` wholesale (so generated corpora aren't committed). Tests in this task build their CSVs in `tmp_path` and commit none, but a later engineer adding a fixture CSV under `tests/` would have it silently ignored — Step 5 adds a narrow negation to prevent that trap.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_upload.py`:

```python
"""Upload-template parsing and validation.

Every derived value (total N, the four remarks, the three targets, both
formulations) is asserted to equal what the existing production functions
return for the same inputs -- the point of this module is that it never
re-implements agronomy, only routes to it.
"""

import io

import openpyxl
import pytest

from data.nutrient_recommendation_guide import k_cmol_to_ppm, target_unit_for_crop
from data.training_upload import (
    TEMPLATE_COLUMNS, build_training_row, parse_upload, validate_rows,
)
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark

VALID_ROW = {
    "laboratory_code": "2026-UP1",
    "crop": "Corn",
    "ph": "6.4",
    "organic_matter_pct": "2.8",
    "phosphorus_ppm": "18.0",
    "potassium_cmol_kg": "0.35",
}


def csv_bytes(rows, columns=TEMPLATE_COLUMNS):
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    return ("\n".join(lines) + "\n").encode()


class TestParseUpload:
    def test_reads_a_csv_into_dicts_keyed_by_header(self):
        rows = parse_upload(csv_bytes([VALID_ROW]), "upload.csv")
        assert len(rows) == 1
        assert rows[0]["laboratory_code"] == "2026-UP1"
        assert rows[0]["crop"] == "Corn"

    def test_reads_an_xlsx_into_the_same_shape(self):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(TEMPLATE_COLUMNS)
        sheet.append([VALID_ROW.get(column, "") for column in TEMPLATE_COLUMNS])
        buffer = io.BytesIO()
        workbook.save(buffer)

        rows = parse_upload(buffer.getvalue(), "upload.xlsx")
        assert len(rows) == 1
        assert rows[0]["laboratory_code"] == "2026-UP1"

    def test_rejects_an_unsupported_extension(self):
        with pytest.raises(ValueError, match="\\.csv, \\.xlsx"):
            parse_upload(b"whatever", "upload.pdf")

    def test_rejects_a_file_missing_a_required_column(self):
        columns = [c for c in TEMPLATE_COLUMNS if c != "ph"]
        with pytest.raises(ValueError, match="ph"):
            parse_upload(csv_bytes([VALID_ROW], columns=columns), "upload.csv")

    def test_skips_fully_blank_trailing_rows(self):
        content = csv_bytes([VALID_ROW]) + b",,,,,,,,,,\n"
        assert len(parse_upload(content, "upload.csv")) == 1


class TestValidateRowsAccepts:
    def test_accepts_a_well_formed_row(self):
        result = validate_rows([VALID_ROW], existing_codes=set())
        assert result.rejected == []
        assert len(result.accepted) == 1
        assert result.accepted[0]["laboratory_code"] == "2026-UP1"
        assert result.accepted[0]["row_number"] == 2

    def test_derived_values_match_the_production_functions_exactly(self):
        result = validate_rows([VALID_ROW], existing_codes=set())
        derived = result.accepted[0]["derived"]

        ph, om, p, k = 6.4, 2.8, 18.0, 0.35
        n_t, p_t, k_t = nutrient_requirements("Corn", om, p, k)
        app1, app2 = plan_applications(n_t, p_t, k_t, target_unit_for_crop("Corn"))

        assert derived["total_nitrogen_pct"] == pytest.approx(total_nitrogen_from_om(om))
        assert derived["ph_remark"] == derive_ph_remark(ph)
        assert derived["om_remark"] == derive_om_remark(om)
        assert derived["p_remark"] == derive_p_remark(p)
        assert derived["k_remark"] == derive_k_remark(k_cmol_to_ppm(k))
        assert derived["target_n"] == n_t
        assert derived["target_p2o5"] == p_t
        assert derived["target_k2o"] == k_t
        assert derived["app1_formulation"] == app1
        assert derived["app2_formulation"] == app2
        assert derived["target_unit"] == target_unit_for_crop("Corn")

    def test_optional_metadata_is_carried_through(self):
        row = {**VALID_ROW, "farm_area_sqm": "5000", "sampling_site": "Argao",
               "date_of_sampling": "2026-03-04"}
        accepted = validate_rows([row], existing_codes=set()).accepted[0]
        built = build_training_row(accepted["raw"])
        assert built["farm_area_sqm"] == 5000.0
        assert built["sampling_site"] == "Argao"
        assert built["date_of_sampling"] == "2026-03-04"

    def test_data_source_is_always_lab(self):
        accepted = validate_rows([VALID_ROW], existing_codes=set()).accepted[0]
        assert build_training_row(accepted["raw"])["data_source"] == "lab"


class TestValidateRowsRejects:
    def _reason(self, row, existing_codes=frozenset()):
        result = validate_rows([row], existing_codes=set(existing_codes))
        assert result.accepted == []
        assert len(result.rejected) == 1
        return result.rejected[0]["reason"]

    def test_unknown_crop_lists_the_valid_ones(self):
        reason = self._reason({**VALID_ROW, "crop": "Papaya"})
        assert "Papaya" in reason
        assert "Corn" in reason

    def test_missing_laboratory_code(self):
        assert "laboratory_code" in self._reason({**VALID_ROW, "laboratory_code": ""})

    def test_duplicate_of_an_existing_row(self):
        reason = self._reason(VALID_ROW, existing_codes={"2026-UP1"})
        assert "already" in reason.lower()

    def test_duplicate_within_the_same_file(self):
        result = validate_rows([VALID_ROW, VALID_ROW], existing_codes=set())
        assert len(result.accepted) == 1
        assert len(result.rejected) == 1
        assert "twice" in result.rejected[0]["reason"].lower()

    def test_ph_out_of_range(self):
        assert "ph" in self._reason({**VALID_ROW, "ph": "15"}).lower()

    def test_non_numeric_ph(self):
        assert "ph" in self._reason({**VALID_ROW, "ph": "acidic"}).lower()

    @pytest.mark.parametrize("column", ["organic_matter_pct", "phosphorus_ppm", "potassium_cmol_kg"])
    def test_missing_a_field_the_target_lookup_needs(self, column):
        reason = self._reason({**VALID_ROW, column: ""})
        assert column in reason

    def test_organic_matter_out_of_range(self):
        assert "organic_matter_pct" in self._reason({**VALID_ROW, "organic_matter_pct": "140"})

    def test_negative_phosphorus(self):
        assert "phosphorus_ppm" in self._reason({**VALID_ROW, "phosphorus_ppm": "-1"})

    def test_potassium_above_the_supported_ceiling(self):
        assert "potassium_cmol_kg" in self._reason({**VALID_ROW, "potassium_cmol_kg": "9"})

    def test_unresolvable_age_band(self):
        reason = self._reason({
            "laboratory_code": "2026-UP2", "crop": "Coconut", "ph": "6.4",
            "organic_matter_pct": "2.8", "phosphorus_ppm": "18.0",
            "potassium_cmol_kg": "0.35", "age_band": "700yrs",
        })
        assert "age_band" in reason

    def test_a_bad_row_does_not_block_the_good_rows_around_it(self):
        good_a = {**VALID_ROW, "laboratory_code": "2026-A"}
        bad = {**VALID_ROW, "laboratory_code": "2026-B", "crop": "Nope"}
        good_b = {**VALID_ROW, "laboratory_code": "2026-C"}

        result = validate_rows([good_a, bad, good_b], existing_codes=set())
        assert [entry["laboratory_code"] for entry in result.accepted] == ["2026-A", "2026-C"]
        assert result.rejected[0]["row_number"] == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_training_upload.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.training_upload'`

- [ ] **Step 3: Implement the module**

Create `data/training_upload.py`:

```python
"""Parse and validate the admin upload template.

The template carries only RAW measurements. Every derived value -- total
nitrogen, the four remarks, the three targets, both formulations -- is
computed here by calling the same production functions /predict uses, so an
admin can neither supply nor mistype a derived value, and an uploaded row can
never disagree with what the service would have derived itself.

Validation is per-row and non-fatal: one malformed row is rejected with a
specific reason while every other row in the file still goes through.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import openpyxl

from data.nutrient_recommendation_guide import (
    CROPS, k_cmol_to_ppm, resolve_age_band, resolve_cultivation_method,
    resolve_variety, target_unit_for_crop,
)
from ml.agronomy import nutrient_requirements, total_nitrogen_from_om
from ml.recipes import plan_applications
from ml.remarks import (
    derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark,
)

# Raw measurements only -- nothing derived, nothing an admin should be
# transcribing by hand. data_source is deliberately absent: every
# admin-uploaded row is real lab data and is stamped "lab" by
# build_training_row(), matching data/parse_real_soil_database.py.
REQUIRED_COLUMNS = [
    "laboratory_code", "crop", "ph",
    "organic_matter_pct", "phosphorus_ppm", "potassium_cmol_kg",
]

OPTIONAL_COLUMNS = [
    "variety", "cultivation_method", "age_band",
    "farm_area_sqm", "sampling_site", "date_of_sampling",
]

TEMPLATE_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Mirrors main.py's SoilTestRequest Pydantic bounds exactly, so a row that
# would be refused at /predict is also refused as training data.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "ph": (0.0, 14.0),
    "organic_matter_pct": (0.0, 100.0),
    "phosphorus_ppm": (0.0, 500.0),
    "potassium_cmol_kg": (0.0, 6.0),
}

# The ml_soil_training_data column order (data/parse_real_soil_database.py's
# OUTPUT_COLUMNS).
TRAINING_ROW_COLUMNS = [
    "laboratory_code", "data_source", "sampling_site", "date_of_sampling",
    "crop", "variety", "cultivation_method", "age_band", "farm_area_sqm",
    "ph", "organic_matter_pct", "total_nitrogen_pct",
    "phosphorus_ppm", "potassium_cmol_kg",
    "temp", "moist", "ec", "sal",
    "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


@dataclass(frozen=True)
class ValidationResult:
    accepted: list[dict]
    rejected: list[dict]


class UploadFormatError(ValueError):
    """The file itself is unusable (wrong type, missing a required column) --
    as opposed to a single bad row, which is reported per-row instead."""


def _require_columns(header: list[str]) -> None:
    present = {(column or "").strip() for column in header}
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if missing:
        raise UploadFormatError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected header: {', '.join(TEMPLATE_COLUMNS)}"
        )


def _parse_csv(content: bytes) -> list[dict]:
    text = content.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise UploadFormatError("The file is empty.")
    _require_columns(header)

    keys = [(column or "").strip() for column in header]
    rows = []
    for values in reader:
        if not any((value or "").strip() for value in values):
            continue
        rows.append({
            key: (values[index] if index < len(values) else "")
            for index, key in enumerate(keys)
        })
    return rows


def _parse_xlsx(content: bytes) -> list[dict]:
    workbook = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    iterator = sheet.iter_rows(values_only=True)
    try:
        header = [str(cell) if cell is not None else "" for cell in next(iterator)]
    except StopIteration:
        raise UploadFormatError("The file is empty.")
    _require_columns(header)

    keys = [column.strip() for column in header]
    rows = []
    for values in iterator:
        if not any(value is not None and str(value).strip() for value in values):
            continue
        rows.append({
            key: ("" if index >= len(values) or values[index] is None else str(values[index]))
            for index, key in enumerate(keys)
        })
    return rows


def parse_upload(content: bytes, filename: str) -> list[dict]:
    lowered = filename.lower()
    if lowered.endswith(".csv"):
        return _parse_csv(content)
    if lowered.endswith((".xlsx", ".xlsm")):
        return _parse_xlsx(content)
    raise UploadFormatError(
        f"Unsupported file type {filename!r}. Upload .csv, .xlsx, or .xlsm."
    )


def _text(raw: dict, column: str) -> str:
    return str(raw.get(column, "") or "").strip()


def _optional_text(raw: dict, column: str) -> str | None:
    value = _text(raw, column)
    return value or None


def _number(raw: dict, column: str, required: bool) -> float | None:
    value = _text(raw, column)
    if not value:
        if required:
            raise ValueError(f"{column} is required and was empty.")
        return None
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{column} must be a number, got {value!r}.")

    bounds = NUMERIC_BOUNDS.get(column)
    if bounds is not None:
        low, high = bounds
        if not low <= parsed <= high:
            raise ValueError(f"{column} must be between {low} and {high}, got {parsed}.")
    elif parsed <= 0:
        raise ValueError(f"{column} must be greater than 0, got {parsed}.")
    return parsed


def build_training_row(raw: dict) -> dict:
    """Turn one validated raw template row into a full ml_soil_training_data
    row. Raises ValueError/KeyError for anything unusable -- validate_rows()
    is what turns those into per-row rejection reasons."""
    crop = _text(raw, "crop")
    if crop not in CROPS:
        raise ValueError(f"Unknown crop {crop!r}. Valid crops: {', '.join(CROPS)}")

    ph = _number(raw, "ph", required=True)
    om = _number(raw, "organic_matter_pct", required=True)
    phosphorus = _number(raw, "phosphorus_ppm", required=True)
    potassium = _number(raw, "potassium_cmol_kg", required=True)
    farm_area = _number(raw, "farm_area_sqm", required=False)

    variety = _optional_text(raw, "variety")
    cultivation_method = _optional_text(raw, "cultivation_method")
    age_band = _optional_text(raw, "age_band")

    # Raises KeyError for an age_band this crop's guide rows don't have --
    # the same failure /predict turns into a 422.
    target_n, target_p2o5, target_k2o = nutrient_requirements(
        crop, om, phosphorus, potassium,
        variety=variety, cultivation_method=cultivation_method, age_band=age_band,
    )
    target_unit = target_unit_for_crop(crop)
    app1, app2 = plan_applications(target_n, target_p2o5, target_k2o, target_unit)

    return {
        "laboratory_code": _text(raw, "laboratory_code"),
        "data_source": "lab",
        "sampling_site": _optional_text(raw, "sampling_site"),
        "date_of_sampling": _optional_text(raw, "date_of_sampling"),
        "crop": crop,
        "variety": resolve_variety(crop, variety),
        "cultivation_method": resolve_cultivation_method(crop, cultivation_method),
        "age_band": resolve_age_band(crop, age_band),
        "farm_area_sqm": farm_area,
        "ph": ph,
        "organic_matter_pct": om,
        "total_nitrogen_pct": round(total_nitrogen_from_om(om), 4),
        "phosphorus_ppm": phosphorus,
        "potassium_cmol_kg": potassium,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": derive_ph_remark(ph),
        "om_remark": derive_om_remark(om),
        "p_remark": derive_p_remark(phosphorus),
        "k_remark": derive_k_remark(k_cmol_to_ppm(potassium)),
        "target_n": target_n,
        "target_p2o5": target_p2o5,
        "target_k2o": target_k2o,
        "target_unit": target_unit,
        "app1_formulation": app1,
        "app2_formulation": app2,
    }


# Fields the admin reviews before committing -- everything the service
# derived rather than everything it stored.
_REVIEW_FIELDS = [
    "total_nitrogen_pct", "ph_remark", "om_remark", "p_remark", "k_remark",
    "target_n", "target_p2o5", "target_k2o", "target_unit",
    "app1_formulation", "app2_formulation",
]


def validate_rows(raw_rows: list[dict], existing_codes: set[str]) -> ValidationResult:
    """One rejected row never blocks the rest of the file. row_number is
    1-based including the header, so it matches what the admin sees in Excel."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_in_file: set[str] = set()

    for offset, raw in enumerate(raw_rows):
        row_number = offset + 2
        code = _text(raw, "laboratory_code")

        if not code:
            rejected.append({
                "row_number": row_number, "laboratory_code": None,
                "reason": "laboratory_code is required and was empty.",
            })
            continue
        if code in seen_in_file:
            rejected.append({
                "row_number": row_number, "laboratory_code": code,
                "reason": f"laboratory_code {code!r} appears twice in this file.",
            })
            continue
        if code in existing_codes:
            rejected.append({
                "row_number": row_number, "laboratory_code": code,
                "reason": f"laboratory_code {code!r} is already in the training data.",
            })
            continue

        try:
            built = build_training_row(raw)
        except KeyError as error:
            rejected.append({
                "row_number": row_number, "laboratory_code": code,
                "reason": f"Unrecognized age_band for this crop: {error}",
            })
            continue
        except ValueError as error:
            rejected.append({
                "row_number": row_number, "laboratory_code": code, "reason": str(error),
            })
            continue

        seen_in_file.add(code)
        accepted.append({
            "row_number": row_number,
            "laboratory_code": code,
            "raw": dict(raw),
            "derived": {field: built[field] for field in _REVIEW_FIELDS},
        })

    return ValidationResult(accepted=accepted, rejected=rejected)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_training_upload.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Stop `.gitignore` from swallowing future test fixtures**

In `.gitignore`, replace this block:

```
# Generated data — reproducible from the seeded generator, not committed
*.csv
```

with:

```
# Generated data — reproducible from the seeded generator, not committed.
# Test fixtures are exempt: they're small, hand-written, and part of the
# suite's meaning, so a blanket *.csv would silently drop them.
*.csv
!tests/fixtures/*.csv
```

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add data/training_upload.py tests/test_training_upload.py .gitignore
git commit -m "feat: add upload-template parser and per-row validator"
```

---

### Task 4: Model version registry (Supabase table + accessor)

**Files:**
- Create: `supabase/migrations/20260818_ml_model_versions.sql`
- Create: `ml/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `data.ingest_to_supabase.get_client`
- Produces:
  - `ml.registry.TABLE: str` (`"ml_model_versions"`)
  - `ml.registry.record_version(version: str, trained_at: str, row_count: int, metrics: dict, client=None) -> None`
  - `ml.registry.list_versions(client=None) -> list[dict]` — newest first
  - `ml.registry.current_version(client=None) -> str | None`
  - `ml.registry.set_current(version: str, promoted_by: str | None = None, client=None) -> None`
  - `ml.registry.get_version(version: str, client=None) -> dict | None`

- [ ] **Step 1: Write the migration**

Create `supabase/migrations/20260818_ml_model_versions.sql`:

```sql
-- Model version registry for the admin retraining pipeline
-- (docs/superpowers/specs/2026-08-18-training-data-pipeline-design.md §5).
--
-- Replaces main.py's previous implicit "whichever artifact directory name
-- sorts latest" rule with an explicit, auditable record of which version is
-- live, when it was trained, how it scored, and who promoted it.

CREATE TABLE public.ml_model_versions (
  version       TEXT PRIMARY KEY,
  trained_at    TIMESTAMPTZ NOT NULL,
  row_count     INTEGER NOT NULL,
  metrics       JSONB NOT NULL,
  promoted_at   TIMESTAMPTZ,
  promoted_by   TEXT,
  is_current    BOOLEAN NOT NULL DEFAULT FALSE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- At most one live version at a time. A partial unique index (rather than a
-- CHECK) is what makes "exactly one true" enforceable at the database level.
CREATE UNIQUE INDEX idx_ml_model_versions_single_current
  ON public.ml_model_versions(is_current)
  WHERE is_current;

CREATE INDEX idx_ml_model_versions_trained_at
  ON public.ml_model_versions(trained_at DESC);

ALTER TABLE public.ml_model_versions ENABLE ROW LEVEL SECURITY;

-- Same admin-only policy shape the sibling ML tables already use
-- (supabase/migrations/20260803_ml_fertilizer_tables.sql). The service role
-- bypasses RLS entirely, which is how the ML backend itself reads and writes
-- this table; this policy is for direct dashboard/SQL access by humans.
CREATE POLICY "Admin access to ml_model_versions" ON public.ml_model_versions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE profiles.id = auth.uid()
        AND profiles.role IN ('admin', 'superadmin')
    )
  );

-- Private bucket holding a copy of every trained artifact directory, so a
-- rebuilt EC2 instance can recover any version rather than only the one
-- pinned into git (see .gitignore's ml/artifacts/* exception).
INSERT INTO storage.buckets (id, name, public)
VALUES ('ml-model-artifacts', 'ml-model-artifacts', FALSE)
ON CONFLICT (id) DO NOTHING;
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_registry.py`:

```python
"""Registry accessors, exercised against a fake Supabase client.

These assert the call shapes (which table, which filters, which payload) --
the real table's behaviour is the migration's job, not this suite's."""

import pytest

from ml import registry


class FakeQuery:
    def __init__(self, log, action, payload=None):
        self.log = log
        self.action = action
        self.payload = payload
        self.filters = []
        self.ordering = None
        self.rows = []

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def neq(self, column, value):
        self.filters.append(("neq:" + column, value))
        return self

    def order(self, column, desc=False):
        self.ordering = (column, desc)
        return self

    def limit(self, count):
        return self

    def execute(self):
        self.log.append({
            "action": self.action, "payload": self.payload,
            "filters": self.filters, "ordering": self.ordering,
        })
        return type("R", (), {"data": self.rows})()


class FakeTable:
    def __init__(self, log, rows):
        self.log = log
        self.rows = rows

    def _query(self, action, payload=None):
        query = FakeQuery(self.log, action, payload)
        query.rows = self.rows
        return query

    def upsert(self, payload, **_kwargs):
        return self._query("upsert", payload)

    def update(self, payload):
        return self._query("update", payload)

    def select(self, _columns):
        return self._query("select")


class FakeClient:
    def __init__(self, rows=None):
        self.log = []
        self.rows = rows or []
        self.table_names = []

    def table(self, name):
        self.table_names.append(name)
        return FakeTable(self.log, self.rows)


class TestRecordVersion:
    def test_upserts_the_version_row(self):
        client = FakeClient()
        registry.record_version(
            "2026-08-18-101112", "2026-08-18T10:11:12+00:00", 3076, {"overall": {}},
            client=client,
        )
        assert client.table_names == [registry.TABLE]
        entry = client.log[0]
        assert entry["action"] == "upsert"
        assert entry["payload"]["version"] == "2026-08-18-101112"
        assert entry["payload"]["row_count"] == 3076
        assert entry["payload"]["is_current"] is False


class TestCurrentVersion:
    def test_returns_the_flagged_version(self):
        client = FakeClient(rows=[{"version": "2026-08-18-101112"}])
        assert registry.current_version(client=client) == "2026-08-18-101112"
        assert client.log[0]["filters"] == [("is_current", True)]

    def test_returns_none_when_nothing_is_flagged(self):
        assert registry.current_version(client=FakeClient(rows=[])) is None

    def test_returns_none_when_supabase_is_unreachable(self):
        class BrokenClient:
            def table(self, _name):
                raise RuntimeError("network down")

        assert registry.current_version(client=BrokenClient()) is None


class TestSetCurrent:
    def test_clears_the_old_flag_before_setting_the_new_one(self):
        client = FakeClient()
        registry.set_current("2026-08-18-101112", promoted_by="admin@suri", client=client)

        clear, promote = client.log[0], client.log[1]
        assert clear["action"] == "update"
        assert clear["payload"] == {"is_current": False}
        assert promote["action"] == "update"
        assert promote["payload"]["is_current"] is True
        assert promote["payload"]["promoted_by"] == "admin@suri"
        assert ("version", "2026-08-18-101112") in promote["filters"]


class TestListVersions:
    def test_orders_newest_first(self):
        client = FakeClient(rows=[{"version": "a"}])
        assert registry.list_versions(client=client) == [{"version": "a"}]
        assert client.log[0]["ordering"] == ("trained_at", True)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.registry'`

- [ ] **Step 4: Implement the registry**

Create `ml/registry.py`:

```python
"""Supabase-backed record of every trained model version and which one is live.

main.py previously inferred the live model from directory-name sort order.
That works only while "newest on disk" and "the one an admin approved" are the
same thing -- which stops being true the moment rollback exists. This table is
the authority; the on-disk directories are just storage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("suri-ml")

TABLE = "ml_model_versions"


def _client(client):
    if client is not None:
        return client
    from data.ingest_to_supabase import get_client
    return get_client()


def record_version(
    version: str, trained_at: str, row_count: int, metrics: dict, client=None
) -> None:
    """Register a freshly trained version. Never marks it current -- promotion
    is a separate, explicitly-approved action."""
    _client(client).table(TABLE).upsert({
        "version": version,
        "trained_at": trained_at,
        "row_count": row_count,
        "metrics": metrics,
        "is_current": False,
    }, on_conflict="version").execute()


def list_versions(client=None) -> list[dict]:
    try:
        return _client(client).table(TABLE).select("*").order(
            "trained_at", desc=True
        ).execute().data or []
    except Exception as error:
        logger.warning("Could not list model versions: %s", error)
        return []


def get_version(version: str, client=None) -> dict | None:
    try:
        rows = _client(client).table(TABLE).select("*").eq(
            "version", version
        ).limit(1).execute().data or []
    except Exception as error:
        logger.warning("Could not read model version %s: %s", version, error)
        return None
    return rows[0] if rows else None


def current_version(client=None) -> str | None:
    """None means "the registry could not answer" -- unreachable Supabase, an
    empty table, or nothing promoted yet. Callers fall back to the newest
    artifact on disk, so a registry outage degrades to the old behaviour
    instead of refusing to boot."""
    try:
        rows = _client(client).table(TABLE).select("version").eq(
            "is_current", True
        ).limit(1).execute().data or []
    except Exception as error:
        logger.warning("Could not read the current model version: %s", error)
        return None
    return rows[0]["version"] if rows else None


def set_current(version: str, promoted_by: str | None = None, client=None) -> None:
    """Clear-then-set, in that order: a partial unique index allows at most one
    is_current row, so setting before clearing would violate it. The brief
    window where no row is current is safe -- boot-time fallback covers it."""
    handle = _client(client)
    handle.table(TABLE).update({"is_current": False}).eq("is_current", True).execute()
    handle.table(TABLE).update({
        "is_current": True,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_by": promoted_by,
    }).eq("version", version).execute()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS

- [ ] **Step 6: Apply the migration**

Open the Supabase SQL editor for the shared project and run the full contents of `supabase/migrations/20260818_ml_model_versions.sql`. Confirm with:

```sql
SELECT COUNT(*) FROM public.ml_model_versions;
SELECT id, public FROM storage.buckets WHERE id = 'ml-model-artifacts';
```

Expected: `0` rows in the table; one bucket row with `public = false`.

- [ ] **Step 7: Commit**

```bash
git add supabase/migrations/20260818_ml_model_versions.sql ml/registry.py tests/test_registry.py
git commit -m "feat: add the ml_model_versions registry and artifact bucket"
```

---

### Task 5: Thread-safe model state and the main.py hot-swap refactor

**The bug this fixes:** `main.py` binds `ARTIFACTS` once at import and reads it at four separate points inside one `/predict` call (`ARTIFACTS["preprocessor"]`, `ARTIFACTS["regressor"]`, `ARTIFACTS["metadata"]` inside `_compute_confidence`, and `ARTIFACTS["version"]` when building the response). Rebinding that global to promote a new model would let a single response mix two versions — predicting with the old regressor while reporting the new `model_version`. Every request must take exactly one snapshot.

**Files:**
- Create: `ml/model_state.py`
- Modify: `main.py`
- Test: `tests/test_model_state.py`

**Interfaces:**
- Consumes: `ml.registry.current_version`, `ml.train.ARTIFACT_ROOT`
- Produces:
  - `ml.model_state.load_bundle(version: str, artifact_root: str) -> dict` — keys `version`, `preprocessor`, `regressor`, `metadata`
  - `ml.model_state.set_bundle(bundle: dict) -> None`
  - `ml.model_state.snapshot() -> dict` — raises `RuntimeError` when nothing is loaded
  - `ml.model_state.available_versions(artifact_root: str) -> list[str]` — sorted ascending
  - `ml.model_state.verify_bundle(bundle: dict) -> None` — raises `ValueError` if the bundle can't serve

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model_state.py`:

```python
import threading

import pytest

from ml import model_state


class TestSnapshotAtomicity:
    def test_snapshot_raises_before_anything_is_loaded(self, monkeypatch):
        monkeypatch.setattr(model_state, "_current", None)
        with pytest.raises(RuntimeError, match="No model artifacts loaded"):
            model_state.snapshot()

    def test_snapshot_returns_the_bundle_that_was_set(self, monkeypatch):
        monkeypatch.setattr(model_state, "_current", None)
        bundle = {"version": "v1", "preprocessor": object(),
                  "regressor": object(), "metadata": {}}
        model_state.set_bundle(bundle)
        assert model_state.snapshot()["version"] == "v1"

    def test_a_snapshot_is_never_half_swapped(self, monkeypatch):
        """The whole point of the refactor: a reader must see one coherent
        version, never the old regressor paired with the new version string."""
        monkeypatch.setattr(model_state, "_current", None)
        model_state.set_bundle({"version": "v1", "preprocessor": "p1",
                                "regressor": "r1", "metadata": {}})

        seen = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                snapshot = model_state.snapshot()
                seen.append((snapshot["version"], snapshot["preprocessor"],
                             snapshot["regressor"]))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        for index in range(2, 60):
            model_state.set_bundle({
                "version": f"v{index}", "preprocessor": f"p{index}",
                "regressor": f"r{index}", "metadata": {},
            })
        stop.set()
        thread.join(timeout=5)

        for version, preprocessor, regressor in seen:
            suffix = version[1:]
            assert preprocessor == f"p{suffix}"
            assert regressor == f"r{suffix}"


class TestAvailableVersions:
    def test_lists_only_directories_sorted_ascending(self, tmp_path):
        (tmp_path / "2026-08-01-000000").mkdir()
        (tmp_path / "2026-08-03-000000").mkdir()
        (tmp_path / "2026-08-02-000000").mkdir()
        (tmp_path / "notes.txt").write_text("ignored")

        assert model_state.available_versions(str(tmp_path)) == [
            "2026-08-01-000000", "2026-08-02-000000", "2026-08-03-000000",
        ]

    def test_missing_root_returns_empty(self, tmp_path):
        assert model_state.available_versions(str(tmp_path / "nope")) == []


class TestVerifyBundle:
    def test_rejects_a_bundle_whose_preprocessor_cannot_transform(self):
        class BrokenPreprocessor:
            def transform(self, _frame):
                raise ValueError("categories changed")

        bundle = {"version": "v1", "preprocessor": BrokenPreprocessor(),
                  "regressor": object(), "metadata": {}}
        with pytest.raises(ValueError, match="cannot serve"):
            model_state.verify_bundle(bundle)

    def test_accepts_the_real_currently_loaded_bundle(self):
        model_state.verify_bundle(model_state.snapshot())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_model_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.model_state'`

- [ ] **Step 3: Implement `ml/model_state.py`**

Create `ml/model_state.py`:

```python
"""Holds the artifact bundle /predict is currently serving, swappable at runtime.

The bundle dict is treated as immutable: promoting rebinds the module-level
reference to a brand-new dict rather than mutating the live one, and readers
take exactly one snapshot per request. That combination is what makes a
promotion invisible to an in-flight request instead of visible halfway
through it.
"""

from __future__ import annotations

import json
import os
import threading

import joblib
import pandas as pd

_lock = threading.Lock()
_current: dict | None = None


def load_bundle(version: str, artifact_root: str) -> dict:
    directory = os.path.join(artifact_root, version)
    with open(os.path.join(directory, "metadata.json")) as handle:
        metadata = json.load(handle)
    return {
        "version": version,
        "preprocessor": joblib.load(os.path.join(directory, "preprocessor.joblib")),
        "regressor": joblib.load(os.path.join(directory, "regressor.joblib")),
        "metadata": metadata,
    }


def set_bundle(bundle: dict) -> None:
    global _current
    with _lock:
        _current = bundle


def snapshot() -> dict:
    with _lock:
        if _current is None:
            raise RuntimeError("No model artifacts loaded")
        return _current


def available_versions(artifact_root: str) -> list[str]:
    if not os.path.isdir(artifact_root):
        return []
    return sorted(
        entry for entry in os.listdir(artifact_root)
        if os.path.isdir(os.path.join(artifact_root, entry))
    )


def _probe_frame() -> pd.DataFrame:
    """One syntactically valid row, built from the first supported crop, used
    only to prove a candidate bundle can actually transform input before it is
    allowed to serve traffic."""
    from data.nutrient_recommendation_guide import (
        CROPS, resolve_age_band, resolve_cultivation_method, resolve_variety,
    )
    from ml.preprocessing import FEATURE_COLUMNS
    from ml.remarks import derive_k_remark, derive_om_remark, derive_p_remark, derive_ph_remark

    crop = CROPS[0]
    row = {
        "ph": 6.5, "organic_matter_pct": 2.5, "total_nitrogen_pct": 0.125,
        "phosphorus_ppm": 15.0, "potassium_cmol_kg": 0.3,
        "temp": None, "moist": None, "ec": None, "sal": None,
        "ph_remark": derive_ph_remark(6.5),
        "om_remark": derive_om_remark(2.5),
        "p_remark": derive_p_remark(15.0),
        "k_remark": derive_k_remark(117.3),
        "crop": crop,
        "variety": resolve_variety(crop, None),
        "cultivation_method": resolve_cultivation_method(crop, None),
        "age_band": resolve_age_band(crop, None),
        "data_source": "lab",
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


def verify_bundle(bundle: dict) -> None:
    """Promotion gate. An artifact trained against a different feature vocabulary
    (e.g. before the supported-crop list changed) loads fine from joblib but
    raises the moment it sees a real row -- which would turn every /predict
    into a 422 if it were promoted. Catch that here, before the swap."""
    try:
        transformed = bundle["preprocessor"].transform(_probe_frame())
        bundle["regressor"].predict(transformed)
    except Exception as error:
        raise ValueError(
            f"Artifact {bundle.get('version')!r} cannot serve requests: {error}"
        ) from error
```

- [ ] **Step 4: Refactor `main.py` onto the new state module**

In `main.py`, replace the import of `ARTIFACT_ROOT` and the artifact-loading block. Delete `latest_version()`, `load_artifacts()`, and `ARTIFACTS = load_artifacts()`, and put this in their place:

```python
from ml import model_state, registry
from ml.train import ARTIFACT_ROOT


def boot_version() -> str:
    """The registry is the authority on which version is live; the newest
    directory on disk is the fallback so a Supabase outage can't stop the
    service from booting. A registry version that isn't on disk (e.g. this
    instance was rebuilt) also falls back rather than crashing."""
    on_disk = model_state.available_versions(ARTIFACT_ROOT)
    if not on_disk:
        raise RuntimeError(
            f"No artifact directory at {ARTIFACT_ROOT}. Run: python -m ml.train --csv <path>"
        )

    promoted = registry.current_version()
    if promoted and promoted in on_disk:
        return promoted
    if promoted:
        logger.warning(
            "Registry says %s is current but it is not on disk; falling back to %s",
            promoted, on_disk[-1],
        )
    return on_disk[-1]


model_state.set_bundle(model_state.load_bundle(boot_version(), ARTIFACT_ROOT))
```

Change `_compute_confidence` to take the bundle instead of reading the global. Its signature and the two `ARTIFACTS` reads inside it become:

```python
def _compute_confidence(bundle: dict, crop: str, X) -> Confidence | None:
    crop_metrics = bundle["metadata"].get("metrics", {}).get("by_crop", {}).get(crop)
```

and

```python
    for voter in bundle["regressor"].models_:
```

Change `health()` to snapshot:

```python
@app.get("/health")
def health() -> dict:
    supabase_reachable = True
    try:
        from data.ingest_to_supabase import get_client
        get_client()
    except Exception:
        supabase_reachable = False

    bundle = model_state.snapshot()
    return {
        "status": "ok",
        "model_version": bundle["version"],
        "artifacts_loaded": True,
        "supabase_reachable": supabase_reachable,
    }
```

Remove the now-unused `json` and `joblib` imports from `main.py`'s import block — `load_bundle` owns both concerns now, and leaving them would fail a lint pass.

In `predict()`, take one snapshot at the top and use it everywhere:

```python
def predict(request: SoilTestRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    # One snapshot for the whole request: promotion rebinds the module's
    # bundle reference, and reading it more than once could otherwise pair an
    # old regressor's numbers with a new version string in the same response.
    bundle = model_state.snapshot()
    features = _build_feature_row(request)
```

then in the `else` branch:

```python
        try:
            X = bundle["preprocessor"].transform(features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))
        n, p2o5, k2o = bundle["regressor"].predict(X)[0]
        n, p2o5, k2o = float(n), float(p2o5), float(k2o)
        source = "ml-ensemble"
        confidence = _compute_confidence(bundle, request.crop, X)
```

and in the response construction:

```python
        model_version=bundle["version"],
```

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: PASS, including the pre-existing `tests/test_api.py` (which never referenced `ARTIFACTS` directly — verified — so nothing outside `main.py` needs updating).

- [ ] **Step 6: Commit**

```bash
git add ml/model_state.py main.py tests/test_model_state.py
git commit -m "refactor: make the loaded model swappable and snapshot it per request"
```

---

### Task 6: Back trained artifacts up to Supabase Storage

**Files:**
- Create: `ml/artifacts_store.py`
- Test: `tests/test_artifacts_store.py`

**Interfaces:**
- Consumes: `data.ingest_to_supabase.get_client`
- Produces:
  - `ml.artifacts_store.BUCKET: str` (`"ml-model-artifacts"`)
  - `ml.artifacts_store.ARTIFACT_FILES: tuple[str, ...]`
  - `ml.artifacts_store.upload_version(version: str, artifact_root: str, client=None) -> bool`
  - `ml.artifacts_store.download_version(version: str, artifact_root: str, client=None) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_artifacts_store.py`:

```python
import json

from ml import artifacts_store


class FakeBucket:
    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail

    def upload(self, path, data, file_options=None):
        if self.fail:
            raise RuntimeError("storage unavailable")
        self.store[path] = data

    def download(self, path):
        return self.store[path]


class FakeStorage:
    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail
        self.buckets = []

    def from_(self, bucket):
        self.buckets.append(bucket)
        return FakeBucket(self.store, self.fail)


class FakeClient:
    def __init__(self, store=None, fail=False):
        self.storage = FakeStorage(store if store is not None else {}, fail)


def write_artifacts(root, version):
    directory = root / version
    directory.mkdir(parents=True)
    (directory / "preprocessor.joblib").write_bytes(b"pre")
    (directory / "regressor.joblib").write_bytes(b"reg")
    (directory / "metadata.json").write_text(json.dumps({"version": version}))
    return directory


class TestUploadVersion:
    def test_uploads_every_artifact_file_under_the_version_prefix(self, tmp_path):
        write_artifacts(tmp_path, "2026-08-18-101112")
        client = FakeClient()

        assert artifacts_store.upload_version(
            "2026-08-18-101112", str(tmp_path), client=client
        ) is True
        assert sorted(client.storage.store) == [
            "2026-08-18-101112/metadata.json",
            "2026-08-18-101112/preprocessor.joblib",
            "2026-08-18-101112/regressor.joblib",
        ]
        assert client.storage.buckets[0] == artifacts_store.BUCKET

    def test_a_storage_outage_is_reported_not_raised(self, tmp_path):
        """The trained artifact is already on local disk and usable; losing the
        off-instance backup must not fail the retrain."""
        write_artifacts(tmp_path, "2026-08-18-101112")
        assert artifacts_store.upload_version(
            "2026-08-18-101112", str(tmp_path), client=FakeClient(fail=True)
        ) is False


class TestDownloadVersion:
    def test_restores_files_to_disk(self, tmp_path):
        store = {
            "v1/preprocessor.joblib": b"pre",
            "v1/regressor.joblib": b"reg",
            "v1/metadata.json": b'{"version": "v1"}',
        }
        assert artifacts_store.download_version(
            "v1", str(tmp_path), client=FakeClient(store=store)
        ) is True
        assert (tmp_path / "v1" / "regressor.joblib").read_bytes() == b"reg"

    def test_missing_version_returns_false(self, tmp_path):
        assert artifacts_store.download_version(
            "nope", str(tmp_path), client=FakeClient(store={})
        ) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_artifacts_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.artifacts_store'`

- [ ] **Step 3: Implement it**

Create `ml/artifacts_store.py`:

```python
"""Copy trained artifact directories to a private Supabase Storage bucket.

Before this existed, the only artifacts that survived an EC2 rebuild were the
ones someone remembered to git-commit (see .gitignore's single pinned
exception). Every trained version now gets an off-instance copy, so rollback
targets stay recoverable.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("suri-ml")

BUCKET = "ml-model-artifacts"
ARTIFACT_FILES = ("preprocessor.joblib", "regressor.joblib", "metadata.json")


def _client(client):
    if client is not None:
        return client
    from data.ingest_to_supabase import get_client
    return get_client()


def upload_version(version: str, artifact_root: str, client=None) -> bool:
    """Returns False (never raises) when the backup fails -- the caller's
    artifact is already on local disk and fully usable without it."""
    directory = os.path.join(artifact_root, version)
    try:
        bucket = _client(client).storage.from_(BUCKET)
        for name in ARTIFACT_FILES:
            with open(os.path.join(directory, name), "rb") as handle:
                bucket.upload(
                    f"{version}/{name}",
                    handle.read(),
                    file_options={"upsert": "true", "content-type": "application/octet-stream"},
                )
        return True
    except Exception as error:
        logger.warning("Artifact backup for %s failed: %s", version, error)
        return False


def download_version(version: str, artifact_root: str, client=None) -> bool:
    directory = os.path.join(artifact_root, version)
    try:
        bucket = _client(client).storage.from_(BUCKET)
        os.makedirs(directory, exist_ok=True)
        for name in ARTIFACT_FILES:
            payload = bucket.download(f"{version}/{name}")
            with open(os.path.join(directory, name), "wb") as handle:
                handle.write(payload)
        return True
    except Exception as error:
        logger.warning("Artifact restore for %s failed: %s", version, error)
        return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_artifacts_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/artifacts_store.py tests/test_artifacts_store.py
git commit -m "feat: back trained artifacts up to Supabase Storage"
```

---

### Task 7: Background retrain job

**Files:**
- Create: `ml/retrain_job.py`
- Test: `tests/test_retrain_job.py`

**Interfaces:**
- Consumes: `ml.train.{train, _load_frame, ARTIFACT_ROOT}`, `ml.registry.record_version`, `ml.artifacts_store.upload_version`, `ml.model_state.snapshot`
- Produces:
  - `ml.retrain_job.start_job() -> str` (job id)
  - `ml.retrain_job.get_job(job_id: str) -> dict | None` — `{status, version?, metrics?, comparison?, backed_up?, error?, started_at, finished_at?}`
  - `ml.retrain_job.active_job_id() -> str | None`
  - `ml.retrain_job.build_comparison(new_metrics: dict, current_metrics: dict | None) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retrain_job.py`:

```python
import time

import pytest

from ml import retrain_job


@pytest.fixture(autouse=True)
def _clean_jobs(monkeypatch):
    monkeypatch.setattr(retrain_job, "_JOBS", {})
    monkeypatch.setattr(retrain_job, "_ACTIVE", None)


def wait_for(job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = retrain_job.get_job(job_id)
        if job and job["status"] in {"done", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish in time")


METRICS = {
    "overall": {
        "target_n": {"mae": 5.0, "r2": 0.8, "dummy_mae": 20.0},
        "target_p2o5": {"mae": 4.0, "r2": 0.7, "dummy_mae": 15.0},
        "target_k2o": {"mae": 6.0, "r2": 0.6, "dummy_mae": 18.0},
    }
}


class TestBuildComparison:
    def test_flags_every_target_that_beats_the_mean_baseline(self):
        comparison = retrain_job.build_comparison(METRICS, None)
        assert comparison["beats_baseline"] is True
        assert comparison["targets"]["target_n"]["beats_baseline"] is True

    def test_flags_failure_to_beat_the_baseline(self):
        weak = {"overall": {"target_n": {"mae": 25.0, "r2": 0.1, "dummy_mae": 20.0}}}
        comparison = retrain_job.build_comparison(weak, None)
        assert comparison["beats_baseline"] is False
        assert comparison["targets"]["target_n"]["beats_baseline"] is False

    def test_reports_improvement_against_the_live_model(self):
        current = {
            "overall": {
                "target_n": {"mae": 7.0, "r2": 0.7, "dummy_mae": 20.0},
                "target_p2o5": {"mae": 3.0, "r2": 0.75, "dummy_mae": 15.0},
                "target_k2o": {"mae": 6.0, "r2": 0.6, "dummy_mae": 18.0},
            }
        }
        comparison = retrain_job.build_comparison(METRICS, current)

        assert comparison["targets"]["target_n"]["mae_delta"] == pytest.approx(-2.0)
        assert comparison["targets"]["target_n"]["improved"] is True
        assert comparison["targets"]["target_p2o5"]["improved"] is False
        assert comparison["regressed"] is True

    def test_no_current_metrics_means_no_regression_claim(self):
        comparison = retrain_job.build_comparison(METRICS, None)
        assert comparison["regressed"] is False
        assert comparison["targets"]["target_n"]["mae_delta"] is None


class TestJobLifecycle:
    def test_successful_run_records_version_metrics_and_backup(self, monkeypatch):
        calls = {}

        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train", lambda frame: {
            "version": "2026-08-18-101112", "trained_at": "2026-08-18T10:11:12+00:00",
            "n_rows": 3076, "metrics": METRICS,
        })
        monkeypatch.setattr(retrain_job, "_record_version",
                            lambda **kwargs: calls.setdefault("recorded", kwargs))
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: True)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        job = wait_for(retrain_job.start_job())

        assert job["status"] == "done"
        assert job["version"] == "2026-08-18-101112"
        assert job["backed_up"] is True
        assert job["comparison"]["beats_baseline"] is True
        assert calls["recorded"]["row_count"] == 3076

    def test_training_failure_is_captured_not_raised(self, monkeypatch):
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")

        def explode(_frame):
            raise RuntimeError("not enough rows")

        monkeypatch.setattr(retrain_job, "_train", explode)

        job = wait_for(retrain_job.start_job())
        assert job["status"] == "failed"
        assert "not enough rows" in job["error"]

    def test_a_failed_backup_still_completes_the_job(self, monkeypatch):
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train", lambda frame: {
            "version": "v1", "trained_at": "t", "n_rows": 10, "metrics": METRICS,
        })
        monkeypatch.setattr(retrain_job, "_record_version", lambda **kwargs: None)
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: False)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        job = wait_for(retrain_job.start_job())
        assert job["status"] == "done"
        assert job["backed_up"] is False

    def test_refuses_a_second_concurrent_run(self, monkeypatch):
        release = __import__("threading").Event()
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train",
                            lambda frame: (release.wait(5), {
                                "version": "v1", "trained_at": "t",
                                "n_rows": 1, "metrics": METRICS,
                            })[1])
        monkeypatch.setattr(retrain_job, "_record_version", lambda **kwargs: None)
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: True)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        first = retrain_job.start_job()
        with pytest.raises(retrain_job.RetrainInProgress):
            retrain_job.start_job()
        release.set()
        wait_for(first)

    def test_unknown_job_id_returns_none(self):
        assert retrain_job.get_job("nope") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_retrain_job.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.retrain_job'`

- [ ] **Step 3: Implement it**

Create `ml/retrain_job.py`:

```python
"""Runs a retrain off the request thread and exposes its progress by job id.

Training never touches the bundle /predict is serving: it reads rows from
Supabase and writes a brand-new artifact directory. The live model changes
only when an admin promotes the result, which is a separate endpoint.

Job state lives in a module-level dict, which is correct precisely because
the service runs a single uvicorn worker (see the deploy unit file). The
indirection seams (_train, _record_version, ...) exist so the lifecycle can
be tested without training a real model or reaching Supabase.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("suri-ml")

_JOBS: dict[str, dict] = {}
_ACTIVE: str | None = None
_lock = threading.Lock()

TARGETS = ("target_n", "target_p2o5", "target_k2o")


class RetrainInProgress(RuntimeError):
    """A retrain is already running. Two concurrent runs on a 1 GiB instance
    would compete for memory and produce two artifacts racing to be newest,
    for no benefit."""


def _load_training_frame():
    from ml.train import _load_frame
    return _load_frame(None)


def _train(frame) -> dict:
    from ml.train import train
    return train(frame)


def _record_version(**kwargs) -> None:
    from ml import registry
    registry.record_version(**kwargs)


def _upload_version(version: str) -> bool:
    from ml import artifacts_store
    from ml.train import ARTIFACT_ROOT
    return artifacts_store.upload_version(version, ARTIFACT_ROOT)


def _current_metrics() -> dict | None:
    from ml import model_state
    try:
        return model_state.snapshot()["metadata"].get("metrics")
    except Exception:
        return None


def build_comparison(new_metrics: dict, current_metrics: dict | None) -> dict:
    """Everything the review screen needs to answer "is this better?":
    per-target MAE against the mean baseline, and against the live model."""
    new_overall = (new_metrics or {}).get("overall", {})
    current_overall = (current_metrics or {}).get("overall", {})

    targets: dict[str, dict] = {}
    beats_baseline = True
    regressed = False

    for name, scores in new_overall.items():
        mae = scores.get("mae")
        dummy_mae = scores.get("dummy_mae")
        target_beats = mae is not None and dummy_mae is not None and mae < dummy_mae
        beats_baseline = beats_baseline and target_beats

        current_mae = current_overall.get(name, {}).get("mae")
        delta = None if current_mae is None else round(mae - current_mae, 4)
        improved = None if delta is None else delta < 0
        if improved is False:
            regressed = True

        targets[name] = {
            "mae": mae,
            "r2": scores.get("r2"),
            "dummy_mae": dummy_mae,
            "beats_baseline": target_beats,
            "current_mae": current_mae,
            "mae_delta": delta,
            "improved": improved,
        }

    return {
        "targets": targets,
        "beats_baseline": beats_baseline,
        "regressed": regressed,
        "compared_against_live_model": bool(current_overall),
    }


def _run(job_id: str) -> None:
    global _ACTIVE
    try:
        current_metrics = _current_metrics()
        frame = _load_training_frame()
        metadata = _train(frame)

        _record_version(
            version=metadata["version"],
            trained_at=metadata["trained_at"],
            row_count=metadata["n_rows"],
            metrics=metadata["metrics"],
        )
        backed_up = _upload_version(metadata["version"])

        with _lock:
            _JOBS[job_id].update({
                "status": "done",
                "version": metadata["version"],
                "metrics": metadata["metrics"],
                "comparison": build_comparison(metadata["metrics"], current_metrics),
                "backed_up": backed_up,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
    except Exception as error:
        logger.exception("Retrain job %s failed", job_id)
        with _lock:
            _JOBS[job_id].update({
                "status": "failed",
                "error": str(error),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
    finally:
        with _lock:
            _ACTIVE = None


def start_job() -> str:
    global _ACTIVE
    with _lock:
        if _ACTIVE is not None:
            raise RetrainInProgress(f"Retrain {_ACTIVE} is already running.")
        job_id = uuid.uuid4().hex
        _ACTIVE = job_id
        _JOBS[job_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    threading.Thread(target=_run, args=(job_id,), daemon=True, name=f"retrain-{job_id}").start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def active_job_id() -> str | None:
    with _lock:
        return _ACTIVE
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_retrain_job.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/retrain_job.py tests/test_retrain_job.py
git commit -m "feat: add the background retrain job with baseline and live-model comparison"
```

---

### Task 8: The pipeline endpoints

**Files:**
- Modify: `main.py`
- Modify: `requirements.txt`
- Test: `tests/test_pipeline_api.py`

**Interfaces:**
- Consumes: everything produced by Tasks 2–7
- Produces the HTTP surface:
  - `POST /training-data/validate` (multipart, field `file`) → `{accepted, rejected, template_columns}`
  - `POST /training-data/commit` `{rows: [raw...]}` → `{committed, rejected}`
  - `POST /retrain` → `{job_id}` (409 when one is already running)
  - `GET /retrain/{job_id}` → job dict (404 unknown)
  - `GET /model/versions` → `{current, versions, on_disk}`
  - `POST /model/promote` `{version, promoted_by?}` → `{current, previous}`

**`python-multipart` is required** — FastAPI raises at import time on a route declaring `UploadFile` without it.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add below `httpx>=0.27.0`:

```
python-multipart>=0.0.9
```

Then install it: `pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_pipeline_api.py`:

```python
"""The upload/retrain/promote HTTP surface.

Supabase is never reached: every seam is monkeypatched. These tests assert
routing, status codes, and payload shape."""

import io

import pytest
from fastapi.testclient import TestClient

import main as service
from data.training_upload import TEMPLATE_COLUMNS
from ml import model_state, retrain_job


@pytest.fixture(scope="module")
def client():
    return TestClient(service.app)


@pytest.fixture(autouse=True)
def _no_supabase(monkeypatch):
    monkeypatch.setattr(service, "_existing_laboratory_codes", lambda: set())
    monkeypatch.setattr(service, "_commit_rows", lambda rows: len(rows))


def upload_csv(rows):
    lines = [",".join(TEMPLATE_COLUMNS)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in TEMPLATE_COLUMNS))
    return ("\n".join(lines) + "\n").encode()


VALID = {
    "laboratory_code": "2026-API1", "crop": "Corn", "ph": "6.4",
    "organic_matter_pct": "2.8", "phosphorus_ppm": "18.0", "potassium_cmol_kg": "0.35",
}


class TestValidateEndpoint:
    def test_accepts_a_good_file_and_writes_nothing(self, client):
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv([VALID])), "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["accepted"]) == 1
        assert body["rejected"] == []
        assert body["accepted"][0]["derived"]["target_unit"] == "kg_ha"

    def test_reports_bad_rows_without_failing_the_request(self, client):
        rows = [VALID, {**VALID, "laboratory_code": "2026-API2", "crop": "Papaya"}]
        response = client.post(
            "/training-data/validate",
            files={"file": ("upload.csv", io.BytesIO(upload_csv(rows)), "text/csv")},
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["accepted"]) == 1
        assert "Papaya" in body["rejected"][0]["reason"]

    def test_unsupported_file_type_is_a_422(self, client):
        response = client.post(
            "/training-data/validate",
            files={"file": ("notes.pdf", io.BytesIO(b"x"), "application/pdf")},
        )
        assert response.status_code == 422


class TestCommitEndpoint:
    def test_revalidates_and_commits_only_good_rows(self, client):
        response = client.post("/training-data/commit", json={
            "rows": [VALID, {**VALID, "laboratory_code": "2026-API3", "crop": "Nope"}],
        })
        assert response.status_code == 200
        body = response.json()
        assert body["committed"] == 1
        assert len(body["rejected"]) == 1

    def test_empty_payload_is_a_422(self, client):
        assert client.post("/training-data/commit", json={"rows": []}).status_code == 422


class TestRetrainEndpoints:
    def test_starts_a_job_and_reports_its_status(self, client, monkeypatch):
        monkeypatch.setattr(retrain_job, "_JOBS", {})
        monkeypatch.setattr(retrain_job, "_ACTIVE", None)
        monkeypatch.setattr(retrain_job, "_load_training_frame", lambda: "FRAME")
        monkeypatch.setattr(retrain_job, "_train", lambda frame: {
            "version": "v-test", "trained_at": "t", "n_rows": 5,
            "metrics": {"overall": {"target_n": {"mae": 1.0, "r2": 0.9, "dummy_mae": 9.0}}},
        })
        monkeypatch.setattr(retrain_job, "_record_version", lambda **kwargs: None)
        monkeypatch.setattr(retrain_job, "_upload_version", lambda version: True)
        monkeypatch.setattr(retrain_job, "_current_metrics", lambda: None)

        job_id = client.post("/retrain").json()["job_id"]

        import time
        for _ in range(500):
            body = client.get(f"/retrain/{job_id}").json()
            if body["status"] in {"done", "failed"}:
                break
            time.sleep(0.01)

        assert body["status"] == "done"
        assert body["version"] == "v-test"

    def test_unknown_job_is_a_404(self, client):
        assert client.get("/retrain/does-not-exist").status_code == 404

    def test_a_second_concurrent_retrain_is_a_409(self, client, monkeypatch):
        monkeypatch.setattr(retrain_job, "_ACTIVE", "already-running")
        assert client.post("/retrain").status_code == 409


class TestModelVersionEndpoints:
    def test_lists_versions_and_the_live_one(self, client, monkeypatch):
        monkeypatch.setattr(service.registry, "list_versions",
                            lambda: [{"version": "v1", "is_current": True}])
        body = client.get("/model/versions").json()
        assert body["current"] == model_state.snapshot()["version"]
        assert body["versions"][0]["version"] == "v1"

    def test_promoting_an_unknown_version_is_a_404(self, client):
        response = client.post("/model/promote", json={"version": "2099-01-01-000000"})
        assert response.status_code == 404

    def test_promoting_swaps_the_live_bundle(self, client, monkeypatch):
        live = model_state.snapshot()
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"], "candidate"])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})
        monkeypatch.setattr(service.model_state, "verify_bundle", lambda bundle: None)
        monkeypatch.setattr(service.registry, "set_current", lambda **kwargs: None)

        try:
            body = client.post("/model/promote", json={"version": "candidate"}).json()
            assert body["current"] == "candidate"
            assert model_state.snapshot()["version"] == "candidate"
        finally:
            model_state.set_bundle(live)

    def test_an_unservable_artifact_is_refused_and_the_live_model_kept(self, client, monkeypatch):
        live = model_state.snapshot()
        monkeypatch.setattr(service.model_state, "available_versions",
                            lambda root: [live["version"], "broken"])
        monkeypatch.setattr(service.model_state, "load_bundle",
                            lambda version, root: {**live, "version": version})

        def refuse(_bundle):
            raise ValueError("cannot serve requests: categories changed")

        monkeypatch.setattr(service.model_state, "verify_bundle", refuse)

        response = client.post("/model/promote", json={"version": "broken"})
        assert response.status_code == 422
        assert model_state.snapshot()["version"] == live["version"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pipeline_api.py -v`
Expected: FAIL — 404s on the new routes.

- [ ] **Step 4: Implement the endpoints**

In `main.py`, add to the imports:

```python
from fastapi import File, UploadFile
from data.ingest_to_supabase import TABLE as TRAINING_TABLE, ingest_records
from data.training_upload import (
    TEMPLATE_COLUMNS, UploadFormatError, build_training_row, parse_upload, validate_rows,
)
from ml import artifacts_store, retrain_job
```

Add these request models next to `SoilTestRequest`:

```python
class CommitRequest(BaseModel):
    rows: list[dict]


class PromoteRequest(BaseModel):
    version: str
    promoted_by: str | None = None
```

Add the Supabase seams (named so tests can patch them):

```python
def _existing_laboratory_codes() -> set[str]:
    """Every laboratory_code already in the corpus, so an upload can flag
    duplicates before writing anything. Read in pages for the same reason
    ml/train.py's read is paged -- PostgREST caps a response at 1000 rows."""
    from data.ingest_to_supabase import get_client
    from ml.train import SUPABASE_PAGE_SIZE

    client = get_client()
    codes: set[str] = set()
    start = 0
    while True:
        page = client.table(TRAINING_TABLE).select("laboratory_code").range(
            start, start + SUPABASE_PAGE_SIZE - 1
        ).execute().data or []
        codes.update(row["laboratory_code"] for row in page)
        if len(page) < SUPABASE_PAGE_SIZE:
            return codes
        start += SUPABASE_PAGE_SIZE


def _commit_rows(rows: list[dict]) -> int:
    return ingest_records(rows)
```

Add the endpoints:

```python
@app.post("/training-data/validate", dependencies=[Depends(require_api_key)])
async def validate_training_upload(file: UploadFile = File(...)) -> dict:
    """Dry run: parses, validates, and shows what every accepted row would
    contribute. Writes nothing."""
    content = await file.read()
    try:
        raw_rows = parse_upload(content, file.filename or "upload.csv")
    except UploadFormatError as error:
        raise HTTPException(status_code=422, detail=str(error))

    result = validate_rows(raw_rows, _existing_laboratory_codes())
    return {
        "template_columns": TEMPLATE_COLUMNS,
        "accepted": result.accepted,
        "rejected": result.rejected,
    }


@app.post("/training-data/commit", dependencies=[Depends(require_api_key)])
def commit_training_upload(request: CommitRequest) -> dict:
    """Re-validates from the raw rows rather than trusting anything derived
    client-side, then upserts only what passes."""
    if not request.rows:
        raise HTTPException(status_code=422, detail="No rows supplied.")

    result = validate_rows(request.rows, _existing_laboratory_codes())
    committed = _commit_rows([build_training_row(entry["raw"]) for entry in result.accepted])
    return {"committed": committed, "rejected": result.rejected}


@app.post("/retrain", dependencies=[Depends(require_api_key)])
def start_retrain() -> dict:
    try:
        return {"job_id": retrain_job.start_job()}
    except retrain_job.RetrainInProgress as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.get("/retrain/{job_id}", dependencies=[Depends(require_api_key)])
def retrain_status(job_id: str) -> dict:
    job = retrain_job.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No retrain job {job_id!r}.")
    return job


@app.get("/model/versions", dependencies=[Depends(require_api_key)])
def model_versions() -> dict:
    return {
        "current": model_state.snapshot()["version"],
        "versions": registry.list_versions(),
        "on_disk": model_state.available_versions(ARTIFACT_ROOT),
    }


@app.post("/model/promote", dependencies=[Depends(require_api_key)])
def promote_model(request: PromoteRequest) -> dict:
    previous = model_state.snapshot()["version"]

    if request.version not in model_state.available_versions(ARTIFACT_ROOT):
        # Not on this instance -- try the off-instance backup before giving up,
        # which is what makes rollback survive an EC2 rebuild.
        if not artifacts_store.download_version(request.version, ARTIFACT_ROOT):
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {request.version!r} is not on disk and could not be restored.",
            )

    try:
        bundle = model_state.load_bundle(request.version, ARTIFACT_ROOT)
        model_state.verify_bundle(bundle)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not load {request.version!r}: {error}")

    # Swap first, then record: if the registry write fails, the service is
    # already serving the version the admin approved, and the next boot's
    # fallback picks a valid version rather than a phantom one.
    model_state.set_bundle(bundle)
    try:
        registry.set_current(request.version, promoted_by=request.promoted_by)
    except Exception as error:
        logger.warning("Promoted %s in memory but could not record it: %s",
                       request.version, error)

    return {"current": request.version, "previous": previous}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipeline_api.py -v`
Expected: PASS

- [ ] **Step 6: Run the whole suite and commit**

Run: `python -m pytest`
Expected: PASS

```bash
git add main.py requirements.txt tests/test_pipeline_api.py
git commit -m "feat: add upload, retrain, and promote endpoints"
```

---

### Task 9: Deployment notes and the single-worker guard

**Files:**
- Modify: `deploy/suri-ml.service`
- Modify: `deploy/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing
- Produces: nothing (documentation and configuration)

- [ ] **Step 1: Document the single-worker requirement in the unit file**

In `deploy/suri-ml.service`, add above the `ExecStart` line:

```
# Exactly one worker, deliberately. The admin retraining pipeline promotes a
# new model by hot-swapping it in this process's memory
# (ml/model_state.py). Adding --workers N (N>1) would give each worker its
# own copy, so a promotion would update one of them and leave the rest
# serving the old model with no way to tell which answered a given request.
# Retrain job state (ml/retrain_job.py) is per-process for the same reason.
```

- [ ] **Step 2: Add a pipeline section to the deploy README**

Append to `deploy/README.md`:

```markdown
## 5. Retraining pipeline

Once `supabase/migrations/20260818_ml_model_versions.sql` is applied, admins
retrain from `suri`'s admin dashboard rather than over SSH. What changes
operationally:

- **Artifacts no longer need to be git-committed to go live.** `main.py` reads
  the live version from the `ml_model_versions` registry at boot and swaps it
  in memory on promotion. The pinned `ml/artifacts/<version>/` directory in
  `.gitignore` still matters as the bootstrap model for a *fresh* instance
  that has no local artifacts and no registry entry yet.
- **Artifacts trained on the instance survive deploys.** The Actions workflow
  runs `git reset --hard origin/main`, which does not delete untracked files,
  so locally trained artifact directories persist. They are also copied to the
  `ml-model-artifacts` Supabase Storage bucket, and `/model/promote` restores
  from there automatically when a requested version is missing from disk —
  which is what makes rollback work on a rebuilt instance.
- **`ML_API_KEY` now guards more than `/predict`.** Every pipeline endpoint
  requires it. `suri`'s server-side proxy sends it; nothing else should be able
  to trigger a retrain or a promotion.
- **Retraining runs on the instance while it serves traffic.** `/predict` is
  never paused. Before relying on this in a live demo, run one retrain while
  firing `/predict` calls and confirm latency stays acceptable on a
  `t3.micro` — if it doesn't, lower training's priority rather than blocking
  requests.
```

- [ ] **Step 3: Add the endpoints to the service README**

In `README.md`, after the `### GET /health` section, add:

```markdown
### Retraining pipeline

All of these require `X-API-Key` when `ML_API_KEY` is set, and are driven by
`suri`'s admin dashboard rather than called by hand:

| Endpoint | Purpose |
|---|---|
| `POST /training-data/validate` | Multipart upload; parses and validates a CSV/XLSX against the template, returns accepted rows (with the values the service derived) and rejected rows with reasons. Writes nothing. |
| `POST /training-data/commit` | Re-validates the raw rows and upserts what passes into `ml_soil_training_data`. |
| `POST /retrain` | Starts a background retrain; returns `{job_id}`. 409 if one is already running. |
| `GET /retrain/{job_id}` | `running` / `done` / `failed`, plus metrics and a comparison against the live model once done. |
| `GET /model/versions` | Every registered version, which one is live, and which are on local disk. |
| `POST /model/promote` | Hot-swaps the live model to `{version}` (restoring it from Supabase Storage first if it isn't on disk). Rollback is the same call with an older version. |

Upload template columns — raw measurements only, everything else is derived
server-side:

`laboratory_code`, `crop`, `ph`, `organic_matter_pct`, `phosphorus_ppm`,
`potassium_cmol_kg` (all required), plus optional `variety`,
`cultivation_method`, `age_band`, `farm_area_sqm`, `sampling_site`,
`date_of_sampling`.
```

- [ ] **Step 4: Commit**

```bash
git add deploy/suri-ml.service deploy/README.md README.md
git commit -m "docs: document the retraining pipeline and the single-worker requirement"
```

---

### Task 10: suri proxy route

**Files:**
- Create: `suri/app/api/admin/ml-training/[...path]/route.ts`

**Interfaces:**
- Consumes: `@/lib/auth/session.getSession`, `@/lib/auth/roles.canAccessAdminPanel`
- Produces: `GET`/`POST` handlers proxying `/<path>` to the ML backend with `X-API-Key`

**Why a catch-all:** six backend endpoints share one auth check, one base URL, and one error-shaping rule. Six near-identical route files would be six places to forget the admin check.

- [ ] **Step 1: Create the route**

Create `suri/app/api/admin/ml-training/[...path]/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { getSession } from "@/lib/auth/session";
import { canAccessAdminPanel } from "@/lib/auth/roles";

const ML_API_URL = process.env.ML_API_URL || "http://localhost:8001";
const ML_API_KEY = process.env.ML_API_KEY;

// Retraining is a background job on the ML side (POST /retrain returns a job
// id immediately and the UI polls), so no request proxied here should be
// long-running. Validation of a few thousand rows is the slowest, well under
// this ceiling.
const ML_REQUEST_TIMEOUT_MS = 60_000;

// Exactly the endpoints the admin UI drives. An allowlist rather than blind
// forwarding: this route runs with the backend's API key, so anything
// reachable through it is effectively unauthenticated-to-the-backend, and
// only the admin session check stands in front of it.
const ALLOWED: Record<string, readonly ("GET" | "POST")[]> = {
  "training-data/validate": ["POST"],
  "training-data/commit": ["POST"],
  retrain: ["POST"],
  "model/versions": ["GET"],
  "model/promote": ["POST"],
};

function isAllowed(path: string, method: "GET" | "POST"): boolean {
  if (ALLOWED[path]?.includes(method)) return true;
  // Job polling: GET retrain/<job_id>
  return method === "GET" && /^retrain\/[A-Za-z0-9-]+$/.test(path);
}

async function proxy(request: NextRequest, path: string, method: "GET" | "POST") {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (!canAccessAdminPanel(session.role)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  if (!isAllowed(path, method)) {
    return NextResponse.json({ error: `Unsupported path: ${path}` }, { status: 404 });
  }

  const headers: Record<string, string> = {};
  if (ML_API_KEY) headers["X-API-Key"] = ML_API_KEY;

  let body: BodyInit | undefined;
  if (method === "POST") {
    const contentType = request.headers.get("content-type") ?? "";
    if (contentType.includes("multipart/form-data")) {
      // Re-read as FormData and let fetch set its own boundary — forwarding
      // the original content-type header with a re-serialized body produces a
      // boundary mismatch the backend cannot parse.
      body = await request.formData();
    } else {
      body = await request.text();
      headers["Content-Type"] = "application/json";
    }
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), ML_REQUEST_TIMEOUT_MS);

  let backendResponse: Response;
  try {
    backendResponse = await fetch(`${ML_API_URL}/${path}`, {
      method,
      headers,
      body,
      signal: controller.signal,
    });
  } catch (err) {
    const isTimeout = err instanceof Error && err.name === "AbortError";
    console.error(`[ml-training] ${isTimeout ? "Timed out" : "Failed"} calling ${path}:`, err);
    return NextResponse.json(
      { error: isTimeout ? "ML backend timed out" : "ML backend is unreachable" },
      { status: 502 }
    );
  } finally {
    clearTimeout(timeoutId);
  }

  const text = await backendResponse.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    console.error("[ml-training] Backend returned non-JSON:", text.slice(0, 500));
    return NextResponse.json({ error: "ML backend returned an invalid response" }, { status: 502 });
  }

  return NextResponse.json(data, { status: backendResponse.status });
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxy(request, path.join("/"), "GET");
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  return proxy(request, path.join("/"), "POST");
}
```

- [ ] **Step 2: Verify it compiles**

Run from `suri/`: `npx tsc --noEmit`
Expected: no errors referencing `app/api/admin/ml-training`.

> If `context.params` is not a Promise in this Next.js version, check a sibling dynamic route (e.g. `app/api/admin/color-vectors/[id]/history/route.ts`) and match its signature exactly.

- [ ] **Step 3: Commit**

```bash
git add app/api/admin/ml-training
git commit -m "feat: add the admin ML-training proxy route"
```

---

### Task 11: suri admin UI

**Files:**
- Create: `suri/features/admin/components/ml-training/MLTrainingManagement.tsx`
- Create: `suri/features/admin/components/ml-training/index.ts`

**Interfaces:**
- Consumes: the proxy route from Task 10
- Produces: `MLTrainingManagement` — default-exported named component taking no required props (mounted by `AdminTabsHost`, which passes `userRole`/`userMunicipality` that this component ignores)

- [ ] **Step 1: Create the component**

Create `suri/features/admin/components/ml-training/MLTrainingManagement.tsx`:

```tsx
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Upload, Loader2, AlertCircle, CheckCircle2, Database, History, RefreshCw, Rocket,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

const BASE = "/api/admin/ml-training";
const POLL_INTERVAL_MS = 2000;

type RawRow = Record<string, string>;

interface AcceptedRow {
  row_number: number;
  laboratory_code: string;
  raw: RawRow;
  derived: Record<string, string | number>;
}

interface RejectedRow {
  row_number: number;
  laboratory_code: string | null;
  reason: string;
}

interface ValidationResponse {
  accepted: AcceptedRow[];
  rejected: RejectedRow[];
}

interface TargetComparison {
  mae: number;
  r2: number;
  dummy_mae: number;
  beats_baseline: boolean;
  current_mae: number | null;
  mae_delta: number | null;
  improved: boolean | null;
}

interface RetrainJob {
  status: "running" | "done" | "failed";
  version?: string;
  error?: string;
  backed_up?: boolean;
  comparison?: {
    targets: Record<string, TargetComparison>;
    beats_baseline: boolean;
    regressed: boolean;
    compared_against_live_model: boolean;
  };
}

interface ModelVersion {
  version: string;
  trained_at: string;
  row_count: number;
  promoted_at: string | null;
  promoted_by: string | null;
  is_current: boolean;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}/${path}`, init);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail || body?.error || `Request failed (${response.status})`);
  }
  return body as T;
}

export function MLTrainingManagement() {
  const [file, setFile] = useState<File | null>(null);
  const [validating, setValidating] = useState(false);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [committing, setCommitting] = useState(false);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<RetrainJob | null>(null);
  const [promoting, setPromoting] = useState(false);

  const [current, setCurrent] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);

  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadVersions = useCallback(async () => {
    try {
      const body = await call<{ current: string; versions: ModelVersion[] }>("model/versions");
      setCurrent(body.current);
      setVersions(body.versions ?? []);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not load model versions");
    }
  }, []);

  useEffect(() => {
    void loadVersions();
  }, [loadVersions]);

  // Poll the retrain job until it settles. Cleared on unmount so switching
  // admin tabs (which keeps this component mounted but hidden) never leaves a
  // timer running against a finished job.
  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;

    const tick = async () => {
      try {
        const body = await call<RetrainJob>(`retrain/${jobId}`);
        if (cancelled) return;
        setJob(body);
        if (body.status === "running") {
          pollTimer.current = setTimeout(tick, POLL_INTERVAL_MS);
        } else if (body.status === "done") {
          toast.success(`Training finished — version ${body.version}`);
          void loadVersions();
        } else {
          toast.error(`Training failed: ${body.error ?? "unknown error"}`);
        }
      } catch (error) {
        if (cancelled) return;
        toast.error(error instanceof Error ? error.message : "Lost track of the retrain job");
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, [jobId, loadVersions]);

  const handleValidate = async () => {
    if (!file) return;
    setValidating(true);
    setValidation(null);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const body = await call<ValidationResponse>("training-data/validate", {
        method: "POST",
        body: form,
      });
      setValidation(body);
      toast.success(`${body.accepted.length} row(s) ready, ${body.rejected.length} rejected`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Validation failed");
    } finally {
      setValidating(false);
    }
  };

  const handleCommit = async () => {
    if (!validation?.accepted.length) return;
    setCommitting(true);
    try {
      const body = await call<{ committed: number }>("training-data/commit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: validation.accepted.map((entry) => entry.raw) }),
      });
      toast.success(`Added ${body.committed} row(s) to the training data`);
      setValidation(null);
      setFile(null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save the rows");
    } finally {
      setCommitting(false);
    }
  };

  const handleRetrain = async () => {
    try {
      const body = await call<{ job_id: string }>("retrain", { method: "POST" });
      setJob({ status: "running" });
      setJobId(body.job_id);
      toast.info("Training started — predictions keep running while it works");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not start training");
    }
  };

  const promote = async (version: string, regressed: boolean) => {
    if (regressed && !window.confirm(
      `Version ${version} scores worse than the live model on at least one target. Promote it anyway?`
    )) return;

    setPromoting(true);
    try {
      await call("model/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version }),
      });
      toast.success(`Version ${version} is now live`);
      await loadVersions();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Promotion failed");
    } finally {
      setPromoting(false);
    }
  };

  return (
    <div className="flex-1 space-y-4 p-4 md:p-6">
      <div>
        <h2 className="text-2xl font-semibold tracking-tight">Model Training</h2>
        <p className="text-sm text-muted-foreground">
          Add new soil-test data, retrain the fertilizer model, and choose which version is live.
          Predictions keep serving normally throughout.
        </p>
      </div>

      <Tabs defaultValue="upload">
        <TabsList>
          <TabsTrigger value="upload">Upload Data</TabsTrigger>
          <TabsTrigger value="retrain">Retrain</TabsTrigger>
          <TabsTrigger value="history">Version History</TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Upload className="h-4 w-4" /> Upload soil-test data
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                CSV or Excel with these columns — required:{" "}
                <code className="text-xs">laboratory_code, crop, ph, organic_matter_pct,
                phosphorus_ppm, potassium_cmol_kg</code>; optional:{" "}
                <code className="text-xs">variety, cultivation_method, age_band, farm_area_sqm,
                sampling_site, date_of_sampling</code>. Nutrient targets, remarks, and
                formulations are calculated automatically — don&apos;t include them.
              </p>

              <div className="flex items-center gap-3">
                <input
                  type="file"
                  accept=".csv,.xlsx,.xlsm"
                  onChange={(event) => {
                    setFile(event.target.files?.[0] ?? null);
                    setValidation(null);
                  }}
                  className="text-sm file:mr-3 file:rounded-md file:border-0 file:bg-primary file:px-3 file:py-1.5 file:text-sm file:text-primary-foreground"
                />
                <Button onClick={handleValidate} disabled={!file || validating}>
                  {validating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                  Check file
                </Button>
              </div>

              {validation ? (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary">
                      <CheckCircle2 className="mr-1 h-3 w-3" />
                      {validation.accepted.length} ready
                    </Badge>
                    {validation.rejected.length > 0 ? (
                      <Badge variant="destructive">
                        <AlertCircle className="mr-1 h-3 w-3" />
                        {validation.rejected.length} rejected
                      </Badge>
                    ) : null}
                  </div>

                  {validation.rejected.length > 0 ? (
                    <div className="max-h-52 overflow-auto rounded-md border">
                      <table className="w-full text-sm">
                        <thead className="bg-muted/50">
                          <tr>
                            <th className="p-2 text-left font-medium">Row</th>
                            <th className="p-2 text-left font-medium">Lab code</th>
                            <th className="p-2 text-left font-medium">Why it was rejected</th>
                          </tr>
                        </thead>
                        <tbody>
                          {validation.rejected.map((row) => (
                            <tr key={`${row.row_number}`} className="border-t">
                              <td className="p-2">{row.row_number}</td>
                              <td className="p-2">{row.laboratory_code ?? "—"}</td>
                              <td className="p-2 text-destructive">{row.reason}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}

                  {validation.accepted.length > 0 ? (
                    <div className="max-h-72 overflow-auto rounded-md border">
                      <table className="w-full text-sm">
                        <thead className="bg-muted/50">
                          <tr>
                            <th className="p-2 text-left font-medium">Lab code</th>
                            <th className="p-2 text-left font-medium">N</th>
                            <th className="p-2 text-left font-medium">P₂O₅</th>
                            <th className="p-2 text-left font-medium">K₂O</th>
                            <th className="p-2 text-left font-medium">Unit</th>
                            <th className="p-2 text-left font-medium">Remarks (pH / OM / P / K)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {validation.accepted.map((row) => (
                            <tr key={row.laboratory_code} className="border-t">
                              <td className="p-2 font-mono text-xs">{row.laboratory_code}</td>
                              <td className="p-2">{row.derived.target_n}</td>
                              <td className="p-2">{row.derived.target_p2o5}</td>
                              <td className="p-2">{row.derived.target_k2o}</td>
                              <td className="p-2 text-muted-foreground">{row.derived.target_unit}</td>
                              <td className="p-2 text-xs text-muted-foreground">
                                {row.derived.ph_remark} / {row.derived.om_remark} /{" "}
                                {row.derived.p_remark} / {row.derived.k_remark}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}

                  <Button
                    onClick={handleCommit}
                    disabled={committing || validation.accepted.length === 0}
                  >
                    {committing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : (
                      <Database className="mr-2 h-4 w-4" />
                    )}
                    Add {validation.accepted.length} row(s) to training data
                  </Button>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="retrain" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <RefreshCw className="h-4 w-4" /> Retrain the model
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Live model: <span className="font-mono">{current ?? "—"}</span>. Training builds a
                separate new version and never disturbs the one currently answering predictions.
              </p>

              <Button onClick={handleRetrain} disabled={job?.status === "running"}>
                {job?.status === "running" ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="mr-2 h-4 w-4" />
                )}
                {job?.status === "running" ? "Training…" : "Start training"}
              </Button>

              {job?.status === "failed" ? (
                <p className="text-sm text-destructive">
                  Training failed: {job.error}. The live model is unchanged.
                </p>
              ) : null}

              {job?.status === "done" && job.comparison ? (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary">New version {job.version}</Badge>
                    <Badge variant={job.comparison.beats_baseline ? "secondary" : "destructive"}>
                      {job.comparison.beats_baseline
                        ? "Beats the mean baseline"
                        : "Does NOT beat the mean baseline"}
                    </Badge>
                    {job.backed_up === false ? (
                      <Badge variant="outline">Backup to storage failed</Badge>
                    ) : null}
                  </div>

                  <div className="overflow-auto rounded-md border">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/50">
                        <tr>
                          <th className="p-2 text-left font-medium">Target</th>
                          <th className="p-2 text-left font-medium">MAE (new)</th>
                          <th className="p-2 text-left font-medium">MAE (live)</th>
                          <th className="p-2 text-left font-medium">Change</th>
                          <th className="p-2 text-left font-medium">R²</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(job.comparison.targets).map(([name, scores]) => (
                          <tr key={name} className="border-t">
                            <td className="p-2 font-medium">{name}</td>
                            <td className="p-2">{scores.mae?.toFixed(3)}</td>
                            <td className="p-2">{scores.current_mae?.toFixed(3) ?? "—"}</td>
                            <td
                              className={`p-2 ${
                                scores.improved === true
                                  ? "text-emerald-600"
                                  : scores.improved === false
                                    ? "text-destructive"
                                    : "text-muted-foreground"
                              }`}
                            >
                              {scores.mae_delta === null
                                ? "—"
                                : `${scores.mae_delta > 0 ? "+" : ""}${scores.mae_delta.toFixed(3)}`}
                            </td>
                            <td className="p-2">{scores.r2?.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Lower MAE is better. A negative change means this version is more accurate than
                    the one currently live.
                  </p>

                  <Button
                    onClick={() => job.version && promote(job.version, job.comparison!.regressed)}
                    disabled={promoting || !job.version}
                  >
                    {promoting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : (
                      <Rocket className="mr-2 h-4 w-4" />
                    )}
                    Promote this version
                  </Button>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <History className="h-4 w-4" /> Version history
              </CardTitle>
            </CardHeader>
            <CardContent>
              {versions.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No versions recorded yet. The first retrain from this page registers one.
                </p>
              ) : (
                <div className="overflow-auto rounded-md border">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50">
                      <tr>
                        <th className="p-2 text-left font-medium">Version</th>
                        <th className="p-2 text-left font-medium">Trained</th>
                        <th className="p-2 text-left font-medium">Rows</th>
                        <th className="p-2 text-left font-medium">Promoted</th>
                        <th className="p-2 text-left font-medium" />
                      </tr>
                    </thead>
                    <tbody>
                      {versions.map((version) => (
                        <tr key={version.version} className="border-t">
                          <td className="p-2 font-mono text-xs">
                            {version.version}
                            {version.version === current ? (
                              <Badge variant="secondary" className="ml-2">Live</Badge>
                            ) : null}
                          </td>
                          <td className="p-2 text-muted-foreground">
                            {new Date(version.trained_at).toLocaleString()}
                          </td>
                          <td className="p-2">{version.row_count}</td>
                          <td className="p-2 text-muted-foreground">
                            {version.promoted_at
                              ? new Date(version.promoted_at).toLocaleString()
                              : "—"}
                          </td>
                          <td className="p-2 text-right">
                            {version.version === current ? null : (
                              <Button
                                size="sm"
                                variant="outline"
                                disabled={promoting}
                                onClick={() => promote(version.version, false)}
                              >
                                Roll back to this
                              </Button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

- [ ] **Step 2: Create the barrel export**

Create `suri/features/admin/components/ml-training/index.ts`:

```typescript
export { MLTrainingManagement } from "./MLTrainingManagement";
```

- [ ] **Step 3: Verify it compiles**

Run from `suri/`: `npx tsc --noEmit`
Expected: no errors referencing `ml-training`.

- [ ] **Step 4: Commit**

```bash
git add features/admin/components/ml-training
git commit -m "feat: add the Model Training admin panel"
```

---

### Task 12: Wire the tab into the admin dashboard

**Files:**
- Modify: `suri/lib/admin/feature-flags.ts`
- Modify: `suri/features/admin/components/dashboard/AdminTabsHost.tsx`
- Modify: `suri/features/admin/components/dashboard/AdminSidebar.tsx`

**Interfaces:**
- Consumes: `MLTrainingManagement` from Task 11
- Produces: the `"ml-training"` `AdminTab`, reachable at `/admin?tab=ml-training`

- [ ] **Step 1: Add the tab to the union and role access**

In `suri/lib/admin/feature-flags.ts`, add to the `AdminTab` union after `"iot-devices"`:

```typescript
  | "ml-training";
```

Then add `"ml-training"` to the end of both the `superadmin` and `admin` arrays in `ROLE_TAB_ACCESS` (**not** `mao` — retraining is not a municipal-officer capability):

```typescript
  superadmin: [
    "dashboard", "users", "stk-uploads", "archiving",
    "activity", "map", "export",
    "templates", "roles", "soil-testing", "fertilizer-guidelines",
    "iot-devices", "ml-training",
  ],
  admin: [
    "dashboard", "users", "stk-uploads", "archiving",
    "activity", "map", "export",
    "templates", "roles", "soil-testing", "fertilizer-guidelines",
    "iot-devices", "ml-training",
  ],
```

No `ADMIN_TAB_FEATURE_MAP` entry is added: tabs with no mapped feature flag are enabled by default (`isAdminTabEnabled` returns `true` when `mappedFeature` is undefined), matching how `iot-devices` and `fertilizer-guidelines` already behave.

- [ ] **Step 2: Register the component**

In `suri/features/admin/components/dashboard/AdminTabsHost.tsx`, add the import after the `IoTDeviceManagement` import:

```typescript
import { MLTrainingManagement } from "@/features/admin/components/ml-training/MLTrainingManagement";
```

and the `componentMap` entry after `"iot-devices"`:

```typescript
  "ml-training": MLTrainingManagement,
```

- [ ] **Step 3: Add the sidebar link**

In `suri/features/admin/components/dashboard/AdminSidebar.tsx`, add `Brain` to the `lucide-react` import list, then add to `ALL_NAV_LINKS` after the `iot-devices` entry:

```typescript
  { tab: "ml-training", label: "Model Training", icon: Brain },
```

- [ ] **Step 4: Verify it compiles**

Run from `suri/`: `npx tsc --noEmit`
Expected: no errors. `AdminTab` is exhaustively keyed by `componentMap`'s `Record<AdminTab, ...>`, so a missing entry would fail here — a passing run proves the tab is fully wired.

- [ ] **Step 5: Verify end to end in a browser**

1. Start the ML backend: `cd suri-ml-backend && uvicorn main:app --port 8001`
2. Start suri: `cd suri && npm run dev`
3. Sign in as an admin and open `http://localhost:3000/admin?tab=ml-training`.
4. **Upload tab:** upload a small CSV with one valid row and one unknown-crop row. Confirm the valid row shows its calculated N/P₂O₅/K₂O and remarks, and the bad row shows a reason naming the crop.
5. **Retrain tab:** click "Start training". While it runs, `curl` a prediction and confirm it still answers:
   ```bash
   curl -s -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
     -d '{"crop":"Corn","data_source":"iot","ph":6.4}' | python -m json.tool
   ```
   Confirm the metrics table renders when it finishes.
6. **History tab:** confirm the new version is listed and that promoting it flips the "Live" badge.

- [ ] **Step 6: Commit**

```bash
git add lib/admin/feature-flags.ts features/admin/components/dashboard/AdminTabsHost.tsx features/admin/components/dashboard/AdminSidebar.tsx
git commit -m "feat: add the Model Training tab to the admin dashboard"
```

---

## Known Gaps and Deliberate Limits

Recorded so a reviewer doesn't mistake any of these for oversights:

1. **`/predict` performance during training is unmeasured on real hardware.** The design says training and serving coexist; on a 1 GiB `t3.micro` that is an assumption until Task 12 Step 5 is run on the actual instance rather than a laptop. If latency degrades badly, the fix is to lower the training thread's priority — not to block requests (spec §10).
2. **A retrain job does not survive a process restart.** Deliberate (spec §7): no resume logic, no persisted job queue. A restart mid-training loses the run; the promoted model is unaffected.
3. **The upload path has no row-count ceiling.** Vercel caps a serverless request body at ~4.5 MB, which bounds it in practice (the entire current corpus is ~700 KB). A file large enough to matter would fail at the proxy with a platform error rather than a friendly message.
4. **`_existing_laboratory_codes()` reads every code on each validate/commit call.** At a few thousand rows this is one or two paged requests. It would need an index-backed existence check per row if the corpus grew by orders of magnitude.
5. **Promotion trusts the admin over the metrics.** A version failing the baseline check can still be promoted after an extra confirm. That is the spec's explicit choice (§5) — the human is the gate.
6. **`promoted_by` is not populated from the session.** The proxy has the admin's session but the UI does not send an identifier, so the column stays null. Wiring `session.email` through the proxy is a small follow-up, deliberately not bundled here.
