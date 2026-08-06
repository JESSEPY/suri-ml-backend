"""Load the real training CSV (data/parse_real_soil_database.py's output)
into Supabase.

Upserts on the (laboratory_code, data_source) primary key so re-running is
idempotent rather than duplicating the corpus.
"""

from __future__ import annotations

import os
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from data.parse_real_soil_database import OUTPUT_CSV

TABLE = "ml_soil_training_data"
DEFAULT_BATCH_SIZE = 500


def get_client() -> Client:
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
            "Copy them from suri/.env.local into suri-ml-backend/.env"
        )
    return create_client(url, key)


def chunked(records: list, size: int) -> Iterator[list]:
    for start in range(0, len(records), size):
        yield records[start:start + size]


def frame_to_records(frame: pd.DataFrame) -> list[dict]:
    """Convert to JSON-safe dicts. pandas uses NaN for missing numerics, which
    is not valid JSON and which Supabase rejects — it must become None."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _load_frame(csv_path: str) -> pd.DataFrame:
    # keep_default_na=False + explicit na_values=[""]: several of this
    # schema's nominal columns (variety/cultivation_method/age_band) use
    # the literal string "None" as a real, valid category (see
    # ml/preprocessing.py's VARIETIES/CULTIVATION_METHODS/AGE_BANDS) --
    # pandas' default NA-sniffing would otherwise silently read that text
    # as NaN, which frame_to_records() below would then corrupt into a
    # real SQL NULL on upsert (mirrors ml/train.py's _load_frame() fix in
    # commit 4dfed4d for the same sentinel/NA-sniffing bug on the direct-CSV
    # training path). An actually-missing numeric cell (e.g. an ungraded
    # organic_matter_pct) is written by csv.DictWriter as a genuinely empty
    # cell, which na_values=[""] still correctly treats as NaN.
    return pd.read_csv(csv_path, keep_default_na=False, na_values=[""])


def ingest(csv_path: str = OUTPUT_CSV, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    client = get_client()
    records = frame_to_records(_load_frame(csv_path))

    total = 0
    for index, batch in enumerate(chunked(records, batch_size), start=1):
        client.table(TABLE).upsert(
            batch, on_conflict="laboratory_code,data_source"
        ).execute()
        total += len(batch)
        print(f"  batch {index}: upserted {len(batch)} rows ({total} total)")

    return total


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} rows into {TABLE}.")
