"""Load the synthetic training CSV into Supabase (spec section 9).

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


def ingest(csv_path: str = OUTPUT_CSV, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    client = get_client()
    records = frame_to_records(pd.read_csv(csv_path))

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
