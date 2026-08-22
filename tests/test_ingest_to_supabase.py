import math

import numpy as np
import pandas as pd
import pytest

from data.ingest_to_supabase import chunked, frame_to_records, _load_frame
from tests.fixtures import sample_training_frame


class TestChunking:
    def test_splits_into_batches_of_the_given_size(self):
        batches = list(chunked(list(range(250)), 100))
        assert [len(batch) for batch in batches] == [100, 100, 50]

    def test_empty_input_yields_no_batches(self):
        assert list(chunked([], 100)) == []

    def test_exact_multiple_has_no_trailing_empty_batch(self):
        assert len(list(chunked(list(range(200)), 100))) == 2


class TestRecordConversion:
    @pytest.fixture(scope="class")
    def records(self):
        return frame_to_records(sample_training_frame(num_records=10, seed=42))

    def test_produces_one_record_per_row(self, records):
        assert len(records) == 10

    def test_nan_becomes_none_for_json(self, records):
        """Supabase rejects NaN — it is not valid JSON. Any NaN produced by
        pandas (e.g. from an unset optional field) must serialise as null."""
        for record in records:
            for value in record.values():
                assert not (isinstance(value, float) and math.isnan(value))

    def test_records_are_json_serialisable(self, records):
        import json
        json.dumps(records)


class TestLoadFrameCsvNoneSentinel:
    """Regression test: data/parse_real_soil_database.py writes the literal
    4-character string "None" into variety/cultivation_method/age_band as a
    deliberate sentinel category (see ml/preprocessing.py's
    VARIETIES/CULTIVATION_METHODS/AGE_BANDS). Plain pd.read_csv() silently
    sniffs that text as a missing value and turns it into NaN, which
    frame_to_records() would then corrupt into a real SQL NULL on upsert to
    Supabase -- and any later retrain sourced from Supabase (ml/train.py's
    no-`--csv` branch) would trip OneHotEncoder(handle_unknown="error") on
    that NULL. Mirrors the fix for the same bug class in ml/train.py's
    _load_frame() (commit 4dfed4d). _load_frame() here must round-trip the
    literal string, while still treating a genuinely blank numeric cell as
    missing.
    """

    @pytest.fixture(scope="class")
    def csv_path(self, tmp_path_factory):
        frame = sample_training_frame(num_records=5, seed=7)
        frame = frame.copy()
        frame.loc[0, "variety"] = "None"
        frame.loc[0, "cultivation_method"] = "None"
        frame.loc[0, "age_band"] = "None"
        # A genuinely missing optional numeric cell -- e.g. an ungraded
        # organic_matter_pct -- must still round-trip as NaN, and become
        # None (not the string "None") once passed through frame_to_records.
        frame.loc[1, "organic_matter_pct"] = np.nan

        path = tmp_path_factory.mktemp("data") / "sentinel.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_literal_none_string_survives_as_string(self, csv_path):
        loaded = _load_frame(csv_path)
        assert loaded.loc[0, "variety"] == "None"
        assert not pd.isna(loaded.loc[0, "variety"])
        assert loaded.loc[0, "cultivation_method"] == "None"
        assert not pd.isna(loaded.loc[0, "cultivation_method"])
        assert loaded.loc[0, "age_band"] == "None"
        assert not pd.isna(loaded.loc[0, "age_band"])

    def test_genuinely_blank_numeric_cell_still_reads_as_nan(self, csv_path):
        loaded = _load_frame(csv_path)
        assert pd.isna(loaded.loc[1, "organic_matter_pct"])

    def test_none_sentinel_survives_frame_to_records_as_string(self, csv_path):
        records = frame_to_records(_load_frame(csv_path))
        assert records[0]["variety"] == "None"
        assert records[0]["cultivation_method"] == "None"
        assert records[0]["age_band"] == "None"

    def test_genuinely_blank_numeric_cell_becomes_none_in_records(self, csv_path):
        records = frame_to_records(_load_frame(csv_path))
        assert records[1]["organic_matter_pct"] is None


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


class TestDeleteAllRecords:
    def test_deletes_unfiltered_and_returns_the_count_removed(self):
        from data.ingest_to_supabase import delete_all_records

        captured = {}

        class FakeQuery:
            def neq(self, column, value):
                captured["filter"] = (column, value)
                return self

            def execute(self):
                return type("R", (), {"data": [{"laboratory_code": "A"}, {"laboratory_code": "B"}]})()

        class FakeTable:
            def delete(self):
                return FakeQuery()

        class FakeClient:
            def table(self, name):
                captured["table"] = name
                return FakeTable()

        assert delete_all_records(client=FakeClient()) == 2
        assert captured["table"] == "ml_soil_training_data"
        # neq on a column no real row ever has a blank value for -- delete()
        # with no filter at all is refused by Supabase.
        assert captured["filter"] == ("laboratory_code", "")


class TestListRecordsPaginated:
    def test_requests_the_right_zero_indexed_inclusive_range(self):
        from data.ingest_to_supabase import list_records_paginated

        captured = {}

        class FakeQuery:
            def order(self, _column):
                return self

            def range(self, start, end):
                captured["range"] = (start, end)
                return self

            def execute(self):
                return type("R", (), {
                    "data": [{"laboratory_code": "A"}], "count": 57,
                })()

        class FakeTable:
            def select(self, _columns, count=None):
                captured["count_mode"] = count
                return FakeQuery()

        class FakeClient:
            def table(self, _name):
                return FakeTable()

        rows, total = list_records_paginated(page=3, page_size=20, client=FakeClient())

        assert captured["range"] == (40, 59)
        assert captured["count_mode"] == "exact"
        assert rows == [{"laboratory_code": "A"}]
        assert total == 57
