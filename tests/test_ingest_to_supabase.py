import math

import pytest

from data.ingest_to_supabase import chunked, frame_to_records
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
