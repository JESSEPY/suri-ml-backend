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

    def delete(self):
        return self._query("delete")


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


class TestDeleteVersion:
    def test_deletes_the_single_row_by_version(self):
        client = FakeClient()
        registry.delete_version("2026-08-18-101112", client=client)

        entry = client.log[0]
        assert entry["action"] == "delete"
        assert ("version", "2026-08-18-101112") in entry["filters"]


class TestDeleteAll:
    def test_deletes_every_row(self):
        client = FakeClient()
        registry.delete_all(client=client)

        entry = client.log[0]
        assert entry["action"] == "delete"
        # neq on a column no real version equals is the standard Supabase
        # "delete everything" idiom (delete() alone refuses to run
        # unfiltered) -- same pattern already used for the training table.
        assert any(name.startswith("neq:") for name, _ in entry["filters"])
