"""Tests for IssueWorklogs deletion reconciliation."""

from __future__ import annotations

import json
import typing as t

from singer_sdk.exceptions import FatalAPIError, RetriableAPIError

from tap_jira.streams import IssueWorklogs
from tap_jira.tap import TapJira

if t.TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest

CONFIG = {
    "domain": "test.atlassian.net",
    "email": "test@example.com",
    "api_token": "test-token",
}

ISSUE_ID = "10001"


def make_worklog(worklog_id: str) -> dict[str, t.Any]:
    return {
        "id": worklog_id,
        "self": (
            f"https://test.atlassian.net/rest/api/3/issue/{ISSUE_ID}"
            f"/worklog/{worklog_id}"
        ),
        "author": {"accountId": "user-1"},
        "updateAuthor": {"accountId": "user-1"},
        "updated": "2026-01-02T10:00:00.000+0000",
        "started": "2026-01-01T09:00:00.000+0000",
        "timeSpentSeconds": 3600,
    }


def make_tap(reconciliation: dict[str, t.Any] | None = None) -> TapJira:
    state: dict[str, t.Any] = {"bookmarks": {}}
    if reconciliation is not None:
        state["bookmarks"]["issue_worklogs"] = {
            "worklog_reconciliation": reconciliation,
        }
    return TapJira(config=CONFIG, state=state)


def get_stream(tap: TapJira) -> IssueWorklogs:
    stream = tap.streams["issue_worklogs"]
    assert isinstance(stream, IssueWorklogs)
    return stream


def reconciliation_state(tap: TapJira) -> dict[str, t.Any]:
    state = t.cast("dict[str, t.Any]", tap.state)
    return t.cast(
        "dict[str, t.Any]",
        state["bookmarks"]["issue_worklogs"]["worklog_reconciliation"],
    )


def test_first_read_establishes_baseline_without_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap()
    stream = get_stream(tap)
    monkeypatch.setattr(
        stream,
        "request_records",
        lambda _: iter([make_worklog("1"), make_worklog("2")]),
    )

    records = list(stream.get_records({"issue_id": ISSUE_ID}))

    assert [r["id"] for r in records] == ["1", "2"]
    assert all(not r.get("withdrawn") for r in records)
    issue_state = reconciliation_state(tap)[ISSUE_ID]
    assert issue_state["delivered_worklog_ids"] == ["1", "2"]
    assert "reconciliation_pending" not in issue_state


def test_complete_read_delivers_withdrawal_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap({ISSUE_ID: {"delivered_worklog_ids": ["1", "2", "3"]}})
    stream = get_stream(tap)
    monkeypatch.setattr(
        stream,
        "request_records",
        lambda _: iter([make_worklog("2"), make_worklog("3")]),
    )

    records = list(stream.get_records({"issue_id": ISSUE_ID}))

    live = [r for r in records if not r.get("withdrawn")]
    markers = [r for r in records if r.get("withdrawn")]
    assert [r["id"] for r in live] == ["2", "3"]
    assert len(markers) == 1
    marker = markers[0]
    assert marker["id"] == "1"
    assert marker["issueId"] == ISSUE_ID
    assert marker["withdrawn"] is True
    assert marker["withdrawnAt"]

    issue_state = reconciliation_state(tap)[ISSUE_ID]
    assert issue_state["delivered_worklog_ids"] == ["2", "3"]
    assert "reconciliation_pending" not in issue_state


def test_empty_worklog_list_withdraws_all_and_prunes_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap({ISSUE_ID: {"delivered_worklog_ids": ["1", "2"]}})
    stream = get_stream(tap)
    monkeypatch.setattr(stream, "request_records", lambda _: iter([]))

    records = list(stream.get_records({"issue_id": ISSUE_ID}))

    assert sorted(r["id"] for r in records) == ["1", "2"]
    assert all(r["withdrawn"] for r in records)
    assert ISSUE_ID not in reconciliation_state(tap)


def test_pagination_failure_leaves_pending_retry_and_no_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap({ISSUE_ID: {"delivered_worklog_ids": ["1", "2", "3"]}})
    stream = get_stream(tap)

    def flaky_request_records(
        context: dict[str, t.Any],  # noqa: ARG001
    ) -> Iterator[dict[str, t.Any]]:
        yield make_worklog("2")
        msg = "500 Server Error"
        raise RetriableAPIError(msg)

    monkeypatch.setattr(stream, "request_records", flaky_request_records)

    records = list(stream.get_records({"issue_id": ISSUE_ID}))

    # Only the live record read before the failure is delivered, and rows not
    # returned by the incomplete read are not treated as deleted.
    assert [r["id"] for r in records] == ["2"]
    assert all(not r.get("withdrawn") for r in records)

    issue_state = reconciliation_state(tap)[ISSUE_ID]
    assert issue_state["reconciliation_pending"] is True
    assert "500 Server Error" in issue_state["reconciliation_error"]
    # The delivered baseline keeps every previously delivered ID.
    assert issue_state["delivered_worklog_ids"] == ["1", "2", "3"]


def test_permission_failure_leaves_pending_retry_and_no_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap({ISSUE_ID: {"delivered_worklog_ids": ["1", "2", "3"]}})
    stream = get_stream(tap)

    def forbidden_request_records(
        context: dict[str, t.Any],  # noqa: ARG001
    ) -> t.NoReturn:
        msg = "403 Client Error: Forbidden for url"
        raise FatalAPIError(msg)

    monkeypatch.setattr(stream, "request_records", forbidden_request_records)

    records = list(stream.get_records({"issue_id": ISSUE_ID}))

    assert records == []
    issue_state = reconciliation_state(tap)[ISSUE_ID]
    assert issue_state["reconciliation_pending"] is True
    assert "403 Client Error" in issue_state["reconciliation_error"]
    assert issue_state["delivered_worklog_ids"] == ["1", "2", "3"]


def test_pending_issue_withdraws_only_after_confirmed_complete_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap({ISSUE_ID: {"delivered_worklog_ids": ["1", "2", "3"]}})
    stream = get_stream(tap)

    def flaky_request_records(
        context: dict[str, t.Any],  # noqa: ARG001
    ) -> Iterator[dict[str, t.Any]]:
        yield make_worklog("2")
        msg = "500 Server Error"
        raise RetriableAPIError(msg)

    monkeypatch.setattr(stream, "request_records", flaky_request_records)
    first = list(stream.get_records({"issue_id": ISSUE_ID}))
    assert all(not r.get("withdrawn") for r in first)

    monkeypatch.setattr(
        stream,
        "request_records",
        lambda _: iter([make_worklog("2"), make_worklog("3")]),
    )
    second = list(stream.get_records({"issue_id": ISSUE_ID}))

    markers = [r for r in second if r.get("withdrawn")]
    assert [m["id"] for m in markers] == ["1"]
    issue_state = reconciliation_state(tap)[ISSUE_ID]
    assert "reconciliation_pending" not in issue_state
    assert issue_state["delivered_worklog_ids"] == ["2", "3"]


def test_sync_pending_reconciliations_rechecks_pending_issues(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tap = make_tap(
        {
            ISSUE_ID: {
                "delivered_worklog_ids": ["1", "2"],
                "reconciliation_pending": True,
                "reconciliation_error": "403 Client Error",
            },
            "10002": {"delivered_worklog_ids": ["9"]},
        },
    )
    stream = get_stream(tap)
    requested: list[str] = []

    def request_records(context: dict[str, t.Any]) -> Iterator[dict[str, t.Any]]:
        requested.append(str(context["issue_id"]))
        yield make_worklog("2")

    monkeypatch.setattr(stream, "request_records", request_records)

    stream.sync_pending_reconciliations()

    # Only the pending issue is re-checked.
    assert requested == [ISSUE_ID]

    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    record_messages = [m for m in messages if m["type"] == "RECORD"]
    live = [m["record"] for m in record_messages if not m["record"]["withdrawn"]]
    markers = [m["record"] for m in record_messages if m["record"]["withdrawn"]]
    assert [r["id"] for r in live] == ["2"]
    assert live[0]["issueId"] == ISSUE_ID
    assert len(markers) == 1
    assert markers[0]["id"] == "1"
    assert markers[0]["issueId"] == ISSUE_ID
    assert markers[0]["withdrawnAt"]

    # A SCHEMA message precedes the RECORD messages.
    schema_index = next(
        i
        for i, m in enumerate(messages)
        if m["type"] == "SCHEMA" and m["stream"] == "issue_worklogs"
    )
    first_record_index = messages.index(record_messages[0])
    assert schema_index < first_record_index

    state = reconciliation_state(tap)
    assert state[ISSUE_ID]["delivered_worklog_ids"] == ["2"]
    assert "reconciliation_pending" not in state[ISSUE_ID]
    assert state["10002"]["delivered_worklog_ids"] == ["9"]


def test_sweep_skips_issues_rechecked_during_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap({ISSUE_ID: {"delivered_worklog_ids": ["1"]}})
    stream = get_stream(tap)
    calls: list[str] = []

    def failing_request_records(context: dict[str, t.Any]) -> t.NoReturn:
        calls.append(str(context["issue_id"]))
        msg = "500 Server Error"
        raise RetriableAPIError(msg)

    monkeypatch.setattr(stream, "request_records", failing_request_records)

    assert list(stream.get_records({"issue_id": ISSUE_ID})) == []
    stream.sync_pending_reconciliations()

    # The failed issue is left pending but is not retried twice in one run.
    assert calls == [ISSUE_ID]
    issue_state = reconciliation_state(tap)[ISSUE_ID]
    assert issue_state["reconciliation_pending"] is True


def test_issue_stream_sync_triggers_pending_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tap = make_tap(
        {ISSUE_ID: {"delivered_worklog_ids": ["1"], "reconciliation_pending": True}},
    )
    issue_stream = tap.streams["issues"]
    worklog_stream = get_stream(tap)
    monkeypatch.setattr(issue_stream, "request_records", lambda _: iter([]))
    requested: list[str] = []

    def worklog_request_records(
        context: dict[str, t.Any],
    ) -> Iterator[dict[str, t.Any]]:
        requested.append(str(context["issue_id"]))
        return iter([])

    monkeypatch.setattr(worklog_stream, "request_records", worklog_request_records)

    assert list(issue_stream.get_records(None)) == []

    # Exhausting the issues stream re-checks the pending issue, and the
    # confirmed-complete empty read withdraws its remaining worklog.
    assert requested == [ISSUE_ID]
    assert ISSUE_ID not in reconciliation_state(tap)


def test_post_process_marks_live_records_and_adds_issue_id() -> None:
    stream = get_stream(make_tap())

    live = stream.post_process(make_worklog("7"), {"issue_id": ISSUE_ID})
    assert live["issueId"] == ISSUE_ID
    assert live["withdrawn"] is False

    marker = stream.post_process(
        {
            "id": "7",
            "issueId": ISSUE_ID,
            "withdrawn": True,
            "withdrawnAt": "2026-01-03T00:00:00+00:00",
        },
        {"issue_id": ISSUE_ID},
    )
    assert marker["withdrawn"] is True
