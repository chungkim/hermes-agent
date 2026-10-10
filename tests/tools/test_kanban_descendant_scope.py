"""Real shell/CLI ingress: inherited context is not a board-write grant."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect
from tools import kanban_tools
from tools.environments.local import LocalEnvironment


ROOT = Path(__file__).resolve().parents[2]


def _worker_board(tmp_path, monkeypatch):
    db = tmp_path / "assigned.db"
    conn = connect(db)
    own, foreign = [kb.create_task(conn, title=title) for title in ("own", "foreign")]
    for tid in (own, foreign):
        kb.claim_task(conn, tid)
    task = kb.get_task(conn, own)
    for key, value in {
        "HERMES_KANBAN_DB": str(db), "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_TASK": own, "HERMES_KANBAN_RUN_ID": str(task.current_run_id),
        "HERMES_KANBAN_CLAIM_LOCK": task.claim_lock, "HOME": str(tmp_path),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    return conn, own, foreign


def test_terminal_descendants_cannot_mutate_even_after_task_is_removed(tmp_path, monkeypatch):
    conn, own, foreign = _worker_board(tmp_path, monkeypatch)
    script = tmp_path / "descendant.py"
    script.write_text(
        "import os, sys, json, subprocess\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tools import kanban_tools as kt\n"
        "from agent.delegation_context import is_dispatcher_owned_worker_context\n"
        f"own, foreign = {own!r}, {foreign!r}\n"
        "out = {'owner': is_dispatcher_owned_worker_context(), 'default': kt._default_task_id(None),"
        " 'db': os.getenv('HERMES_KANBAN_DB'), 'board': os.getenv('HERMES_KANBAN_BOARD')}\n"
        "out['show'] = json.loads(kt._handle_show({'task_id':own}))\n"
        "out['tools'] = [json.loads(kt._handle_complete({'task_id': t, 'summary':'must refuse'})) for t in (own,foreign)]\n"
        "os.environ.pop('HERMES_KANBAN_TASK', None)\n"
        f"p = subprocess.run([sys.executable, '-m', 'hermes_cli.main', 'kanban', 'complete', foreign, '--result', 'must refuse'], cwd={str(ROOT)!r}, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=45)\n"
        "out['later_cli'] = {'rc': p.returncode, 'out':p.stdout, 'err':p.stderr}\n"
        "print('SCOPE_RESULT=' + json.dumps(out))\n"
    )
    terminal = LocalEnvironment(cwd=str(tmp_path))
    try:
        result = terminal.execute(f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
    finally:
        terminal.cleanup()
    from agent.skill_preprocessing import run_inline_shell
    from agent.shell_hooks import ShellHookSpec, _spawn
    from tools.code_execution_env import _build_child_env
    from tools.mcp_tool_config import _build_safe_env

    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    outputs = [result.get("output", ""), run_inline_shell(command, tmp_path, 45),
               _spawn(ShellHookSpec(event="session:start", command=command, timeout=45), "{}")['stdout']]
    child_envs = [
        _build_child_env(rpc_endpoint="fixture", rpc_token="fixture", tmpdir=str(tmp_path), child_python=sys.executable),
        _build_safe_env({"HERMES_HOME": os.environ["HERMES_HOME"]}),
    ]
    for env in child_envs:
        proc = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    for output in outputs:
        row = json.loads(next(line.split("SCOPE_RESULT=", 1)[1] for line in output.splitlines() if "SCOPE_RESULT=" in line))
        assert row["show"]["task"]["id"] == own, row
        assert not row["owner"] and row["default"] is None, row
        assert row["db"] == str(tmp_path / "assigned.db") and row["board"] == "default"
        assert all("error" in value for value in row["tools"]), row
        assert row["later_cli"]["rc"] != 0, row
    assert [kb.get_task(conn, t).status for t in (own, foreign)] == ["running", "running"]
    assert json.loads(kanban_tools._handle_complete({"summary": "parent handoff"}))["ok"]
    assert kb.get_task(conn, own).status == "done"
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


def test_worker_cli_cannot_use_foreign_task_to_drop_run_scope(tmp_path, monkeypatch):
    conn, own, foreign = _worker_board(tmp_path, monkeypatch)
    assert "error" in json.loads(kanban_tools._handle_complete({"task_id": foreign, "summary": "no"}))
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "complete", foreign, "--result", "no"],
        cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=45,
    )
    assert proc.returncode != 0 and "worker is scoped to task" in proc.stderr, (proc.stdout, proc.stderr)
    assert kb.get_task(conn, foreign).status == "running"
    attachment = tmp_path / "note.txt"
    attachment.write_text("fixture")
    assert kb.block_task(conn, foreign, reason="fixture awaiting orchestrator")
    for arguments in (["attach", foreign, str(attachment)], ["unblock", foreign]):
        proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *arguments],
                              cwd=ROOT, env=dict(os.environ), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=45)
        assert proc.returncode != 0, (arguments, proc.stdout, proc.stderr)
    assert not kb.list_attachments(conn, foreign)
    assert json.loads(kanban_tools._handle_complete({"task_id": own, "summary": "parent"}))["ok"]
    conn.close()


def test_child_shell_can_write_a_kanban_board_outside_its_lineage_root(tmp_path, monkeypatch):
    """The fence a delegate_task child inherits applies to ITS lineage's board, not to every Kanban
    DB its shell touches: a repro run against a scratch HERMES_HOME got a silently read-only board
    (``connect`` opened ``?mode=ro``; ``write_txn`` raised PermissionError). Real ``terminal``
    ingress, real subprocess, real SQLite — the lineage board stays fenced in the same shell."""
    from agent.delegation_context import delegated_child_context

    lineage_home = tmp_path / "lineage"
    scratch_home = tmp_path / "scratch"
    for home in (lineage_home, scratch_home):
        home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(lineage_home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(key, raising=False)
    connect(kb.kanban_db_path()).close()  # the owner initializes the lineage board
    script = tmp_path / "repro.py"
    script.write_text(
        "import os, sys, json\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from hermes_cli import kanban_db as kb\n"
        "from hermes_cli.kanban_db_connect import connect\n"
        "out = {'marker': os.environ.get('HERMES_DELEGATED_CHILD_CONTEXT')}\n"
        "try:\n"
        "    conn = connect(kb.kanban_db_path()); kb.create_task(conn, title='lineage'); out['lineage'] = 'WROTE'\n"
        "except PermissionError as exc:\n"
        "    out['lineage'] = 'fenced: ' + str(exc)\n"
        f"os.environ['HERMES_HOME'] = {str(scratch_home)!r}\n"
        "import hermes_constants; hermes_constants._default_hermes_root_memo = None\n"
        "conn = connect(kb.kanban_db_path()); out['scratch'] = kb.create_task(conn, title='scratch')\n"
        "print('SCOPE_RESULT=' + json.dumps(out))\n"
    )
    with delegated_child_context("child-repro"):
        terminal = LocalEnvironment(cwd=str(tmp_path))
        try:
            result = terminal.execute(f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}")
        finally:
            terminal.cleanup()
    output = result.get("output", "")
    row = json.loads(next(line.split("SCOPE_RESULT=", 1)[1] for line in output.splitlines() if "SCOPE_RESULT=" in line))
    assert row["marker"] and row["marker"] != "1", row
    assert row["lineage"].startswith("fenced"), row
    assert row["scratch"], row
    scratch_conn = connect(scratch_home / "kanban.db")
    assert kb.get_task(scratch_conn, row["scratch"]).title == "scratch"
    scratch_conn.close()


def _fenced_descendant_env(monkeypatch, tmp_path, home):
    """The live condition: an inherited fence marker over the lineage's own board
    with NO ``HERMES_KANBAN_TASK`` -- inherited scope, not a dispatcher worker."""
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", str(home))


def test_fenced_descendant_can_list_without_a_write(tmp_path, monkeypatch):
    """RED regression for the live ``kanban_list`` fence error.

    ``_handle_list`` called ``kb.recompute_ready(conn)``, which opens ``write_txn``
    -> ``_assert_not_delegated_child_mutation`` and raised
    ``PermissionError: delegate_task child contexts cannot mutate Kanban tasks or
    boards``. Listing is a READ, and a fenced board is already served read-only by
    ``connect()`` (``?mode=ro``); attaching a promotion write to it made an ordinary
    orchestrator's board view fail outright.

    The fence itself is correct and must stay: this asserts the READ succeeds and
    that the descendant still cannot promote or write anything.
    """
    import json as _json

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)

    # The owner initializes and seeds the board.
    owner = connect(home / "kanban.db")
    parent = kb.create_task(owner, title="parent")
    # Still open, so this child sits in 'todo' until somebody runs the promotion.
    child = kb.create_task(owner, title="child", parents=[parent])
    assert kb.get_task(owner, child).status == "todo", "precondition: promotion is pending"
    owner.close()

    # A descendant with inherited scope over the same board.
    _fenced_descendant_env(monkeypatch, tmp_path, home)
    from agent.delegation_context import kanban_path_is_fenced
    db_path = home / "kanban.db"
    assert kanban_path_is_fenced(db_path), "precondition: the board is fenced"

    listed = _json.loads(kanban_tools._handle_list({"limit": 50}))
    assert "error" not in listed, listed
    assert {t["id"] for t in listed["tasks"]} >= {parent, child}, listed
    assert listed["count"] == len(listed["tasks"]), listed
    # Truncation + filters still behave.
    one = _json.loads(kanban_tools._handle_list({"limit": 1}))
    assert one["truncated"] is True and one["count"] == 1, one
    assert one["next_limit"] == 2, one
    by_assignee = _json.loads(kanban_tools._handle_list({"limit": 50, "assignee": "nobody"}))
    assert by_assignee["tasks"] == [], by_assignee

    # promoted=0 is reported honestly rather than a fabricated count.
    assert listed["promoted"] == 0, listed
    # The descendant performed ZERO promotions: the gated child is still todo, and
    # the owner's own recompute still promotes it -- i.e. the read neither advanced
    # nor poisoned the parent's state.
    after = connect(db_path)
    assert kb.get_task(after, child).status == "todo", "a fenced read must not promote"
    # The parent's promotion state is intact, not advanced and not poisoned: the
    # gated child is still waiting on its parent, exactly as seeded.
    assert kb.get_task(after, parent).status == "ready"
    assert [pid for pid, _ in kb.unsatisfied_parents(after, child)] == [parent], "still gated by its parent"
    assert after.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    after.close()


def test_fenced_descendant_list_still_refuses_real_mutation(tmp_path, monkeypatch):
    """The fence is NOT weakened by the read fix: real writes stay refused."""
    import json as _json

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                "HERMES_DELEGATED_CHILD_CONTEXT"):
        monkeypatch.delenv(key, raising=False)
    owner = connect(home / "kanban.db")
    existing = kb.create_task(owner, title="existing")
    owner.close()

    _fenced_descendant_env(monkeypatch, tmp_path, home)

    # Direct DB mutation primitives are still denied.
    conn = connect(home / "kanban.db")
    for call in (
        lambda: kb.create_task(conn, title="should not exist"),
        lambda: kb.add_comment(conn, existing, "descendant", "should not exist"),
        lambda: kb.set_current_board("other"),
    ):
        with pytest.raises(PermissionError):
            call()
    conn.close()

    # Nothing was created or commented by the refused calls.
    audit = connect(home / "kanban.db")
    assert len(kb.list_tasks(audit)) == 1, "no task was created"
    assert kb.list_comments(audit, existing) == [], "no comment was written"
    audit.close()

    # And the orchestrator-only creation tool is still refused for a descendant.
    created = _json.loads(kanban_tools._handle_create({"title": "no", "assignee": "x"}))
    assert "error" in created, created
