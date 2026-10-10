"""Real subprocess/CLI integration: a fenced descendant LISTS over the CLI.

Hermes's acceptance requires more than mocked unit tests -- the ordinary
orchestrator path and the real `hermes kanban list` ingress must both work when
the marker is inherited, while real delegated children still cannot mutate.

Runs a genuine child process through the real CLI (not the tool handler) against
an isolated HERMES_HOME.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


def _seed(home: Path, *, parent_done: bool = False):
    """A child waits in ``todo`` until an orchestrator's recompute sees its parent
    terminal, so ``parent_done`` decides whether a promotion is actually pending."""
    conn = connect(home / "kanban.db")
    parent = kb.create_task(conn, title="parent")
    child = kb.create_task(conn, title="child", parents=[parent])
    if parent_done:
        # complete_task() recomputes internally, so finish the parent FIRST and add
        # the dependent child afterwards -- otherwise seeding promotes it and there
        # is nothing left for a list to (not) promote.
        kb.complete_task(conn, parent, summary="owner finished the parent")
        kb.link_tasks(conn, parent, child)
    conn.close()
    return parent, child


def _cli(env_extra, args, home):
    # Inherit the ambient env (the CLI's runtime/PM bootstrap depends on it and
    # must NOT be triggered by a stripped environment) and override ONLY the
    # home/kanban keys under test. The parent process's own delegate marker and
    # HERMES_KANBAN_* are removed first so the fixture starts unfenced.
    env = {k: v for k, v in os.environ.items()
           if k not in ("HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_KANBAN_DB",
                        "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
                        "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK")}
    env["HOME"] = str(home.parent)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=120,
    )


def test_fenced_descendant_cli_list_succeeds_and_cannot_mutate(tmp_path):
    """RED->GREEN at the real CLI seam.

    Before the fix `hermes kanban list` in this environment failed with
    "delegate_task child contexts cannot mutate Kanban tasks or boards" because the
    list handler recomputed ready-dependencies (a write) on a read-only board.
    """
    home = tmp_path / "home"
    home.mkdir()
    parent, child = _seed(home)

    fenced = {
        # Inherited scope over the lineage's own board; NOT a dispatcher worker.
        "HERMES_DELEGATED_CHILD_CONTEXT": str(home),
    }

    listed = _cli(fenced, ["list"], home)
    assert listed.returncode == 0, (listed.stdout, listed.stderr)
    assert "cannot mutate Kanban" not in listed.stderr, listed.stderr
    assert parent in listed.stdout and child in listed.stdout, listed.stdout

    # The dependent child is still gated: the fenced read promoted nothing.
    audit = connect(home / "kanban.db")
    assert kb.get_task(audit, child).status == "todo", "a fenced list must not promote"
    audit.close()

    # And real mutation through the CLI is still refused in the same env.
    for args in (["create", "nope"], ["complete", child, "--result", "nope"]):
        proc = _cli(fenced, args, home)
        assert proc.returncode != 0, (args, proc.stdout)
        assert "cannot mutate" in (proc.stderr + proc.stdout).lower() or "orchestrator" in (proc.stderr + proc.stdout).lower(), \
            (args, proc.stdout, proc.stderr)

    audit = connect(home / "kanban.db")
    assert len(kb.list_tasks(audit)) == 2, "no task was created"
    assert kb.get_task(audit, child).status == "todo", "no task was completed"
    audit.close()


def test_ordinary_orchestrator_cli_list_still_promotes(tmp_path):
    """Parent semantics preserved: with no marker, list still recomputes ready."""
    home = tmp_path / "home"
    home.mkdir()
    # Parent already terminal: an unfenced list must perform the promotion.
    parent, child = _seed(home, parent_done=True)

    listed = _cli({}, ["list"], home)
    assert listed.returncode == 0, (listed.stdout, listed.stderr)
    assert "cannot mutate" not in listed.stderr
    assert parent in listed.stdout and child in listed.stdout

    # No marker -> the promotion write is legal and runs, exactly as before the fix.
    audit = connect(home / "kanban.db")
    assert kb.get_task(audit, child).status == "ready", "an orchestrator list still promotes"
    audit.close()