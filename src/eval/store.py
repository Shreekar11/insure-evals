import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from src.eval.schema import ProbeResult
from config import TRACES_DB


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traces (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    probe_id          TEXT NOT NULL,
    axis              TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt            TEXT NOT NULL,
    response          TEXT NOT NULL,
    flagged           INTEGER NOT NULL,
    severity          INTEGER NOT NULL,
    latency_ms        REAL NOT NULL,
    cost_usd          REAL NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    judge_reasoning   TEXT DEFAULT '',
    turn              INTEGER DEFAULT 1,
    extra             TEXT DEFAULT '{}'
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_axis_model ON traces (axis, model);
"""


@contextmanager
def _conn(db_path: Path = TRACES_DB) -> Generator[sqlite3.Connection, None, None]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path: Path = TRACES_DB):
    with _conn(db_path) as con:
        con.execute(_CREATE_TABLE)
        con.execute(_CREATE_INDEX)


def insert(result: ProbeResult, db_path: Path = TRACES_DB) -> int:
    sql = """
    INSERT INTO traces
        (probe_id, axis, model, prompt, response, flagged, severity,
         latency_ms, cost_usd, prompt_tokens, completion_tokens,
         judge_reasoning, turn, extra)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with _conn(db_path) as con:
        cur = con.execute(sql, (
            result.probe_id,
            result.axis,
            result.model,
            result.prompt,
            result.response,
            int(result.flagged),
            result.severity,
            result.latency_ms,
            result.cost_usd,
            result.prompt_tokens,
            result.completion_tokens,
            result.judge_reasoning,
            result.turn,
            json.dumps(result.extra),
        ))
        return cur.lastrowid


def fetch_from_json(json_path: Path | None = None) -> list[dict]:
    """Read traces from JSON export (used on HF Spaces where SQLite is unavailable)."""
    import json as _json
    p = json_path or TRACES_DB.parent / "traces.json"
    if not p.exists():
        return []
    return _json.loads(p.read_text())


def fetch_all(db_path: Path = TRACES_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute("SELECT * FROM traces ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def fetch_axis_model(axis: str, model: str, db_path: Path = TRACES_DB) -> list[dict]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT * FROM traces WHERE axis=? AND model=? ORDER BY id",
            (axis, model),
        ).fetchall()
    return [dict(r) for r in rows]


def summary(db_path: Path = TRACES_DB) -> dict:
    """
    Aggregate rate (% flagged) and degree (mean severity) per axis per model.
    Returns the structure written to summary.json.
    """
    with _conn(db_path) as con:
        rows = con.execute(
            """
            SELECT axis, model,
                   COUNT(*) AS n,
                   SUM(flagged) AS n_flagged,
                   ROUND(AVG(CAST(severity AS REAL)), 2) AS mean_severity,
                   ROUND(AVG(latency_ms), 0) AS mean_latency_ms,
                   ROUND(SUM(cost_usd), 5) AS total_cost_usd
            FROM traces
            GROUP BY axis, model
            ORDER BY axis, model
            """,
        ).fetchall()

    result: dict[str, dict] = {}
    for r in rows:
        axis = r["axis"]
        model = r["model"]
        n = r["n"]
        n_flagged = r["n_flagged"] or 0
        rate = round(n_flagged / n, 3) if n else 0.0
        result.setdefault(axis, {})[model] = {
            "n": n,
            "n_flagged": n_flagged,
            "rate": rate,
            "mean_severity": r["mean_severity"],
            "mean_latency_ms": r["mean_latency_ms"],
            "total_cost_usd": r["total_cost_usd"],
        }

    # Context-rot: hallucination rate by turn
    with _conn(db_path) as con:
        rot_rows = con.execute(
            """
            SELECT model, turn,
                   COUNT(*) AS n,
                   ROUND(AVG(CAST(flagged AS REAL)), 3) AS rate
            FROM traces
            WHERE axis='context_rot'
            GROUP BY model, turn
            ORDER BY model, turn
            """,
        ).fetchall()
    context_rot: dict = {}
    for r in rot_rows:
        context_rot.setdefault(r["model"], []).append(
            {"turn": r["turn"], "n": r["n"], "rate": r["rate"]}
        )
    result["context_rot"] = context_rot

    return result
