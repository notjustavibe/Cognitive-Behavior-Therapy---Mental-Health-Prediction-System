"""
SQLite storage for MindCheck profiles and assessment records.

Everything lives in ONE file: mindcheck.db (created automatically on first run).
Set the MINDCHECK_DB environment variable to keep it somewhere else, e.g. on a
persistent disk (see README - free Render disks are wiped on every redeploy).
"""
import os
import json
import sqlite3
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("MINDCHECK_DB", os.path.join(BASE_DIR, "mindcheck.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    full_name     TEXT    NOT NULL,
    password_hash TEXT    NOT NULL,
    gender        TEXT,
    age_group     TEXT,
    country       TEXT,
    created_at    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at    TEXT    NOT NULL,
    overall_risk  INTEGER NOT NULL,
    overall_label TEXT    NOT NULL,
    overall_score REAL    NOT NULL,
    confidence    REAL    NOT NULL,
    country       TEXT,
    answers_json  TEXT    NOT NULL,
    result_json   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assess_user ON assessments(user_id, created_at);
"""


def _now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat(sep=" ")


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


# ── Users ─────────────────────────────────────────────────────────────────────
def create_user(username, full_name, password_hash, gender=None, age_group=None, country=None):
    """Returns the new user id, or None if the username is already taken."""
    try:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, full_name, password_hash, gender, age_group, country, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (username, full_name, password_hash, gender, age_group, country, _now()))
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user_by_username(username):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


def get_user(user_id):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def update_user(user_id, full_name, gender, age_group, country):
    with connect() as conn:
        conn.execute(
            "UPDATE users SET full_name=?, gender=?, age_group=?, country=? WHERE id=?",
            (full_name, gender, age_group, country, user_id))


def delete_user(user_id):
    """Deletes the profile and (via ON DELETE CASCADE) all of its records."""
    with connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# ── Assessments ───────────────────────────────────────────────────────────────
def save_assessment(user_id, answers, result):
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO assessments (user_id, created_at, overall_risk, overall_label, overall_score,"
            " confidence, country, answers_json, result_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, _now(), result["overall_risk"], result["overall_label"],
             result["overall_score"], result["confidence"], result["demo"]["country"],
             json.dumps(answers, ensure_ascii=False), json.dumps(result, ensure_ascii=False)))
        return cur.lastrowid


def list_assessments(user_id):
    """Newest first. Light columns only (no big JSON blobs)."""
    with connect() as conn:
        return conn.execute(
            "SELECT id, created_at, overall_risk, overall_label, overall_score, confidence, country"
            " FROM assessments WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,)).fetchall()


def get_assessment(user_id, assessment_id):
    """Only returns the record if it belongs to this user."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE id = ? AND user_id = ?",
            (assessment_id, user_id)).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "created_at": row["created_at"],
            "answers": json.loads(row["answers_json"]),
            "result": json.loads(row["result_json"])}


def delete_assessment(user_id, assessment_id):
    with connect() as conn:
        conn.execute("DELETE FROM assessments WHERE id = ? AND user_id = ?", (assessment_id, user_id))
