from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date
from typing import Optional
import sqlite3
from settings import config

app = FastAPI(title="Hubble Chatbot Backend")


def get_db():
    # Support both sqlite:/// and sqlite: URLs; strip prefix to path
    url = config.database_url
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "", 1)
    elif url.startswith("sqlite:"):
        path = url.replace("sqlite:", "", 1)
    else:
        # Fall back to treating as a direct file path
        path = url
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    # Enforce foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


class TimesheetEntryCreate(BaseModel):
    user_id: int = Field(..., gt=0)
    project_id: int = Field(..., gt=0)
    module_id: int = Field(..., gt=0)
    task_id: int = Field(..., gt=0)
    team_id: int = Field(..., gt=0)
    description: str = Field(..., min_length=1)
    entry_date: date
    working_hours: float = Field(..., gt=0)
    approved_hours: Optional[float] = Field(default=0, ge=0)
    authorized_hours: Optional[float] = Field(default=0, ge=0)
    billed_hours: Optional[float] = Field(default=0, ge=0)
    admin_comments: Optional[str] = None

    @field_validator("approved_hours", "authorized_hours", "billed_hours", mode="before")
    @classmethod
    def default_zero_for_missing(cls, v):
        return 0 if v is None or v == "" else v

    @field_validator("working_hours", "approved_hours", "authorized_hours", "billed_hours")
    @classmethod
    def non_negative(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_hours_monotonic(self):
        if self.working_hours is not None and self.working_hours > 24:
            raise ValueError("working_hours cannot exceed 24 hours for a single entry")
        if self.approved_hours is not None and self.approved_hours > self.working_hours:
            raise ValueError("approved_hours cannot exceed working_hours")
        if self.authorized_hours is not None and self.approved_hours is not None and self.authorized_hours > self.approved_hours:
            raise ValueError("authorized_hours cannot exceed approved_hours")
        if self.billed_hours is not None and self.approved_hours is not None and self.billed_hours > self.approved_hours:
            raise ValueError("billed_hours cannot exceed approved_hours")
        return self


@app.get("/")
def hello_world():
    return {"message": "Hello, World!"}


@app.get("/config")
def get_config():
    return {"open_api_key": config.open_api_key}


@app.post("/timesheet-entries", status_code=201)
def create_timesheet_entry(payload: TimesheetEntryCreate):
    conn = get_db()
    try:
        cur = conn.cursor()

        # existence checks
        refs = {
            "users": payload.user_id,
            "projects": payload.project_id,
            "modules": payload.module_id,
            "tasks": payload.task_id,
            "teams": payload.team_id,
        }
        for table, value in refs.items():
            cur.execute(f"SELECT 1 FROM {table} WHERE id = ?", (value,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"{table[:-1]} not found: {value}")

        # relationship checks
        cur.execute(
            "SELECT 1 FROM modules WHERE id = ? AND project_id = ?",
            (payload.module_id, payload.project_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=422, detail="module_id does not belong to project_id")

        cur.execute(
            "SELECT 1 FROM tasks WHERE id = ? AND module_id = ?",
            (payload.task_id, payload.module_id),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=422, detail="task_id does not belong to module_id")

        # allow multiple entries per user per day; daily cap enforced below

        # daily cap
        cur.execute(
            "SELECT COALESCE(SUM(working_hours), 0) FROM timesheet_entries WHERE user_id = ? AND entry_date = ?",
            (payload.user_id, payload.entry_date),
        )
        row = cur.fetchone()
        current_total = (row[0] if row and row[0] is not None else 0)
        if current_total + payload.working_hours > 24:
            raise HTTPException(status_code=422, detail="Total working_hours for the day would exceed 24 hours")

        # insert
        cur.execute(
            """
            INSERT INTO timesheet_entries
                (user_id, project_id, module_id, task_id, description, entry_date,
                 working_hours, approved_hours, authorized_hours, billed_hours, team_id, admin_comments, created_at, updated_at)
            VALUES
                (?, ?, ?, ?, ?, ?,
                 ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                payload.user_id,
                payload.project_id,
                payload.module_id,
                payload.task_id,
                payload.description,
                payload.entry_date,
                payload.working_hours,
                payload.approved_hours or 0,
                payload.authorized_hours or 0,
                payload.billed_hours or 0,
                payload.team_id,
                payload.admin_comments,
            ),
        )
        # SQLite returns the new rowid here
        rowid = cur.lastrowid
        # Try to read id; if NULL, sync id to rowid for this row
        cur.execute("SELECT id FROM timesheet_entries WHERE rowid = ?", (rowid,))
        got = cur.fetchone()
        db_id = got[0] if got else None
        if db_id is None:
            cur.execute("UPDATE timesheet_entries SET id = ? WHERE rowid = ?", (rowid, rowid))
            db_id = rowid

        conn.commit()
        return {"id": db_id}
    finally:
        conn.close()
