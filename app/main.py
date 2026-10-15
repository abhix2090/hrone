import os
import re
from datetime import date as Date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))

MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000

PRESENT_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = PRESENT_STATUSES | {"ABSENT", "LEAVE"}

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    serverSelectionTimeoutMS=2000,
    tz_aware=True,
    tzinfo=UTC,
)
db: Database = client[os.getenv("MONGO_DB", "attendance_db")]

app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def truncate_seconds(value: datetime) -> datetime:
    return as_utc(value).replace(microsecond=0)


def to_utc_datetime(epoch_ms: int) -> datetime:
    if isinstance(epoch_ms, bool) or not isinstance(epoch_ms, int):
        raise ValueError("Timestamp must be an integer in epoch milliseconds")

    if not MIN_EPOCH_MS <= epoch_ms <= MAX_EPOCH_MS:
        raise ValueError("Timestamp is outside the supported epoch-millisecond range")

    try:
        return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).replace(microsecond=0)
    except (OverflowError, OSError, ValueError):
        raise ValueError("Timestamp is outside the supported range")


def to_epoch_ms(value: Optional[datetime]) -> Optional[int]:
    if value is None:
        return None
    return int(truncate_seconds(value).timestamp() * 1000)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def round_half_up(value: float, places: int = 2) -> float:
    quantum = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def validate_hhmm(value: str) -> str:
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Time must use HH:MM format")
    return value


def validate_date_string(value: str, field_name: str = "date") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{field_name} must use YYYY-MM-DD format")
    try:
        Date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format")
    return value


def validate_month_string(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        raise ValueError("month must use YYYY-MM format")
    return value


def month_bounds(month: str) -> tuple[str, str]:
    validate_month_string(month)
    year, month_number = map(int, month.split("-"))
    first = Date(year, month_number, 1)
    if month_number == 12:
        next_month = Date(year + 1, 1, 1)
    else:
        next_month = Date(year, month_number + 1, 1)
    return first.isoformat(), (next_month - timedelta(days=1)).isoformat()


def working_day(day: Date) -> bool:
    return day.weekday() < 5


def shift_start_for_instant(punch_in: datetime, shift_start: str, shift_end: str) -> datetime:
    local_in = truncate_seconds(punch_in).astimezone(IST)
    start_hour, start_minute = map(int, shift_start.split(":"))
    end_hour, end_minute = map(int, shift_end.split(":"))

    start_dt = local_in.replace(
        hour=start_hour, minute=start_minute, second=0, microsecond=0
    )

    if (end_hour, end_minute) <= (start_hour, start_minute):
        # For an overnight shift, early-morning punch-ins belong to the previous shift.
        if (local_in.hour, local_in.minute) < (end_hour, end_minute):
            start_dt -= timedelta(days=1)
        elif local_in < start_dt:
            start_dt -= timedelta(days=1)
    elif local_in < start_dt:
        start_dt -= timedelta(days=1)

    return start_dt


def attendance_date_for_punch(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    local_in = truncate_seconds(punch_in).astimezone(IST)
    start_hour, start_minute = map(int, shift_start.split(":"))
    end_hour, end_minute = map(int, shift_end.split(":"))

    if (end_hour, end_minute) <= (start_hour, start_minute):
        if (local_in.hour, local_in.minute) < (end_hour, end_minute):
            return (local_in.date() - timedelta(days=1)).isoformat()

    return local_in.date().isoformat()


def compute_late_minutes(punch_in: datetime, shift_start: str, shift_end: str) -> int:
    local_in = truncate_seconds(punch_in).astimezone(IST)
    shift_start_dt = shift_start_for_instant(punch_in, shift_start, shift_end)
    elapsed_seconds = (local_in - shift_start_dt).total_seconds()

    # Late only when strictly more than ten minutes late.
    if elapsed_seconds > 600:
        return max(0, int(elapsed_seconds // 60))
    return 0


def shift_end_datetime(punch_in: datetime, shift_start: str, shift_end: str) -> datetime:
    start_dt = shift_start_for_instant(punch_in, shift_start, shift_end)
    end_hour, end_minute = map(int, shift_end.split(":"))

    end_dt = start_dt.replace(
        hour=end_hour, minute=end_minute, second=0, microsecond=0
    )

    start_hour, start_minute = map(int, shift_start.split(":"))
    if (end_hour, end_minute) <= (start_hour, start_minute):
        end_dt += timedelta(days=1)

    return end_dt


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    seconds = (truncate_seconds(punch_out) - truncate_seconds(punch_in)).total_seconds()
    return round_half_up(seconds / 3600, 2)


def compute_overtime(
    punch_out: datetime,
    shift_start: str,
    shift_end: str,
    punch_in: datetime,
) -> int:
    end_dt = shift_end_datetime(punch_in, shift_start, shift_end).astimezone(UTC)
    elapsed_seconds = (truncate_seconds(punch_out) - end_dt).total_seconds()
    minutes = int(elapsed_seconds // 60)
    return minutes if minutes >= 30 else 0


def compute_half_day(work_hours: Optional[float]) -> bool:
    return work_hours is not None and work_hours < 4.5


def employee_response(doc: dict) -> dict:
    result = {key: value for key, value in doc.items() if key != "_id"}
    if isinstance(result.get("created_at"), datetime):
        result["created_at"] = to_epoch_ms(result["created_at"])
    return result


def attendance_response(doc: dict) -> dict:
    result = {key: value for key, value in doc.items() if key != "_id"}
    for field in ("punch_in", "punch_out"):
        result[field] = to_epoch_ms(result.get(field))

    history = []
    for entry in result.get("history", []):
        item = dict(entry)
        if isinstance(item.get("at"), datetime):
            item["at"] = to_epoch_ms(item["at"])

        changes = {}
        for field, change in item.get("changes", {}).items():
            if isinstance(change, dict):
                change_copy = dict(change)
                for key in ("from", "to"):
                    if isinstance(change_copy.get(key), datetime):
                        change_copy[key] = to_epoch_ms(change_copy[key])
                changes[field] = change_copy
            else:
                changes[field] = change
        item["changes"] = changes
        history.append(item)

    result["history"] = history
    return result


def add_change(changes: dict, field: str, old_value: Any, new_value: Any) -> None:
    if old_value != new_value:
        changes[field] = {"from": old_value, "to": new_value}


def serialize_change_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return to_epoch_ms(value)
    return value


def require_db() -> None:
    try:
        db.command("ping")
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EmployeeIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("shift_start", "shift_end")
    @classmethod
    def validate_shift_time(cls, value: str) -> str:
        return validate_hhmm(value)

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, value: str) -> str:
        return validate_date_string(value, "joined_on")

    @model_validator(mode="after")
    def validate_shifts_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    punched_at: Optional[int] = None
    status: str = "PRESENT"

    @field_validator("punched_at")
    @classmethod
    def validate_timestamp(cls, value: Optional[int]) -> Optional[int]:
        if value is not None:
            to_utc_datetime(value)
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in PRESENT_STATUSES:
            raise ValueError(f"status must be one of {sorted(PRESENT_STATUSES)}")
        return value


class PunchOutIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    punched_at: Optional[int] = None

    @field_validator("punched_at")
    @classmethod
    def validate_timestamp(cls, value: Optional[int]) -> Optional[int]:
        if value is not None:
            to_utc_datetime(value)
        return value


class AttendanceCorrectionIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    regularized_by: str = Field(min_length=1, max_length=50)
    reason: str = Field(min_length=5, max_length=200)
    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ALL_STATUSES:
            raise ValueError(f"status must be one of {sorted(ALL_STATUSES)}")
        return value

    @field_validator("punch_in", "punch_out")
    @classmethod
    def validate_timestamp(cls, value: Optional[int]) -> Optional[int]:
        if value is not None:
            to_utc_datetime(value)
        return value


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

def ensure_indexes() -> None:
    db.employees.create_index(
        [("emp_code", ASCENDING)],
        unique=True,
        name="uq_employee_code",
    )
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)],
        name="ix_department_employee",
    )
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING), ("emp_code", ASCENDING)],
        name="ix_department_joined_employee",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="uq_attendance_employee_date",
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="ix_attendance_date_employee",
    )
    db.attendance_logs.create_index(
        [("date", ASCENDING), ("status", ASCENDING), ("emp_code", ASCENDING)],
        name="ix_date_status_employee",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING), ("status", ASCENDING)],
        name="ix_employee_date_status",
    )


@app.on_event("startup")
def startup() -> None:
    ensure_indexes()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    try:
        db.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Employees
# ---------------------------------------------------------------------------

@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    employee = body.model_dump()
    employee["created_at"] = utc_now()

    try:
        db.employees.insert_one(employee)
        return employee_response(employee)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    query = {}
    if department is not None:
        query["department"] = department

    try:
        total = db.employees.count_documents(query)
        docs = list(
            db.employees.find(query, {"_id": 0})
            .sort("emp_code", ASCENDING)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        return {
            "items": [employee_response(doc) for doc in docs],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Punch in
# ---------------------------------------------------------------------------

@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    try:
        employee = db.employees.find_one({"emp_code": body.emp_code})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        punched_at = truncate_seconds(
            to_utc_datetime(body.punched_at) if body.punched_at is not None else utc_now()
        )
        attendance_day = attendance_date_for_punch(
            punched_at, employee["shift_start"], employee["shift_end"]
        )
        late_minutes = compute_late_minutes(
            punched_at, employee["shift_start"], employee["shift_end"]
        )

        record = {
            "emp_code": body.emp_code,
            "date": attendance_day,
            "status": body.status,
            "punch_in": punched_at,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": late_minutes,
            "overtime_minutes": 0,
            "half_day": False,
            "history": [],
        }

        db.attendance_logs.insert_one(record)
        return attendance_response(record)

    except HTTPException:
        raise
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail="Attendance already exists for this employee and date",
        )
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Punch out
# ---------------------------------------------------------------------------

@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    try:
        employee = db.employees.find_one({"emp_code": body.emp_code})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        punched_out_at = truncate_seconds(
            to_utc_datetime(body.punched_at) if body.punched_at is not None else utc_now()
        )

        attendance = db.attendance_logs.find_one(
            {
                "emp_code": body.emp_code,
                "punch_in": {"$ne": None, "$lte": punched_out_at},
            },
            sort=[("punch_in", DESCENDING)],
        )

        if not attendance:
            raise HTTPException(status_code=404, detail="No punch-in record found")

        punched_in_at = truncate_seconds(attendance["punch_in"])

        if attendance.get("punch_out") is not None:
            raise HTTPException(
                status_code=409,
                detail="The most recent eligible attendance record is already closed",
            )

        elapsed_seconds = (punched_out_at - punched_in_at).total_seconds()
        if elapsed_seconds <= 0:
            raise HTTPException(
                status_code=422,
                detail="Punch-out time must be after punch-in time",
            )
        if elapsed_seconds > 24 * 60 * 60:
            raise HTTPException(
                status_code=422,
                detail="Punch-out must be within 24 hours of punch-in",
            )

        work_hours = compute_work_hours(punched_in_at, punched_out_at)
        overtime_minutes = compute_overtime(
            punched_out_at,
            employee["shift_start"],
            employee["shift_end"],
            punched_in_at,
        )
        half_day = compute_half_day(work_hours)

        result = db.attendance_logs.update_one(
            {"_id": attendance["_id"], "punch_out": None},
            {
                "$set": {
                    "punch_out": punched_out_at,
                    "work_hours": work_hours,
                    "overtime_minutes": overtime_minutes,
                    "half_day": half_day,
                }
            },
        )

        if result.modified_count != 1:
            raise HTTPException(
                status_code=409,
                detail="Attendance record was updated concurrently",
            )

        updated = db.attendance_logs.find_one({"_id": attendance["_id"]})
        return attendance_response(updated)

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Attendance listing
# ---------------------------------------------------------------------------

@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    query = {}
    if emp_code is not None:
        query["emp_code"] = emp_code

    try:
        if date_from is not None:
            validate_date_string(date_from, "date_from")
        if date_to is not None:
            validate_date_string(date_to, "date_to")
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must not be after date_to")

        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            query["date"] = date_filter

        if status is not None:
            if status not in ALL_STATUSES:
                raise HTTPException(status_code=422, detail="Invalid attendance status")
            query["status"] = status

        total = db.attendance_logs.count_documents(query)
        docs = list(
            db.attendance_logs.find(query)
            .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        return {
            "items": [attendance_response(doc) for doc in docs],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Attendance regularization / correction
# ---------------------------------------------------------------------------

@app.patch("/attendance/{emp_code}/{attendance_date}")
def regularize_attendance(
    emp_code: str,
    attendance_date: str,
    body: AttendanceCorrectionIn,
):
    validate_date_string(attendance_date, "date")

    try:
        employee = db.employees.find_one({"emp_code": emp_code})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        record = db.attendance_logs.find_one(
            {"emp_code": emp_code, "date": attendance_date}
        )
        if not record:
            raise HTTPException(status_code=404, detail="Attendance record not found")

        old_status = record["status"]
        old_in = truncate_seconds(record["punch_in"]) if record.get("punch_in") else None
        old_out = truncate_seconds(record["punch_out"]) if record.get("punch_out") else None

        new_status = body.status if body.status is not None else old_status
        new_in = to_utc_datetime(body.punch_in) if body.punch_in is not None else old_in
        new_out = to_utc_datetime(body.punch_out) if body.punch_out is not None else old_out

        if new_status in {"ABSENT", "LEAVE"}:
            if body.punch_in is not None or body.punch_out is not None:
                raise HTTPException(
                    status_code=422,
                    detail="ABSENT and LEAVE records cannot include punch times",
                )
            new_in = None
            new_out = None
        elif new_in is None:
            raise HTTPException(
                status_code=422,
                detail="Presence status requires punch_in",
            )

        if new_in is not None:
            new_in = truncate_seconds(new_in)
            expected_date = attendance_date_for_punch(
                new_in, employee["shift_start"], employee["shift_end"]
            )
            if expected_date != attendance_date:
                raise HTTPException(
                    status_code=422,
                    detail="punch_in must belong to the attendance date",
                )

        if new_out is not None:
            new_out = truncate_seconds(new_out)
            if new_in is None:
                raise HTTPException(
                    status_code=422,
                    detail="punch_out requires punch_in",
                )
            elapsed = (new_out - new_in).total_seconds()
            if elapsed <= 0:
                raise HTTPException(
                    status_code=422,
                    detail="punch_out must be after punch_in",
                )
            if elapsed > 24 * 60 * 60:
                raise HTTPException(
                    status_code=422,
                    detail="punch_out must be within 24 hours of punch_in",
                )

        if new_status in {"ABSENT", "LEAVE"}:
            new_late = 0
            new_work_hours = None
            new_overtime = 0
            new_half_day = False
        else:
            new_late = compute_late_minutes(
                new_in, employee["shift_start"], employee["shift_end"]
            )
            new_work_hours = (
                compute_work_hours(new_in, new_out) if new_out is not None else None
            )
            new_overtime = (
                compute_overtime(
                    new_out,
                    employee["shift_start"],
                    employee["shift_end"],
                    new_in,
                )
                if new_out is not None
                else 0
            )
            new_half_day = compute_half_day(new_work_hours)

        old_values = {
            "status": old_status,
            "punch_in": old_in,
            "punch_out": old_out,
            "late_minutes": record.get("late_minutes", 0),
            "work_hours": record.get("work_hours"),
            "overtime_minutes": record.get("overtime_minutes", 0),
            "half_day": record.get("half_day", False),
        }
        new_values = {
            "status": new_status,
            "punch_in": new_in,
            "punch_out": new_out,
            "late_minutes": new_late,
            "work_hours": new_work_hours,
            "overtime_minutes": new_overtime,
            "half_day": new_half_day,
        }

        changes = {}
        for field, new_value in new_values.items():
            old_value = old_values[field]
            if old_value != new_value:
                changes[field] = {
                    "from": old_value,
                    "to": new_value,
                }

        if not changes:
            raise HTTPException(status_code=422, detail="Correction contains no changes")

        history_entry = {
            "at": utc_now(),
            "by": body.regularized_by,
            "reason": body.reason,
            "changes": changes,
        }

        update_fields = dict(new_values)
        result = db.attendance_logs.update_one(
            {"_id": record["_id"], "history": record.get("history", [])},
            {
                "$set": update_fields,
                "$push": {"history": history_entry},
            },
        )
        if result.modified_count != 1:
            raise HTTPException(
                status_code=409,
                detail="Attendance record was updated concurrently",
            )

        updated = db.attendance_logs.find_one({"_id": record["_id"]})
        return attendance_response(updated)

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Analytics shared helpers
# ---------------------------------------------------------------------------

def validate_month_or_422(month: str) -> tuple[str, str]:
    try:
        return month_bounds(month)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def validate_date_or_422(value: str, label: str) -> Date:
    try:
        validate_date_string(value, label)
        return Date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def present_day_expression():
    return {
        "$cond": [
            {
                "$and": [
                    {"$in": [{"$dayOfWeek": {"$dateFromString": {"dateString": "$date"}}}, [2, 3, 4, 5, 6]]},
                    {"$in": ["$status", list(PRESENT_STATUSES)]},
                ]
            },
            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
            0.0,
        ]
    }


def attendance_date_match(start_date: str, end_date: str, emp_code: Optional[str] = None):
    match = {"date": {"$gte": start_date, "$lte": end_date}}
    if emp_code:
        match["emp_code"] = emp_code
    return match


def explain_pipeline(collection: Collection, pipeline: list[dict]) -> dict:
    return db.command(
        "explain",
        {
            "aggregate": collection.name,
            "pipeline": pipeline,
            "cursor": {},
        },
        verbosity="executionStats",
    )


# ---------------------------------------------------------------------------
# Employee monthly analytics
# ---------------------------------------------------------------------------

@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str):
    start_date, end_date = validate_month_or_422(month)

    try:
        employee = db.employees.find_one({"emp_code": emp_code})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        joined = Date.fromisoformat(employee["joined_on"])
        period_start = max(Date.fromisoformat(start_date), joined)
        period_end = Date.fromisoformat(end_date)

        working_days = 0
        if period_start <= period_end:
            current = period_start
            while current <= period_end:
                if working_day(current):
                    working_days += 1
                current += timedelta(days=1)

        pipeline = [
            {"$match": attendance_date_match(start_date, end_date, emp_code)},
            {
                "$group": {
                    "_id": None,
                    "present_days": {"$sum": present_day_expression()},
                    "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                    "late_count": {"$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}},
                    "total_late_minutes": {"$sum": "$late_minutes"},
                    "total_overtime_minutes": {"$sum": "$overtime_minutes"},
                }
            },
        ]
        result = list(db.attendance_logs.aggregate(pipeline))
        stats = result[0] if result else {}
        present_days = round_half_up(stats.get("present_days", 0.0), 2)
        attendance_pct = (
            round_half_up((present_days / working_days) * 100, 2)
            if working_days > 0
            else None
        )

        return {
            "emp_code": emp_code,
            "month": month,
            "working_days": working_days,
            "present_days": present_days,
            "leave_days": stats.get("leave_days", 0),
            "late_count": stats.get("late_count", 0),
            "total_late_minutes": stats.get("total_late_minutes", 0),
            "total_overtime_minutes": stats.get("total_overtime_minutes", 0),
            "attendance_pct": attendance_pct,
        }
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Department summary analytics
# ---------------------------------------------------------------------------

@app.get("/analytics/departments/summary")
def department_summary(month: str, department: Optional[str] = None):
    start_date, end_date = validate_month_or_422(month)

    try:
        employee_match = {"joined_on": {"$lte": end_date}}
        if department is not None:
            employee_match["department"] = department

        employees = list(
            db.employees.find(
                employee_match,
                {"_id": 0, "emp_code": 1, "department": 1},
            )
        )
        if not employees:
            return []

        departments = sorted({employee["department"] for employee in employees})
        output = []

        for dept in departments:
            dept_employees = [
                employee for employee in employees if employee["department"] == dept
            ]
            codes = [employee["emp_code"] for employee in dept_employees]
            pipeline = [
                {
                    "$match": {
                        "emp_code": {"$in": codes},
                        "date": {"$gte": start_date, "$lte": end_date},
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "present_days": {"$sum": present_day_expression()},
                        "work_hours_values": {
                            "$push": {
                                "$cond": [
                                    {
                                        "$and": [
                                            {"$in": ["$status", list(PRESENT_STATUSES)]},
                                            {"$ne": ["$work_hours", None]},
                                        ]
                                    },
                                    "$work_hours",
                                    None,
                                ]
                            }
                        },
                        "late_count": {
                            "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                        },
                        "total_late_minutes": {"$sum": "$late_minutes"},
                        "leave_count": {
                            "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                        },
                        "on_duty_count": {
                            "$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}
                        },
                    }
                },
                {
                    "$project": {
                        "present_days": 1,
                        "late_count": 1,
                        "total_late_minutes": 1,
                        "leave_count": 1,
                        "on_duty_count": 1,
                        "avg_work_hours": {
                            "$let": {
                                "vars": {
                                    "values": {
                                        "$filter": {
                                            "input": "$work_hours_values",
                                            "as": "v",
                                            "cond": {"$ne": ["$$v", None]},
                                        }
                                    }
                                },
                                "in": {
                                    "$cond": [
                                        {"$gt": [{"$size": "$$values"}, 0]},
                                        {"$avg": "$$values"},
                                        None,
                                    ]
                                },
                            }
                        },
                    }
                },
            ]
            rows = list(db.attendance_logs.aggregate(pipeline))
            stats = rows[0] if rows else {}
            output.append(
                {
                    "department": dept,
                    "month": month,
                    "headcount": len(dept_employees),
                    "present_days": round_half_up(stats.get("present_days", 0.0), 2),
                    "avg_work_hours": (
                        round_half_up(stats["avg_work_hours"], 2)
                        if stats.get("avg_work_hours") is not None
                        else None
                    ),
                    "late_count": stats.get("late_count", 0),
                    "total_late_minutes": stats.get("total_late_minutes", 0),
                    "leave_count": stats.get("leave_count", 0),
                    "on_duty_count": stats.get("on_duty_count", 0),
                }
            )

        return output
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Late leaderboard
# ---------------------------------------------------------------------------

@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str,
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = None,
):
    start_date, end_date = validate_month_or_422(month)

    try:
        employee_match = {}
        if department is not None:
            employee_match["department"] = department

        employees = list(
            db.employees.find(
                employee_match,
                {"_id": 0, "emp_code": 1, "name": 1, "department": 1},
            )
        )
        if not employees:
            return []

        by_code = {employee["emp_code"]: employee for employee in employees}
        codes = list(by_code)
        pipeline = [
            {
                "$match": {
                    "emp_code": {"$in": codes},
                    "date": {"$gte": start_date, "$lte": end_date},
                    "late_minutes": {"$gt": 0},
                }
            },
            {
                "$group": {
                    "_id": "$emp_code",
                    "total_late_minutes": {"$sum": "$late_minutes"},
                }
            },
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": -1},
                    "output": {"rank": {"$rank": {}}},
                }
            },
            {"$match": {"rank": {"$lte": limit}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
        ]
        rows = list(db.attendance_logs.aggregate(pipeline))
        output = []
        for row in rows:
            employee = by_code.get(row["_id"])
            if employee is None:
                continue
            output.append(
                {
                    "rank": row["rank"],
                    "emp_code": row["_id"],
                    "name": employee["name"],
                    "department": employee["department"],
                    "total_late_minutes": row["total_late_minutes"],
                }
            )
        return output
    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Department daily trend
# ---------------------------------------------------------------------------

@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_date: str = Query(alias="from"),
    to_date: str = Query(alias="to"),
):
    start = validate_date_or_422(from_date, "from")
    end = validate_date_or_422(to_date, "to")

    if end < start:
        raise HTTPException(status_code=422, detail="to must not be before from")
    if (end - start).days + 1 > 92:
        raise HTTPException(status_code=422, detail="Date range must not exceed 92 days")

    try:
        employee_count = db.employees.count_documents({"department": department})
        if employee_count == 0:
            raise HTTPException(status_code=404, detail="Department not found")

        pipeline = [
            {"$match": {"date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}},
            {
                "$lookup": {
                    "from": "employees",
                    "localField": "emp_code",
                    "foreignField": "emp_code",
                    "as": "employee",
                }
            },
            {"$unwind": "$employee"},
            {"$match": {"employee.department": department}},
            {
                "$group": {
                    "_id": "$date",
                    "present_count": {"$sum": present_day_expression()},
                    "late_count": {
                        "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "date": "$_id",
                    "present_count": 1,
                    "late_count": 1,
                }
            },
            {
                "$densify": {
                    "field": "date",
                    "range": {
                        "step": 1,
                        "unit": "day",
                        "bounds": [start.isoformat(), (end + timedelta(days=1)).isoformat()],
                    },
                }
            },
            {
                "$set": {
                    "present_count": {"$ifNull": ["$present_count", 0.0]},
                    "late_count": {"$ifNull": ["$late_count", 0]},
                }
            },
            {
                "$set": {
                    "is_working_day": {
                        "$in": [
                            {
                                "$dayOfWeek": {
                                    "$dateFromString": {"dateString": "$date"}
                                }
                            },
                            [2, 3, 4, 5, 6],
                        ]
                    }
                }
            },
            {
                "$lookup": {
                    "from": "employees",
                    "let": {"trend_date": "$date"},
                    "pipeline": [
                        {"$match": {"department": department}},
                        {
                            "$match": {
                                "$expr": {
                                    "$lte": [
                                        "$joined_on",
                                        "$$trend_date",
                                    ]
                                }
                            }
                        },
                        {"$count": "count"},
                    ],
                    "as": "headcount_data",
                }
            },
            {
                "$set": {
                    "headcount": {
                        "$ifNull": [
                            {"$arrayElemAt": ["$headcount_data.count", 0]},
                            0,
                        ]
                    }
                }
            },
            {
                "$set": {
                    "attendance_rate": {
                        "$cond": [
                            {
                                "$and": [
                                    "$is_working_day",
                                    {"$gt": ["$headcount", 0]},
                                ]
                            },
                            {
                                "$round": [
                                    {
                                        "$multiply": [
                                            {"$divide": ["$present_count", "$headcount"]},
                                            100,
                                        ]
                                    },
                                    4,
                                ]
                            },
                            None,
                        ]
                    }
                }
            },
            {
                "$setWindowFields": {
                    "sortBy": {"date": 1},
                    "output": {
                        "moving_avg_7d": {
                            "$avg": "$attendance_rate",
                            "window": {"documents": [-6, 0]},
                        }
                    },
                }
            },
            {"$sort": {"date": 1}},
            {
                "$project": {
                    "_id": 0,
                    "date": 1,
                    "is_working_day": 1,
                    "headcount": 1,
                    "present_count": {"$round": ["$present_count", 2]},
                    "late_count": 1,
                    "attendance_rate": 1,
                    "moving_avg_7d": {"$round": ["$moving_avg_7d", 4]},
                }
            },
        ]

        return list(db.attendance_logs.aggregate(pipeline))

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")


# ---------------------------------------------------------------------------
# Query-plan explain
# ---------------------------------------------------------------------------

@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str = Path(
        pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$"
    ),
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=50),
    from_date: Optional[str] = Query(default=None, alias="from"),
    to_date: Optional[str] = Query(default=None, alias="to"),
):
    try:
        if endpoint == "attendance_list":
            query = {}
            if emp_code is not None:
                query["emp_code"] = emp_code
            if date_from is not None:
                validate_date_string(date_from, "date_from")
                query.setdefault("date", {})["$gte"] = date_from
            if date_to is not None:
                validate_date_string(date_to, "date_to")
                query.setdefault("date", {})["$lte"] = date_to
            if date_from and date_to and date_from > date_to:
                raise HTTPException(status_code=422, detail="date_from must not be after date_to")
            if status is not None:
                if status not in ALL_STATUSES:
                    raise HTTPException(status_code=422, detail="Invalid attendance status")
                query["status"] = status

            explain = db.command(
                "explain",
                {
                    "find": "attendance_logs",
                    "filter": query,
                    "sort": {"date": -1, "emp_code": 1},
                    "skip": 0,
                    "limit": 20,
                },
                verbosity="executionStats",
            )
            return {
                "endpoint": endpoint,
                "collection": "attendance_logs",
                "explain": explain,
            }

        if month is None:
            raise HTTPException(status_code=422, detail="month is required")
        start_date, end_date = validate_month_or_422(month)

        if endpoint == "employee_monthly":
            if emp_code is None:
                raise HTTPException(status_code=422, detail="emp_code is required")
            pipeline = [
                {"$match": attendance_date_match(start_date, end_date, emp_code)},
                {
                    "$group": {
                        "_id": None,
                        "present_days": {"$sum": present_day_expression()},
                        "late_count": {"$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}},
                    }
                },
            ]
            explain = explain_pipeline(db.attendance_logs, pipeline)
            collection_name = "attendance_logs"

        elif endpoint == "department_summary":
            employee_match = {"joined_on": {"$lte": end_date}}
            if department is not None:
                employee_match["department"] = department
            pipeline = [
                {"$match": {"date": {"$gte": start_date, "$lte": end_date}}},
                {
                    "$lookup": {
                        "from": "employees",
                        "localField": "emp_code",
                        "foreignField": "emp_code",
                        "as": "employee",
                    }
                },
                {"$unwind": "$employee"},
                {"$match": {"employee.joined_on": {"$lte": end_date}}},
                {
                    "$group": {
                        "_id": "$employee.department",
                        "late_count": {"$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}},
                    }
                },
            ]
            if department is not None:
                pipeline.insert(0, {"$match": {"date": {"$gte": start_date, "$lte": end_date}}})
            explain = explain_pipeline(db.attendance_logs, pipeline)
            collection_name = "attendance_logs"

        elif endpoint == "late_leaderboard":
            pipeline = [
                {
                    "$match": {
                        "date": {"$gte": start_date, "$lte": end_date},
                        "late_minutes": {"$gt": 0},
                    }
                },
                {
                    "$group": {
                        "_id": "$emp_code",
                        "total_late_minutes": {"$sum": "$late_minutes"},
                    }
                },
                {"$sort": {"total_late_minutes": -1, "_id": 1}},
            ]
            explain = explain_pipeline(db.attendance_logs, pipeline)
            collection_name = "attendance_logs"

        else:
            if from_date is None or to_date is None:
                raise HTTPException(status_code=422, detail="from and to are required")
            start = validate_date_or_422(from_date, "from")
            end = validate_date_or_422(to_date, "to")
            if end < start or (end - start).days + 1 > 92:
                raise HTTPException(status_code=422, detail="Invalid trend date range")
            pipeline = [
                {"$match": {"date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}},
                {"$group": {"_id": "$date", "count": {"$sum": 1}}},
            ]
            explain = explain_pipeline(db.attendance_logs, pipeline)
            collection_name = "attendance_logs"

        return {
            "endpoint": endpoint,
            "collection": collection_name,
            "explain": explain,
        }

    except HTTPException:
        raise
    except PyMongoError:
        raise HTTPException(status_code=503, detail="Database unavailable")