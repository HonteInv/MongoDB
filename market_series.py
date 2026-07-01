"""
market_series.py — structured ingestion of daily/weekly market time-series data.

The supervisor's market files (Bloomberg exports) are grids of closing levels:
one row per date, one column per instrument. Vector search over rows of numbers
is meaningless (that was the old, broken weekly_vectors), so this stores the data
STRUCTURED — queried by exact date, like pnl_table and daily_table_data.

Handles three input formats, normalizing all of them to the same shape:
    .md   — markdown table, dates as Excel serials, ticker header + name row
    .csv  — dates as M/D/YYYY, descriptive-name headers
    .xlsx — native Excel dates, descriptive-name headers

Storage: one document per (date, frequency):
    { date: "2025-01-03", frequency: "weekly",
      series: { USGG5YR: 4.4119, GDX: 35.0, ... },
      source: "data_010125-103125_weekly.csv" }

Idempotent: upserts on (date, frequency), MERGING the per-instrument series so a
date accumulates instruments from every file that covers it (files have differing
column sets) — re-ingesting never duplicates or drops instruments. Frequency is
inferred from the median gap between dates.
Header→ticker knowledge lives entirely in ticker_map.json; unknown headers are
FLAGGED (never silently dropped) so the map stays maintainable.

Manual run:
    python market_series.py <file>              # ingest one file
    python market_series.py <folder>            # ingest every file in a folder
    python market_series.py <file|folder> --dry-run   # parse + report only, no write
    python market_series.py --verify            # coverage report
"""

import os
import re
import sys
import json
import statistics
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# NOTE: `from build import get_collection` is imported lazily inside the DB
# functions below, so parse_file() works without the langchain/Mongo stack.


# ============================================================
# Constants
# ============================================================

collectionName = "market_series"
tickerMapPath = Path(__file__).with_name("ticker_map.json")

# Excel's day-zero (accounts for the 1900 leap-year bug). Serial 45658 == 2025-01-01.
excelEpoch = datetime(1899, 12, 30)

# Median-gap thresholds (in days) for frequency inference.
dailyMaxGap = 3     # <= 3 day median gap → daily; otherwise → weekly


# ============================================================
# Ticker map
# ============================================================

def normalize_header(text) -> str:
    """Collapse whitespace and upper-case so header spelling/spacing doesn't matter."""
    return re.sub(r"\s+", " ", str(text).replace("﻿", "").strip()).upper()


def load_ticker_map() -> tuple[dict, dict, set]:
    """
    Return (alias_lookup, ticker_meta, ignore_set):
      alias_lookup — normalized header string → canonical ticker
      ticker_meta  — canonical ticker → {type, unit, label}
      ignore_set   — normalized headers to skip silently (junk/broken cells)
    """
    data = json.loads(tickerMapPath.read_text())
    alias_lookup, ticker_meta = {}, {}
    for inst in data["instruments"]:
        t = inst["ticker"]
        ticker_meta[t] = {"type": inst["type"], "unit": inst["unit"], "label": inst["label"]}
        for alias in inst["aliases"]:
            alias_lookup[normalize_header(alias)] = t
    ignore_set = {normalize_header(x) for x in data.get("ignore", [])}
    return alias_lookup, ticker_meta, ignore_set


# ============================================================
# Date + frequency helpers
# ============================================================

def excel_serial_to_iso(serial) -> str:
    """Excel serial number → ISO YYYY-MM-DD."""
    return (excelEpoch + timedelta(days=int(float(serial)))).strftime("%Y-%m-%d")

def to_iso(value) -> str:
    """
    Normalize any supported date encoding to ISO YYYY-MM-DD:
    Excel serial (int/float), M/D/YYYY string, or a real date/Timestamp.
    """
    # Numeric-looking → Excel serial
    s = str(value).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        return excel_serial_to_iso(s)
    # Everything else → let pandas parse (handles M/D/YYYY, ISO, Timestamp, datetime)
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def infer_frequency(iso_dates: list[str]) -> str:
    """Median gap between consecutive distinct dates → 'daily' or 'weekly'."""
    ds = sorted(set(iso_dates))
    gaps = [
        (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days
        for a, b in zip(ds, ds[1:]) if b > a
    ]
    if not gaps:
        return "weekly"
    return "daily" if statistics.median(gaps) <= dailyMaxGap else "weekly"


# ============================================================
# Format readers — each returns (headers, date_col_values, data_rows)
# ============================================================

def looks_like_date(value) -> bool:
    """True if the cell parses as a date/serial — used to tell a data row from a label row."""
    try:
        to_iso(value)
        return True
    except Exception:
        return False


def read_markdown(path: Path):
    """
    Read a markdown table, auto-detecting which row holds the descriptive labels.
    Two layouts occur in the exports:
      A) header row = Bloomberg tickers, FIRST body row = descriptive names, rest = data
      B) header row = descriptive names directly, body = data
    Detection uses the first body cell: if it's NOT a date, that row is a label row
    (Layout A) and real data starts after it; otherwise the header holds the labels
    (Layout B). Robust to future spellings since it keys off the date column, not names.
    """
    lines = [l for l in path.read_text().splitlines() if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    if len(rows) < 3:
        raise ValueError(f"{path.name}: not enough rows for a table")

    header = rows[0]                 # markdown header
    body = rows[2:]                  # rows[1] is the '---' separator

    if body and not looks_like_date(body[0][0]):
        # Layout A: the first body row is descriptive names; data starts after it.
        labels = body[0][1:]
        data_rows = body[1:]
    else:                            # Layout B: names already in the header.
        labels = header[1:]
        data_rows = body

    dates = [r[0] for r in data_rows]
    data = [r[1:] for r in data_rows]
    return labels, dates, data


def read_table(path: Path):
    """CSV or XLSX — first column is the date, remaining columns are instruments."""
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    headers = list(df.columns[1:])
    dates = list(df.iloc[:, 0])
    data = df.iloc[:, 1:].values.tolist()
    return headers, dates, data


# ============================================================
# Parse (no DB) — the shared normalization core
# ============================================================

def parse_file(path) -> dict:
    """
    Parse one file into normalized documents WITHOUT touching the database.
    Returns { docs, frequency, unmapped_headers, date_range, source }.
    docs = [ {date, frequency, series:{ticker:value}, source}, ... ]
    """
    path = Path(path)
    alias_lookup, _, ignore_set = load_ticker_map()

    if path.suffix.lower() == ".md":
        headers, raw_dates, data = read_markdown(path)
    elif path.suffix.lower() in (".csv", ".xlsx", ".xls"):
        headers, raw_dates, data = read_table(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Resolve each header to a key:
    #   - ignored/junk headers  → None (skipped silently)
    #   - a mapped alias        → canonical ticker
    #   - anything else         → STORED under its raw header text (and reported)
    col_key, unmapped = [], []
    for h in headers:
        nh = normalize_header(h)
        if nh in ignore_set or not str(h).strip():
            col_key.append(None)
            continue
        t = alias_lookup.get(nh)
        if t:
            col_key.append(t)
        else:
            # Store anyway, under the raw header — but strip '.'/'$' which are
            # illegal in Mongo field names.
            key = re.sub(r"[.$]", " ", str(h).strip()).strip()
            col_key.append(key)
            unmapped.append(key)

    iso_dates = [to_iso(d) for d in raw_dates]
    frequency = infer_frequency(iso_dates)
    source = path.name

    docs = []
    for iso, row in zip(iso_dates, data):
        series = {}
        for key, val in zip(col_key, row):
            if key is None:
                continue
            if val is None or (isinstance(val, float) and pd.isna(val)) or str(val).strip() == "":
                continue
            try:
                series[key] = float(val)
            except (TypeError, ValueError):
                continue
        if series:
            docs.append({"date": iso, "frequency": frequency,
                         "series": series, "source": source})

    return {
        "docs": docs,
        "frequency": frequency,
        "unmapped_headers": sorted(set(unmapped)),   # stored under raw name; map later if desired
        "date_range": (iso_dates[0], iso_dates[-1]) if iso_dates else (None, None),
        "source": source,
    }


# ============================================================
# Ingest (DB) — idempotent upsert
# ============================================================

def ensure_index():
    """Plain unique compound index on (date, frequency) — the idempotency guard. No vector index."""
    from build import get_collection
    get_collection(collectionName).create_index(
        [("date", 1), ("frequency", 1)], unique=True
    )


def ingest_file(path, source_name: str = None, dry_run: bool = False) -> dict:
    """
    Parse a file and upsert its documents into market_series. source_name overrides
    the stored source filename (for Streamlit temp-path uploads). Returns a report.
    """
    parsed = parse_file(path)
    if source_name:
        for d in parsed["docs"]:
            d["source"] = source_name
        parsed["source"] = source_name

    report = {
        "source": parsed["source"],
        "frequency": parsed["frequency"],
        "date_range": parsed["date_range"],
        "rows": len(parsed["docs"]),
        "unmapped_headers": parsed["unmapped_headers"],
        "written": 0,
        "dry_run": dry_run,
    }

    if parsed["unmapped_headers"]:
        print(f"  ⓘ unmapped columns stored under raw names (map in ticker_map.json to label/unit them): "
              f"{parsed['unmapped_headers']}")

    if dry_run:
        print(f"  [dry-run] {report['rows']} rows, {parsed['frequency']}, "
              f"{parsed['date_range'][0]}..{parsed['date_range'][1]} — nothing written")
        return report

    from build import get_collection
    ensure_index()
    col = get_collection(collectionName)
    for d in parsed["docs"]:
        # MERGE the series sub-doc rather than replacing it, so a date accumulates
        # instruments from every file that covers it (files have differing column
        # sets). Pipeline update + $mergeObjects keeps arbitrary keys safe. `source`
        # records the most recent file that touched the date.
        col.update_one(
            {"date": d["date"], "frequency": d["frequency"]},
            [{"$set": {
                "date": d["date"],
                "frequency": d["frequency"],
                "source": d["source"],
                "series": {"$mergeObjects": [{"$ifNull": ["$series", {}]}, d["series"]]},
            }}],
            upsert=True,
        )
    report["written"] = len(parsed["docs"])
    print(f"  ✓ {report['written']} rows upserted ({parsed['frequency']}, "
          f"{parsed['date_range'][0]}..{parsed['date_range'][1]}) from {parsed['source']}")
    return report


# ============================================================
# Query + verify
# ============================================================

def ingest_folder(folder, dry_run: bool = False) -> list[dict]:
    """
    Ingest every supported file (.md/.csv/.xlsx/.xls) in a folder. Returns a list
    of per-file reports and prints a summary. Idempotent, so safe to re-run.
    """
    folder = Path(folder)
    files = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in (".md", ".csv", ".xlsx", ".xls")
    )
    if not files:
        print(f"No .md/.csv/.xlsx/.xls files found in {folder}")
        return []

    print(f"Found {len(files)} file(s) in {folder}{'  [DRY RUN]' if dry_run else ''}\n")
    reports, all_unmapped = [], set()
    for f in files:
        print(f"• {f.name}")
        try:
            rep = ingest_file(f, dry_run=dry_run)
            reports.append(rep)
            all_unmapped.update(rep["unmapped_headers"])
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

    total_rows = sum(r["written"] if not dry_run else r["rows"] for r in reports)
    print(f"\n=== SUMMARY ===")
    print(f"files: {len(reports)}/{len(files)}   rows {'parsed' if dry_run else 'written'}: {total_rows}")
    if all_unmapped:
        print(f"ⓘ unmapped columns across folder (stored under raw names; map in ticker_map.json "
              f"to give them a label/unit): {sorted(all_unmapped)}")
    else:
        print("all columns mapped ✓")
    return reports


def get_range(start_iso: str, end_iso: str, frequency: str = None) -> list[dict]:
    """Fetch documents in [start, end], newest instruments merged per date. Prefers daily."""
    from build import get_collection
    q = {"date": {"$gte": start_iso, "$lte": end_iso}}
    if frequency:
        q["frequency"] = frequency
    return list(get_collection(collectionName).find(q, {"_id": 0}).sort("date", 1))


def verify():
    from build import get_collection
    col = get_collection(collectionName)
    total = col.count_documents({})
    print(f"\n=== VERIFY: {collectionName} ===")
    print(f"total documents: {total}")
    for freq in ("daily", "weekly"):
        docs = list(col.find({"frequency": freq}, {"date": 1, "_id": 0}).sort("date", 1))
        if not docs:
            print(f"  {freq}: none")
            continue
        print(f"  {freq}: {len(docs)} rows  {docs[0]['date']}..{docs[-1]['date']}")
    print(f"sources: {col.distinct('source')}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n"
              "  python market_series.py <file> [--dry-run]\n"
              "  python market_series.py <folder> [--dry-run]\n"
              "  python market_series.py --verify")
        sys.exit(1)

    if sys.argv[1] == "--verify":
        verify()
    else:
        target = Path(sys.argv[1])
        dry = "--dry-run" in sys.argv
        if target.is_dir():
            ingest_folder(target, dry_run=dry)
        else:
            ingest_file(target, dry_run=dry)
