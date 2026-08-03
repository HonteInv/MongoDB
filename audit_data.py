"""
audit_data.py — READ-ONLY audit of MongoDB collections for residue of fixed
ingestion bugs. Never writes, updates, or deletes anything.

Background: the 2026-08-02 hardening pass fixed several parser bugs that had
been corrupting data ON THE WAY IN (see the codebase-hardening commit). Data
ingested BEFORE that fix may still carry the damage. This script finds it and
tells you exactly which periods/files to re-ingest via the admin app.

Checks:
    pnl_table         A1 date-string corruption in non-date columns
                         (Excel-serial repair used to rewrite bare 5-digit
                          position sizes in 40000-59999 into dates)
                      A2 literal "None" cells (ragged CSV rows)
                      A3 malformed report_period ("2026-??", "", bad format)
                      A4 stored period disagrees with the FIXED filename parser
                      A5 row-count reconciliation vs the source file in data/pnl
    pnl_summary       B1 periods in pnl_table with no summary doc
                         (CSV uploads never wrote one before the fix)
                      B2 summary source_file != pnl_table source_file
                         (re-upload overwrote one but not the other)
                      B3 AUM disagreement vs re-parsing the local source file
    context_daily     C1 docs missing report_month/content_type — written by
                         the generic vector path, INVISIBLE to the market agent
    daily_table_data  D1 quality.confidence None or 0 (was skipped by the old
                         falsy-zero check), plus incomplete/low-confidence
                      D2 duplicate (report_day, exhibit) pairs
    market_series     E1 same date stored under multiple frequencies (callers
                         must filter — informational)
                      E2 non-ISO date values
    users             F1 roles outside {guest, admin} (pre-whitelist writes)
                      F2 password_hash stored as string (breaks bcrypt check)

Usage:
    python audit_data.py                # full audit, console report
    python audit_data.py --json out.json   # also write machine-readable report

Requires MONGO_URI_USER (and optionally MONGO_URI_ADMIN for the users check)
in .env / environment. Runs the file-side checks (A4/A5/B3 parsing) even
without DB access so it can be smoke-tested anywhere.
"""

import os
import re
import sys
import json
import argparse
import types
from pathlib import Path
from collections import Counter, defaultdict

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PNL_DIR = BASE_DIR / "data" / "pnl"

# ── import build.py's real parsers without requiring the langchain stack ────
# On a full deployment the real packages exist and these setdefault stubs are
# no-ops. On a bare machine they let us reuse the exact production parsing
# code (no drift) for the file-side checks.
for _name in ("langchain_core", "langchain_core.documents", "langchain_text_splitters",
              "langchain_mongodb", "langchain_community",
              "langchain_community.document_loaders",
              "langchain_google_genai", "langchain_openai"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
for _mod, _attr in (("langchain_core.documents", "Document"),
                    ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
                    ("langchain_mongodb", "MongoDBAtlasVectorSearch"),
                    ("langchain_community.document_loaders", "PyPDFLoader"),
                    ("langchain_community.document_loaders", "CSVLoader"),
                    ("langchain_community.document_loaders", "TextLoader")):
    if not hasattr(sys.modules[_mod], _attr):
        setattr(sys.modules[_mod], _attr, object)
for _mod, _attr in (("langchain_google_genai", "GoogleGenerativeAIEmbeddings"),
                    ("langchain_openai", "OpenAIEmbeddings")):
    if not hasattr(sys.modules[_mod], _attr):
        setattr(sys.modules[_mod], _attr, lambda *a, **k: None)

import build  # noqa: E402  (uses the stubs above only where packages are absent)

DB_NAME = os.getenv("MONGO_DB_NAME", "portfolio_rag")
ISO_DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
PERIOD_RE = re.compile(r"^20\d{2}-(0[1-9]|1[0-2])$")
PNL_META_KEYS = {"report_period", "source_file", "uploaded_by", "uploaded_at", "_id"}

# findings: list of dicts {check, severity, subject, detail, action}
findings: list[dict] = []


def add(check: str, severity: str, subject: str, detail: str, action: str):
    findings.append({"check": check, "severity": severity, "subject": subject,
                     "detail": detail, "action": action})


def parse_local_file(path: Path):
    """(rows, summary) via build.py's fixed parsers; None on failure."""
    try:
        if path.suffix.lower() == ".csv":
            return build._parse_pnl_csv(path)
        return build._parse_pnl_markdown(path)
    except Exception as e:
        add("A5", "warn", path.name, f"local file failed to parse: {e}",
            "inspect the file; the reconciliation check was skipped for it")
        return None


def local_file_for(source_file: str) -> Path | None:
    """Find the checked-in source file matching a stored source_file name."""
    if not source_file:
        return None
    candidate = PNL_DIR / Path(source_file).name
    return candidate if candidate.exists() else None


# ============================================================
# A + B — pnl_table / pnl_summary
# ============================================================

def audit_pnl(db):
    rows = list(db["pnl_table"].find({}, {"_id": 0}))
    summaries = {s.get("report_period"): s
                 for s in db["pnl_summary"].find({}, {"_id": 0})}
    print(f"  pnl_table: {len(rows)} rows, pnl_summary: {len(summaries)} docs")

    by_period: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_period[str(r.get("report_period"))].append(r)

    for period, prows in sorted(by_period.items()):
        source_files = Counter(r.get("source_file") for r in prows)
        src = source_files.most_common(1)[0][0] if source_files else None

        # A3 — malformed period key
        if not PERIOD_RE.fullmatch(period or ""):
            add("A3", "error", f"period '{period}' ({len(prows)} rows, {src})",
                "report_period is not YYYY-MM — no query path can ever match these rows",
                "re-ingest the file via admin app with the correct period, then "
                f"delete the bad key (delete_pnl_period('{period}'))")

        # A1 — date-string corruption in non-date columns
        corrupted = []
        for r in prows:
            for k, v in r.items():
                if k in PNL_META_KEYS or "date" in k:
                    continue
                if isinstance(v, str) and ISO_DATE_RE.fullmatch(v.strip()):
                    corrupted.append(f"{r.get('positions', '?')}.{k}={v}")
        if corrupted:
            add("A1", "error", f"period {period} ({src})",
                f"{len(corrupted)} non-date cell(s) contain date strings — the old "
                f"whole-file Excel-serial repair rewrote 5-digit values: "
                f"{corrupted[:3]}{'...' if len(corrupted) > 3 else ''}",
                "re-ingest this file via admin app in Reindex mode")

        # A2 — literal "None" cells
        none_cells = sum(1 for r in prows for k, v in r.items()
                         if k not in PNL_META_KEYS and v == "None")
        if none_cells:
            add("A2", "warn", f"period {period} ({src})",
                f"{none_cells} cell(s) contain the literal string 'None' (ragged CSV rows)",
                "re-ingest this file via admin app in Reindex mode")

        # A4 — stored period vs fixed filename parser
        if src:
            detected = build._extract_report_period(src)
            if PERIOD_RE.fullmatch(detected) and detected != period:
                add("A4", "error", f"period {period} ({src})",
                    f"filename says {detected} but rows are stored under {period} — "
                    f"the old parser defaulted to the current year for underscore filenames",
                    f"re-ingest {src} with period {detected}, then delete_pnl_period('{period}')")

        # A5 — row-count reconciliation vs local source file
        local = local_file_for(src)
        parsed = parse_local_file(local) if local else None
        if parsed:
            file_rows, file_summary = parsed
            if len(file_rows) != len(prows):
                add("A5", "warn", f"period {period} ({src})",
                    f"DB has {len(prows)} rows but re-parsing {local.name} yields "
                    f"{len(file_rows)} — ingested before a parser fix, or file changed since",
                    "re-ingest via admin app in Reindex mode to align")

            # B3 — AUM reconciliation
            summ = summaries.get(period)
            if file_summary and summ:
                for key in ("start_aum", "end_aum"):
                    fv, dv = file_summary.get(key), summ.get(key)
                    if fv is not None and dv is not None and abs(fv - dv) > 0.01:
                        add("B3", "error", f"period {period}",
                            f"{key}: file says {fv:,.0f}, pnl_summary says {dv:,.0f}",
                            "repair via admin app Tab 6, or re-ingest in Reindex mode")

        # B1 — missing summary
        if PERIOD_RE.fullmatch(period or "") and period not in summaries:
            add("B1", "error", f"period {period} ({src})",
                "no pnl_summary doc — return_pct unavailable to every agent for this "
                "period (CSV uploads never wrote one before the fix)",
                "re-ingest in Reindex mode, or use the admin app AUM repair (Tab 6)")

        # B2 — summary/table source mismatch
        summ = summaries.get(period)
        if summ and src and summ.get("source_file") and summ["source_file"] != src:
            add("B2", "warn", f"period {period}",
                f"pnl_table rows come from '{src}' but pnl_summary from "
                f"'{summ['source_file']}' — a re-upload updated one but not the other",
                "re-ingest the correct file in Reindex mode so both match")

    # orphaned summaries
    for period in summaries:
        if period not in by_period:
            add("B2", "warn", f"period {period}",
                "pnl_summary exists but pnl_table has no rows for it",
                f"delete_pnl_period('{period}') or re-ingest the file")


# ============================================================
# C — context_daily writer-schema audit
# ============================================================

def audit_context(db):
    col = db["context_daily"]
    total = col.count_documents({})
    missing_month = col.count_documents({"report_month": {"$exists": False}})
    missing_ctype = col.count_documents({"content_type": {"$exists": False}})
    has_period = col.count_documents({"report_period": {"$exists": True}})
    print(f"  context_daily: {total} docs")
    if missing_month or missing_ctype:
        n = max(missing_month, missing_ctype)
        srcs = col.distinct("source", {"$or": [{"report_month": {"$exists": False}},
                                               {"content_type": {"$exists": False}}]})
        add("C1", "error", f"context_daily ({n} docs, {len(srcs)} sources)",
            f"{missing_month} docs lack report_month, {missing_ctype} lack content_type — "
            f"written by the generic vector path, INVISIBLE to the market agent's "
            f"period-filtered retrieval. Sources: {srcs[:5]}",
            "delete these sources and re-ingest the PDFs through the admin app "
            "context pipeline (daily_data)")
    if has_period:
        add("C1", "info", f"context_daily ({has_period} docs)",
            "docs carry a report_period field (backfill artifact) alongside the "
            "daily pipeline's report_month — two competing period fields",
            "harmless to readers (they filter on report_month), flag to Jade")


# ============================================================
# D — daily_table_data quality
# ============================================================

def audit_tables(db):
    col = db["daily_table_data"]
    total = col.count_documents({})
    print(f"  daily_table_data: {total} docs")
    seen = Counter()
    for doc in col.find({}, {"report_day": 1, "exhibit": 1, "quality": 1, "_id": 0}):
        key = (doc.get("report_day"), doc.get("exhibit"))
        seen[key] += 1
        q = doc.get("quality") or {}
        conf = q.get("confidence")
        if conf is None or conf == 0:
            add("D1", "warn", f"{key[0]} / {key[1]}",
                f"quality.confidence is {conf} — the old falsy-zero check let these "
                f"pass as clean; the vision reads never agreed on this table",
                "re-ingest the source PDF (Reindex mode) or treat this table's "
                "numbers as unverified")
        elif q.get("incomplete") or conf < 95:
            add("D1", "info", f"{key[0]} / {key[1]}",
                f"low confidence ({conf}%) or incomplete={q.get('incomplete')}",
                "spot-check against the PDF if this period matters")
    for key, n in seen.items():
        if n > 1:
            add("D2", "error", f"{key[0]} / {key[1]}",
                f"{n} documents for the same (report_day, exhibit) — readers pick "
                f"one arbitrarily",
                "delete both via admin app (context delete removes by source) and "
                "re-ingest the correct PDF")


# ============================================================
# E — market_series
# ============================================================

def audit_market(db):
    col = db["market_series"]
    total = col.count_documents({})
    print(f"  market_series: {total} docs")
    freq_by_date = defaultdict(set)
    for doc in col.find({}, {"date": 1, "frequency": 1, "_id": 0}):
        d = doc.get("date")
        freq_by_date[d].add(doc.get("frequency"))
        if not (isinstance(d, str) and ISO_DATE_RE.fullmatch(d)):
            add("E2", "error", str(d),
                f"date is not an ISO string ({type(d).__name__}: {d!r}) — "
                f"string range queries will mis-order or miss it",
                "re-ingest the source file via admin app")
    multi = {d: f for d, f in freq_by_date.items() if len(f) > 1}
    if multi:
        add("E1", "info", f"{len(multi)} dates",
            f"stored under multiple frequencies (e.g. {dict(list(multi.items())[:2])}) — "
            f"expected when daily and weekly files overlap; get_range(frequency=None) "
            f"returns both",
            "no action — current callers filter by frequency")


# ============================================================
# F — users
# ============================================================

def audit_users(admin_db):
    col = admin_db["users"]
    n = col.count_documents({})
    print(f"  users: {n} docs")
    for u in col.find({}, {"username": 1, "role": 1, "password_hash": 1, "_id": 0}):
        role = u.get("role")
        if role not in ("guest", "admin"):
            add("F1", "error", u.get("username", "?"),
                f"role is {role!r} — is_admin() compares against exactly 'admin', "
                f"so this user may be silently locked out (or was never valid)",
                "fix via admin app User Management or change_role()")
        if isinstance(u.get("password_hash"), str):
            add("F2", "error", u.get("username", "?"),
                "password_hash stored as string, not BSON binary — inserted outside "
                "create_user(); login tolerates it now but it should be re-set",
                "reset the password via admin app / change_password()")


# ============================================================
# Report
# ============================================================

SEV_ORDER = {"error": 0, "warn": 1, "info": 2}
SEV_LABEL = {"error": "ERROR", "warn": "WARN ", "info": "info "}


def print_report(json_path: str | None):
    print("\n" + "=" * 72)
    if not findings:
        print("CLEAN — no residue of the fixed bugs found in any collection.")
    else:
        counts = Counter(f["severity"] for f in findings)
        print(f"FINDINGS: {counts.get('error', 0)} error(s), "
              f"{counts.get('warn', 0)} warning(s), {counts.get('info', 0)} info")
        print("=" * 72)
        for f in sorted(findings, key=lambda x: (SEV_ORDER[x["severity"]], x["check"])):
            print(f"\n[{SEV_LABEL[f['severity']]}] {f['check']}  {f['subject']}")
            print(f"    what:   {f['detail']}")
            print(f"    action: {f['action']}")
    print("\n" + "=" * 72)
    print("This audit is read-only. All fixes go through the admin app (Reindex\n"
          "mode re-ingests are idempotent: they delete the period/source first).")
    if json_path:
        Path(json_path).write_text(json.dumps(findings, indent=2, default=str),
                                   encoding="utf-8")
        print(f"JSON report written to {json_path}")


def main():
    ap = argparse.ArgumentParser(description="Read-only data audit (see module docstring)")
    ap.add_argument("--json", metavar="PATH", help="also write findings as JSON")
    args = ap.parse_args()

    uri_user = os.getenv("MONGO_URI_USER")
    uri_admin = os.getenv("MONGO_URI_ADMIN")
    if not uri_user and not uri_admin:
        print("No MONGO_URI_USER / MONGO_URI_ADMIN in the environment (.env).")
        print("Run this where the app runs (deployment shell, or a machine with .env).")
        sys.exit(2)

    from pymongo import MongoClient
    import certifi

    print(f"Auditing database '{DB_NAME}' (read-only)...")
    client = MongoClient(uri_user or uri_admin, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    audit_pnl(db)
    audit_context(db)
    audit_tables(db)
    audit_market(db)

    if uri_admin:
        audit_users(MongoClient(uri_admin, tlsCAFile=certifi.where())[DB_NAME])
    else:
        print("  users: skipped (MONGO_URI_ADMIN not set)")

    print_report(args.json)
    sys.exit(1 if any(f["severity"] == "error" for f in findings) else 0)


if __name__ == "__main__":
    main()
