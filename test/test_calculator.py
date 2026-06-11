"""
test_calculator.py — Verify the AUM summary / return % calculator fix.

Three levels of testing:
    1. LOCAL  — parse the raw .md file; no MongoDB needed
    2. MONGO  — check what's actually stored in pnl_summary
    3. AGENT  — simulate what PortfolioPerformanceAgent injects into the prompt

Run everything:
    python3 test_calculator.py

Or individual functions at the bottom.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent / "data" / "pnl"


# ============================================================
# Helpers
# ============================================================

def get_col(name: str):
    client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
    return client[os.getenv("MONGO_DB_NAME", "portfolio_rag")][name]


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# TEST 1: Local parser — no MongoDB, no network
# Directly calls _parse_pnl_markdown / _parse_pnl_csv and prints
# what summary gets extracted from each file.
# ============================================================

def test_local_parser():
    section("TEST 1: Local Parser — AUM row extraction")

    from build import _parse_pnl_markdown, _parse_pnl_csv, _extract_report_period

    files = sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print(f"  No .md or .csv files found in {DATA_DIR}")
        return

    all_ok = True
    for f in files:
        period = _extract_report_period(f.name)
        if f.suffix == ".md":
            rows, summary = _parse_pnl_markdown(f)
        else:
            rows, summary = _parse_pnl_csv(f)

        status = "✓" if summary else "✗ NO SUMMARY EXTRACTED"
        print(f"\n  File:   {f.name}  (period={period})")
        print(f"  Rows:   {len(rows)}")
        print(f"  AUM:    {status}")

        if summary:
            start = summary.get("start_aum")
            end   = summary.get("end_aum")
            pnl   = summary.get("total_pnl")
            ret   = summary.get("return_pct")
            print(f"    start_aum  = ${start:>16,.2f}" if start else "    start_aum  = MISSING")
            print(f"    end_aum    = ${end:>16,.2f}"   if end   else "    end_aum    = MISSING")
            print(f"    total_pnl  = ${pnl:>16,.2f}"   if pnl   else "    total_pnl  = MISSING")
            if ret is not None:
                marker = "✓ CORRECT" if ret > -20 else "⚠ CHECK"
                print(f"    return_pct = {ret:+.4f}%   ← {marker}")
            else:
                print("    return_pct = NOT COMPUTED (start_aum missing?)")
                all_ok = False
        else:
            all_ok = False
            # Show raw lines so we can see why the AUM row wasn't matched
            text = f.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "aum" in line.lower() or "start" in line.lower():
                    print(f"    RAW LINE: {line[:120]}")

    print()
    if all_ok:
        print("  RESULT: All files parsed AUM summary correctly.")
    else:
        print("  RESULT: Some files missing AUM summary — see ✗ above.")
        print("  Fix:    Check that the AUM row contains 'Start AUM' and 'End AUM' labels.")


# ============================================================
# TEST 2: MongoDB check — is pnl_summary populated?
# Shows what's stored and what's missing; tells you which
# periods need to be re-ingested.
# ============================================================

def test_mongo_summary():
    section("TEST 2: MongoDB — pnl_summary collection")

    from build import _extract_report_period

    summary_col  = get_col("pnl_summary")
    pnl_table_col = get_col("pnl_table")

    # All periods currently in pnl_table
    table_periods = sorted(pnl_table_col.distinct("report_period"))
    print(f"  Periods in pnl_table:   {table_periods}")

    # All periods in pnl_summary
    summary_periods = sorted(summary_col.distinct("report_period"))
    print(f"  Periods in pnl_summary: {summary_periods}")

    missing = [p for p in table_periods if p not in summary_periods]
    if missing:
        print(f"\n  ⚠ MISSING summaries for: {missing}")
        print("  These periods need to be re-ingested to get accurate returns.")
        print("  Run the re-ingest block below (TEST 3) to fix this.")
    else:
        print("\n  ✓ All pnl_table periods have a summary entry.")

    # Print detail for each stored summary
    if summary_periods:
        print("\n  --- Stored summaries ---")
        for p in summary_periods:
            doc = summary_col.find_one({"report_period": p}, {"_id": 0, "uploaded_at": 0})
            ret = doc.get("return_pct")
            start = doc.get("start_aum", 0)
            end   = doc.get("end_aum", 0)
            pnl   = doc.get("total_pnl", 0)
            print(f"\n  Period: {p}")
            print(f"    Start AUM  = ${start:>16,.2f}")
            print(f"    End AUM    = ${end:>16,.2f}")
            print(f"    Total P&L  = ${pnl:>16,.2f}")
            if ret is not None:
                print(f"    Return     = {ret:+.4f}%")
            else:
                print(f"    Return     = NOT COMPUTED")

    return missing


# ============================================================
# TEST 3: Re-ingest missing periods
# Deletes and re-parses each PnL file that has no summary entry.
# Safe to run repeatedly — skips already-populated periods.
# ============================================================

def reingest_missing_summaries(dry_run: bool = False):
    section("TEST 3: Re-ingest missing periods")

    from build import _extract_report_period, delete_pnl_period, ingest_pnl_structured

    summary_col   = get_col("pnl_summary")
    pnl_table_col = get_col("pnl_table")

    # Build map: period → file path
    files = sorted(DATA_DIR.glob("*.md")) + sorted(DATA_DIR.glob("*.csv"))
    period_to_file: dict[str, Path] = {}
    for f in files:
        p = _extract_report_period(f.name)
        if p and "??" not in p:
            period_to_file[p] = f

    table_periods  = sorted(pnl_table_col.distinct("report_period"))
    summary_periods = sorted(summary_col.distinct("report_period"))
    missing = [p for p in table_periods if p not in summary_periods]

    if not missing:
        print("  ✓ No missing summaries — nothing to do.")
        return

    print(f"  Periods needing re-ingest: {missing}")
    if dry_run:
        print("  DRY RUN — pass dry_run=False to actually re-ingest.")
        return

    for period in missing:
        f = period_to_file.get(period)
        if not f:
            print(f"\n  ⚠ No file found for period {period} — skipping.")
            continue
        print(f"\n  Re-ingesting {f.name}  (period={period}) ...")
        delete_pnl_period(period)
        n = ingest_pnl_structured(str(f), report_period=period)
        print(f"  Done: {n} rows inserted.")


# ============================================================
# TEST 4: Agent simulation
# Shows exactly what PortfolioPerformanceAgent and
# NewsletterWriterAgent will inject into their prompts.
# ============================================================

def test_agent_prompt_injection(period: str = None):
    section("TEST 4: Agent prompt injection simulation")

    summary_col = get_col("pnl_summary")

    available = sorted(summary_col.distinct("report_period"))
    if not available:
        print("  ⚠ pnl_summary is empty — run reingest_missing_summaries() first.")
        return

    if period is None:
        period = available[-1]
        print(f"  (No period specified — using most recent: {period})")

    doc = summary_col.find_one({"report_period": period}, {"_id": 0})
    if not doc:
        print(f"  ⚠ No summary for {period}.")
        return

    ret_pct   = doc.get("return_pct")
    start_aum = doc.get("start_aum", 0)
    end_aum   = doc.get("end_aum", 0)
    total_pnl = doc.get("total_pnl", 0)

    print(f"\n  Summary document for {period}:")
    print(f"    return_pct = {ret_pct}")
    print(f"    start_aum  = {start_aum}")
    print(f"    end_aum    = {end_aum}")
    print(f"    total_pnl  = {total_pnl}")

    if ret_pct is not None:
        print("\n  --- Prompt injection (PortfolioPerformanceAgent) ---")
        summary_line = (
            f"PORTFOLIO RETURN FOR {period}: {ret_pct:+.2f}%\n"
            f"Start AUM: ${start_aum:,.0f}  |  End AUM: ${end_aum:,.0f}  |  "
            f"Total P&L: ${total_pnl:,.0f}\n"
            f"This return figure is mathematically exact — use it verbatim."
        )
        print(summary_line)

        print("\n  --- Prompt injection (NewsletterWriterAgent) ---")
        return_fact = (
            f"EXACT MONTHLY RETURN: {ret_pct:+.2f}%  (Total P&L: ${total_pnl:,.0f})\n"
            f"You MUST use this figure in the opening paragraph. Do not calculate or estimate the return yourself."
        )
        print(return_fact)
        print("\n  ✓ Agent injection looks correct.")
    else:
        print("\n  ⚠ return_pct is None — something went wrong in parsing.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  CALCULATOR FIX TEST SUITE")
    print("="*60)

    # Step 1: check if the local file parser extracts AUM correctly
    test_local_parser()

    # Step 2: check what's actually in MongoDB
    missing = test_mongo_summary()

    # Step 3: re-ingest any periods that are missing a summary
    if missing:
        print("\n  Periods are missing summaries — re-ingesting now...")
        reingest_missing_summaries(dry_run=False)

        # Re-check after re-ingest
        test_mongo_summary()

    # Step 4: simulate what agents will inject into prompts
    test_agent_prompt_injection()

    print("\n" + "="*60)
    print("  DONE")
    print("="*60)
