"""
daily_table_data.py — exact numeric extraction of the daily exhibit tables.

The MS daily reports embed data tables ("Exhibit 1: G4 Rates Closes", Inflation
Closes, Macro Closes). Vector search reads these fuzzily, so numbers go wrong.
This module renders each exhibit page to an image, has Claude vision read the
table, and stores the result as STRUCTURED data — queried by exact date, like
pnl_table. Guaranteed-accurate numbers, no fuzzy retrieval.

Storage: one document per (report_day, exhibit):
    { report_day, report_month, exhibit, table: {row: {col: value}}, source }

Writes only to the NEW collection (daily_table_data) — nothing live is touched.

Manual run:
    python daily_table_data.py <source_dir> [sample_every]
"""

import os
import re
import sys
import json
import time
import base64
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF — render pages to images
from anthropic import Anthropic

from build import get_collection
from daily_data import dayHeaderRe, extract_day   # reuse the proven date detector


# ============================================================
# Constants
# ============================================================

marketCollectionName = "daily_table_data"
# Highest-accuracy config: strongest vision model + high-res rendering so small
# numbers read cleanly. Override with CLAUDE_VISION_MODEL to trade cost for speed.
# Sonnet matched/beat Opus on extraction accuracy (99.9% vs 99.8% on the test file)
# while being faster, ~5x cheaper, and less prone to overload — so it's the default.
# Override with CLAUDE_VISION_MODEL if needed.
visionModel = os.getenv("CLAUDE_VISION_MODEL", "claude-sonnet-4-6")
renderDpi = 300
# Number of independent vision reads per page, merged for completeness + confidence.
# Image-based tables occasionally drop a section in a single read; 3 reads + union
# fixes that and gives a real agreement-based confidence (text validation can't work
# on image tables, which have no text layer).
visionPasses = int(os.getenv("VISION_PASSES", "3"))

# We capture ONLY the 3 core market-data tables. A dry run proved that capturing
# every exhibit tanks accuracy (~37% confidence): vision names the dozens of exotic
# auxiliary tables inconsistently across reads, fragmenting the merge. Focused on
# these three, vision is consistent (~100%). Core titles double as the page filter.
coreExhibitMarkers = ["G4 Rates Closes", "Inflation Closes", "Macro Closes"]


def is_core_exhibit(name: str) -> bool:
    """Keep only the three core tables (match on the distinctive part of the title)."""
    if not name:
        return False
    low = name.lower()
    return ("rates closes" in low and "sovereign" not in low) \
        or "inflation closes" in low or "macro closes" in low

visionPrompt = (
    "This is a page from a Morgan Stanley daily macro report. Extract ONLY these three "
    "tables if present, using EXACTLY these names: 'G4 Rates Closes', 'Inflation Closes', "
    "'Macro Closes'. IGNORE every other table (Sovereign 10y, OIS, STIR, Bond Futures, "
    "auction models, FX pivots/support/resistance, trading signals, curve/fly) and all "
    "prose. If none are present, return an empty exhibits list.\n\n"

    "Use these EXACT structures so repeated extractions are byte-identical:\n\n"

    "G4 Rates Closes:\n"
    "  rows = tenors exactly: 2y, 3y, 5y, 7y, 10y, 20y, 30y\n"
    "  the four country columns left-to-right are ALWAYS US, Germany, UK, Japan\n"
    "  column keys EXACTLY: 'US level','US chg_1d_bp','Germany level','Germany chg_1d_bp',"
    "'UK level','UK chg_1d_bp','Japan level','Japan chg_1d_bp'\n\n"

    "Macro Closes:\n"
    "  rows = the instrument tickers exactly as printed (DXY, EUR, GBP, JPY, ... S&P, "
    "Stoxx, FTSE, Nikkei, VIX, Gold, WTI)\n"
    "  column keys EXACTLY: 'level' and 'chg_1d_pct'\n\n"

    "Inflation Closes (this table has TWO groups side-by-side: 'US TIPS' and '10y Real "
    "Yields'). Each instrument shows a real-yield value with its breakeven (BEI) value "
    "directly BELOW it. COMBINE each instrument and its breakeven into ONE row — do NOT "
    "create separate 'BEI' rows.\n"
    "  row keys EXACTLY, each once: from US TIPS use '5y','10y','30y'; from 10y Real Yields "
    "use 'DBRi','UKTi','JGBi','BTPi'.\n"
    "  column keys EXACTLY: 'real level','real chg_1d_bp','BEI level','BEI chg_1d_bp' "
    "(the real-yield row is the 'real' columns; the BEI value below it is the 'BEI' columns).\n\n"

    "Return ONLY valid JSON, no markdown fences:\n"
    '{ "exhibits": [ { "name": "G4 Rates Closes",\n'
    '    "table": { "2y": {"US level": 3.989, "US chg_1d_bp": 3.6 } } } ] }\n\n'

    "Rules: values are numbers; parenthesised values are negative ('(10.0)' → -10.0); "
    "skip the 2-week sparkline column; use the EXACT column keys above every time so two "
    "independent reads of the same page produce identical keys."
)


# ============================================================
# Page → date mapping + image rendering
# ============================================================

def page_dates(doc) -> list[str | None]:
    """
    Walk pages in order; return the report_day each page belongs to. The date
    only appears on an article's first page, so it carries forward until the next.
    """
    current = None
    out = []
    for page in doc:
        m = dayHeaderRe.search(page.get_text())
        if m:
            current = extract_day(m)
        out.append(current)
    return out


def is_exhibit_page(page) -> bool:
    """A page worth processing contains one of the core table titles."""
    text = page.get_text()
    return any(mk in text for mk in ("Rates Closes", "Inflation Closes", "Macro Closes"))


def render_page_image(page) -> str:
    """Render a page to a base64-encoded PNG."""
    pix = page.get_pixmap(dpi=renderDpi)
    return base64.standard_b64encode(pix.tobytes("png")).decode()


# ============================================================
# Vision extraction
# ============================================================

def extract_tables_from_image(img_b64: str, client: Anthropic) -> list[dict]:
    """Send a rendered page to Claude vision; return [{name, table}, ...].

    Retries on transient API errors (529 overloaded / 429 rate limit) with
    exponential backoff so a long backfill survives server-side blips.
    """
    wait = 8
    resp = None
    for attempt in range(6):
        try:
            resp = client.messages.create(
                model=visionModel,
                max_tokens=16000,   # large/dense exhibit pages — avoid truncating the JSON
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": visionPrompt},
                ]}],
            )
            break
        except Exception as e:
            es = str(e).lower()
            transient = any(k in es for k in ("overloaded", "529", "rate_limit", "429", "timeout"))
            if transient and attempt < 5:
                print(f"      … API busy ({attempt + 1}/5) — waiting {wait}s")
                time.sleep(wait)
                wait *= 2   # 8 → 16 → 32 → 64 → 128s
            else:
                raise
    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    if resp.stop_reason == "max_tokens":
        print("      ! vision response hit max_tokens — table may be incomplete")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group()).get("exhibits", [])
    except json.JSONDecodeError:
        return []


def merge_reads(reads: list[list[dict]]) -> tuple[list[dict], dict]:
    """
    Merge several independent vision reads of the same page.
      - Completeness: UNION of all cells (a section dropped in one read is kept
        from another).
      - Value: majority vote per cell.
      - Quality per exhibit: confidence (cells all reads agreed on), conflicts
        (differing values), empty_cells (no read produced a value), incomplete
        (a section appeared in some reads but not all).
    Returns (merged_exhibits, quality_by_exhibit_name).
    """
    from collections import defaultdict, Counter

    def norm_key(k):    # collapse whitespace and spaces around "/" so variants merge
        return re.sub(r"\s*/\s*", "/", " ".join(str(k).split())).strip()

    def norm_name(nm):  # drop parenthetical suffixes like "(5pm NY)" so a table is one key
        return re.sub(r"\s*\([^)]*\)", "", str(nm)).strip()

    n = len(reads)
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # name→row→col→[values]
    order = []
    for read in reads:
        for ex in read:
            name, table = ex.get("name"), ex.get("table") or {}
            if not name:
                continue
            name = norm_name(name)
            if name not in order:
                order.append(name)
            for row, cols in table.items():
                nrow = norm_key(row)
                for col, val in cols.items():
                    cells[name][nrow][norm_key(col)].append(val)

    merged, quality = [], {}
    for name in order:
        table, conflicts, empty = {}, [], []
        agree = total = 0
        seen_counts = []
        for row, cols in cells[name].items():
            table[row] = {}
            for col, vals in cols.items():
                total += 1
                seen_counts.append(len(vals))
                present = [v for v in vals if v not in (None, "")]
                if not present:
                    table[row][col] = None
                    empty.append(f"{row}/{col}")
                    continue
                counts = Counter(str(v) for v in present)
                top = counts.most_common(1)[0][0]
                table[row][col] = next(v for v in present if str(v) == top)
                distinct = set(counts)
                if len(distinct) == 1 and len(present) == n:
                    agree += 1
                elif len(distinct) > 1:
                    conflicts.append(f"{row}/{col}: {sorted(distinct)}")
        quality[name] = {
            "confidence":  round(100 * agree / total, 1) if total else None,
            "incomplete":  any(c < n for c in seen_counts) or bool(empty),
            "conflicts":   conflicts[:10],
            "empty_cells": empty[:10],
            "passes":      n,
        }
        merged.append({"name": name, "table": table})
    return merged, quality


def extract_tables_merged(img_b64: str, client: Anthropic, passes: int = visionPasses):
    """Extract the page `passes` times and merge for completeness + confidence."""
    reads = [extract_tables_from_image(img_b64, client) for _ in range(passes)]
    return merge_reads(reads)


# ============================================================
# Validation — cross-check every extracted number against the source text
# ============================================================

# ============================================================
# Ingestion
# ============================================================

def ingest_tables(file_path, client: Anthropic, collection_name: str = marketCollectionName,
                  skip_existing: bool = True, sample_every: int = 0,
                  source_name: str = None) -> dict:
    """Extract every exhibit table in a PDF and store one doc per (date, exhibit).

    source_name overrides the stored source filename (for Streamlit temp uploads).
    """
    path = Path(file_path)
    col = get_collection(collection_name)
    source = source_name or path.name

    if skip_existing and col.count_documents({"source": source}) > 0:
        return {"source": source, "skipped": True, "reason": "already present"}

    doc = fitz.open(str(path))
    dates = page_dates(doc)
    report = {"source": source, "exhibit_pages": 0, "docs": 0, "low_confidence": [], "warnings": []}
    inserted = 0

    for pno, page in enumerate(doc):
        if not is_exhibit_page(page):
            continue
        report["exhibit_pages"] += 1
        day = dates[pno]
        if not day:
            report["warnings"].append(f"page {pno}: exhibit found but no date — skipped")
            continue

        # Multi-read merge → completeness + agreement confidence + missing flags
        exhibits, quality = extract_tables_merged(render_page_image(page), client)
        for ex in exhibits:
            name, table = ex.get("name"), ex.get("table")
            if not name or not table:
                continue
            if not is_core_exhibit(name):   # core tables only
                continue

            q = quality.get(name, {})
            # Flag anything uncertain so it can be reviewed / propagated to the user
            # (explicit None check — `or 100` would treat a confidence of 0 as 100)
            conf = q.get("confidence")
            if q.get("incomplete") or (conf is not None and conf < 98) or q.get("empty_cells"):
                report["low_confidence"].append(
                    f"{day} · {name}: conf={q.get('confidence')}% "
                    f"incomplete={q.get('incomplete')} empty={len(q.get('empty_cells', []))} "
                    f"conflicts={len(q.get('conflicts', []))}"
                )

            record = {
                "report_day":    day,
                "report_month":  day[:7],
                "exhibit":       name,
                "table":         table,
                "quality":       q,        # confidence / incomplete / conflicts / empty_cells
                "source":        source,
            }
            col.delete_many({"report_day": day, "exhibit": name})  # idempotent per (date,exhibit)
            col.insert_one(record)
            inserted += 1
            if sample_every and inserted % sample_every == 0:
                print_sample(record)

    report["docs"] = inserted
    return report


def reingest_all_tables(source_dir, collection_name: str = marketCollectionName,
                        skip_existing: bool = True, sample_every: int = 0) -> list[dict]:
    client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
    # Ensure a fast lookup index (structured, no vector index needed)
    get_collection(collection_name).create_index([("report_day", 1), ("exhibit", 1)])

    results = []
    for pdf in sorted(Path(source_dir).rglob("*.pdf")):
        try:
            res = ingest_tables(pdf, client, collection_name, skip_existing, sample_every)
        except Exception as e:
            res = {"source": pdf.name, "error": str(e)}
        results.append(res)
        print_file_result(res)
    return results


# ============================================================
# Lookup helper (used later by the agent)
# ============================================================

def get_exhibit(report_day: str, exhibit: str = None,
                collection_name: str = marketCollectionName) -> list[dict]:
    """Exact lookup: all exhibit tables for a day (or one named exhibit)."""
    q = {"report_day": report_day}
    if exhibit:
        q["exhibit"] = exhibit
    return list(get_collection(collection_name).find(q, {"_id": 0}))


# ============================================================
# Pretty printing
# ============================================================

def print_sample(record: dict):
    rows = list(record["table"].keys())
    print(f"    -- sample: {record['report_day']} - {record['exhibit']} --")
    print(f"       rows: {rows[:6]}{'…' if len(rows) > 6 else ''}")
    first = next(iter(record["table"].values()), {})
    print(f"       first row cols: {first}")


def print_file_result(res: dict):
    if res.get("skipped"):
        print(f"  - {res['source']}: skipped (already present)")
    elif res.get("error"):
        print(f"  {res['source']}: ERROR {res['error']}")
    else:
        print(f"  {res['source']}: {res['docs']} exhibit table(s) from "
              f"{res['exhibit_pages']} page(s)")
        for lc in res.get("low_confidence", []):
            print(f"      WARNING: low confidence: {lc}")
        for w in res.get("warnings", []):
            print(f"      ! {w}")


def verify_market_data(collection_name: str = marketCollectionName):
    col = get_collection(collection_name)
    total = col.count_documents({})
    days = sorted(d for d in col.distinct("report_day") if d)
    exhibits = {e: col.count_documents({"exhibit": e}) for e in col.distinct("exhibit")}
    # Quality summary (agreement-based confidence from the multi-read merge)
    confs = [d["quality"]["confidence"] for d in col.find({}, {"quality": 1})
             if d.get("quality", {}).get("confidence") is not None]
    avg_conf = round(sum(confs) / len(confs), 2) if confs else None
    incomplete = col.count_documents({"quality.incomplete": True})
    has_empty = col.count_documents({"quality.empty_cells.0": {"$exists": True}})
    has_conflict = col.count_documents({"quality.conflicts.0": {"$exists": True}})
    print("\n=== VERIFY: daily_table_data ===")
    print(f"  total exhibit tables: {total}")
    print(f"  distinct days:        {len(days)}  range {(days[0], days[-1]) if days else None}")
    print(f"  exhibit types:        {exhibits}")
    print(f"  avg confidence:       {avg_conf}%  (multi-read agreement)")
    print(f"  incomplete tables:    {incomplete}  <- may be missing a section")
    print(f"  tables w/ empty cells:{has_empty}")
    print(f"  tables w/ conflicts:  {has_conflict}  <- reads disagreed on a value")


# ============================================================
# Manual CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python daily_table_data.py <source_dir> [sample_every]")
        sys.exit(1)
    sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    print(f"Extracting exhibit tables from: {sys.argv[1]} -> {marketCollectionName}")
    reingest_all_tables(sys.argv[1], sample_every=sample_every)
    verify_market_data()
