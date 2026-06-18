"""
daily_data.py — date-aware ingestion of Morgan Stanley daily macro PDFs.

Goal: increase retrieval accuracy by tagging every chunk with the exact DAY it
belongs to (plus section / region / content type), so the Market Context agent
can pre-filter to the right date window instead of semantic-searching all of time.

Two-level split:
  Level 1  split_into_daily_articles()  — cut a (possibly merged) PDF into
           per-day articles using the timestamp header ("May 1, 2026 10:02 PM GMT")
  Level 2  split_into_sections()        — cut each day into sections
           (Developed Markets / Emerging Markets / Central Bank Monitor / Exhibits),
           deriving region (DM/EM) and content_type (commentary / central_bank / data_table)

Writes only to the NEW collection (context_daily) — the live context_vectors is
never touched, so the running app is unaffected.

Reuses build.py for the embedding object, Mongo handles, PDF loading, and the
filename month parser. Nothing here is imported by the live app yet.

Manual run:
    python daily_data.py <source_dir>     # ingest every PDF in a folder, then verify
"""

import re
import sys
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF — sorted extraction preserves multi-column reading order
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse proven pieces from the existing pipeline
from build import get_collection, embedding, _extract_context_period


def load_pdf_text(path: str | Path) -> str:
    """
    Extract PDF text in proper reading order. PyMuPDF's sort=True orders text blocks
    by visual position, which keeps multi-column layouts (Developed Markets before
    Emerging Markets, etc.) in the right sequence — pypdf scrambles them.
    """
    doc = fitz.open(str(path))
    return "\n".join(page.get_text(sort=True) for page in doc)


# ============================================================
# Constants
# ============================================================

# Target collection + index (see docs/daily_data_setup.md)
collectionName = "context_daily"
indexName = "daily_index"

# Level-1 boundary: a daily article always opens with a timestamp line like
# "May 1, 2026 10:02 PM GMT". This carries the full date incl. year, so we never
# depend on the filename and the 2025-vs-2026 ambiguity disappears.
dayHeaderRe = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+(\d{1,2}),\s+(20\d{2})\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+GMT',
    re.IGNORECASE,
)

# Level-2 boundaries: section headers that appear as plain text lines.
sectionHeaders = ["Central Bank Monitor", "Developed Markets", "Emerging Markets", "Exhibit"]

# Section → region (only the two market sections carry a region)
sectionRegionMap = {"Developed Markets": "DM", "Emerging Markets": "EM"}

# Section → content type (default is "commentary")
sectionContentMap = {"Exhibit": "data_table", "Central Bank Monitor": "central_bank"}

# Asset-class topics. Stored as a LIST per chunk (a chunk can touch several).
#   - Monthly reports: anchored by the filename (reliable)
#   - MS daily chunks: detected from content keywords (best-effort)
# Keywords are matched case-insensitively as whole words where short/ambiguous.
topicKeywords = {
    "rates":       [r"\brates?\b", r"\byield", r"\bdv01\b", r"\bswaps?\b", r"\bbps\b",
                    r"\btreasur", r"\bgilt", r"\bjgb\b", r"\bbund", r"\bswaption", r"\bsofr\b",
                    r"\bfront[- ]end\b", r"\bcurve\b", r"\b\d+y\b"],
    "fx":          [r"\bfx\b", r"\bcurrenc", r"\busd\b", r"\beur\b", r"\bjpy\b", r"\bgbp\b",
                    r"\bchf\b", r"\bcnh\b", r"\bdollar\b", r"\busd/?\w{3}\b", r"\bdxy\b"],
    "equities":    [r"\bequit", r"\bstocks?\b", r"\bs&p\b", r"\bspx\b", r"\bnasdaq\b",
                    r"\bnikkei\b", r"\bstoxx\b", r"\bshares?\b", r"\bindex\b"],
    "commodities": [r"\bcommodit", r"\boil\b", r"\bbrent\b", r"\bwti\b", r"\bcopper\b",
                    r"\bgold\b", r"\bplatinum\b", r"\buranium\b", r"\bcrude\b", r"\bmetal"],
    "crypto":      [r"\bcrypto\b", r"\bbitcoin\b", r"\bbtc\b", r"\bibit\b", r"\bdigital asset"],
    "central_banks": [r"\bfed\b", r"\bfomc\b", r"\becb\b", r"\bboj\b", r"\bboe\b", r"\bbo[ck]\b",
                      r"\brbi\b", r"\bpboc\b", r"\bsnb\b", r"\bcentral bank", r"\bmonetary policy\b",
                      r"\brate (?:cut|hike)", r"\bpolicy rate\b", r"\bfed watch\b", r"\brate decision\b"],
}
topicPatterns = {t: [re.compile(p, re.IGNORECASE) for p in pats] for t, pats in topicKeywords.items()}

# The trailing legal/disclosure block is one contiguous section at the END of each
# article — often >70% of the extracted text. We TRUNCATE at the earliest of these
# markers. NOTE: we deliberately do NOT use "Disclosure Section" — that phrase also
# appears early as a reference ("refer to the Disclosure Section…"), which would cut
# real content. These markers only appear in the actual disclosure paragraphs.
disclosureMarkers = [
    "The information and opinions in Morgan Stanley Research",
    "Morgan Stanley Research does not provide individually tailored",
    "Morgan Stanley is not acting as a municipal advisor",
]

# Shorter per-line contact/footer noise removed within the kept content.
boilerplateMarkers = [
    "morgan stanley does and seeks to do business",
    "for analyst certification",
    "investors should consider morgan stanley",
    "analysts employed by non-u.s.",
    "not for redistribution",
    "downloaded by",
]

chunkSize = 2000
chunkOverlap = 300


# ============================================================
# Level 1 — split a PDF into per-day articles
# ============================================================

def extract_day(match: re.Match) -> str:
    """Turn a dayHeaderRe match into an ISO day string, e.g. '2026-05-01'."""
    month, day, year = match.group(1).capitalize(), match.group(2), match.group(3)
    return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").strftime("%Y-%m-%d")


def topic_from_filename(source: str) -> str | None:
    """Anchor a monthly report to its asset class from the filename (reliable)."""
    low = source.lower().replace("_", " ")   # underscores as separators
    for topic, patterns in topicPatterns.items():
        if any(p.search(low) for p in patterns):
            return topic
    return None


def detect_topics(text: str, anchor: str | None = None) -> list[str]:
    """
    Return the asset-class topics a chunk touches (multi-value). `anchor` is the
    filename-derived topic for monthly reports — always included so those chunks
    keep their primary topic even if the text is sparse. Best-effort for prose.
    """
    found = {t for t, patterns in topicPatterns.items() if any(p.search(text) for p in patterns)}
    if anchor:
        found.add(anchor)
    return sorted(found)


def split_into_daily_articles(text: str) -> list[tuple[str, str]]:
    """
    Cut full document text into [(report_day, article_text), ...] using the
    timestamp headers as boundaries. Returns [] if no daily headers are found
    (the caller then falls back to month-only tagging).
    """
    matches = list(dayHeaderRe.finditer(text))
    if not matches:
        return []
    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        articles.append((extract_day(m), text[start:end]))
    return articles


# ============================================================
# Boilerplate stripping
# ============================================================

def clean_boilerplate(text: str) -> str:
    """
    Remove boilerplate. Two passes:
      1. Truncate the trailing legal/disclosure block (the bulk of the noise — it's
         a contiguous section at the end, so cutting at its start is safe).
      2. Remove remaining contact / footer lines within the kept content.
    Heuristic; refine as needed.
    """
    # Pass 1 — truncate at the earliest disclosure marker
    cut = len(text)
    for mk in disclosureMarkers:
        idx = text.find(mk)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut]

    # Pass 2 — line-level cleanup
    kept = []
    for line in text.splitlines():
        low = line.strip().lower()
        if not low:
            kept.append(line)
            continue
        if "@morganstanley.com" in low:
            continue
        if re.match(r'^\+?\d[\d\s\-()]{6,}$', line.strip()):   # phone-only line
            continue
        if low in ("strategist", "economist", "analyst", "update", "research"):
            continue
        if any(mk in low for mk in boilerplateMarkers):
            continue
        kept.append(line)
    return "\n".join(kept)


# ============================================================
# Level 2 — split a day into sections (region + content_type)
# ============================================================

def split_into_sections(article_text: str) -> list[dict]:
    """
    Cut one day's article into sections. Returns a list of
    {section, region, content_type, text}. Developed/Emerging Markets headers that
    appear AFTER 'Central Bank Monitor' are treated as part of the central-bank
    block, not fresh market commentary.
    """
    markers = []
    for name in sectionHeaders:
        for m in re.finditer(re.escape(name), article_text):
            markers.append((m.start(), name))
    markers.sort()

    if not markers:
        return [{"section": "Summary", "region": None,
                 "content_type": "commentary", "text": article_text}]

    sections = []
    intro = article_text[:markers[0][0]].strip()
    if intro:
        sections.append({"section": "Summary", "region": None,
                         "content_type": "commentary", "text": intro})

    seen_cbm = False
    for i, (pos, name) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(article_text)
        seg = article_text[pos:end].strip()
        if not seg:
            continue

        if name == "Central Bank Monitor":
            seen_cbm = True
            section_label, region, ctype = "Central Bank Monitor", None, "central_bank"
        elif name == "Exhibit":
            section_label, region, ctype = "Exhibits", None, "data_table"
        elif seen_cbm:
            # DM/EM under Central Bank Monitor → still central-bank content
            section_label, region, ctype = "Central Bank Monitor", sectionRegionMap.get(name), "central_bank"
        else:
            section_label, region, ctype = name, sectionRegionMap.get(name), "commentary"

        sections.append({"section": section_label, "region": region,
                         "content_type": ctype, "text": seg})
    return sections


# ============================================================
# Chunking + chunk assembly
# ============================================================

def chunk_section(text: str) -> list[str]:
    """Chunk one section's text. Sections are already coherent, so chunks stay on-topic."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
    return [c for c in splitter.split_text(text) if c.strip()]


def make_chunk(text, source, report_day, report_month, section, region, content_type, topic) -> dict:
    """Build one stored document: content + all top-level filter metadata."""
    tag = "[" + source
    if report_day:
        tag += f" · {report_day}"
    if section:
        tag += f" · {section}"
    tag += "]"
    return {
        "text":         f"{tag}\n{text}",
        "report_day":   report_day,
        "report_month": report_month,
        "section":      section,
        "region":       region,
        "content_type": content_type,
        "topic":        topic,          # list of asset classes (may be empty)
        "source":       source,
    }


def build_chunks_for_file(file_path: str | Path) -> tuple[list[dict], dict]:
    """
    Load a PDF and produce (chunk_dicts, report). The report records detected days
    and any warnings (used for the verify step and the manual-fix UI later).
    """
    path = Path(file_path)
    full = load_pdf_text(path)
    source = path.name

    report = {"source": source, "detected_daily": True, "days": [], "warnings": []}
    chunks: list[dict] = []

    articles = split_into_daily_articles(full)

    # ── Fallback: no daily headers (e.g. monthly commodity/crypto reports) ──
    if not articles:
        report["detected_daily"] = False
        month = _extract_context_period(source)  # month from filename, if any
        anchor = topic_from_filename(source)      # asset class from filename
        report["warnings"].append(
            f"No daily headers detected — ingested WITHOUT filters: "
            f"report_day, section, region (report_month={month}, topic={anchor})"
        )
        for piece in chunk_section(clean_boilerplate(full)):
            topics = detect_topics(piece, anchor=anchor)
            chunks.append(make_chunk(piece, source, None, month, None, None, "commentary", topics))
        return chunks, report

    # ── Normal path: per-day, per-section ──
    for day, article in articles:
        report["days"].append(day)
        article = clean_boilerplate(article)
        month = day[:7]
        for sec in split_into_sections(article):
            for piece in chunk_section(sec["text"]):
                topics = detect_topics(piece)   # content-based for MS dailies
                chunks.append(make_chunk(
                    piece, source, day, month,
                    sec["section"], sec["region"], sec["content_type"], topics,
                ))
    return chunks, report


# ============================================================
# Ingestion
# ============================================================

# Running counter so "every Nth chunk" spans the whole run, not per-file.
sampleCounter = 0


def print_sample(chunk: dict):
    """Print one chunk's filter metadata + a text snippet — for spot-checking tags."""
    snippet = chunk["text"].split("\n", 1)[-1][:90].replace("\n", " ")
    print(f"    ── sample chunk #{sampleCounter} ──")
    print(f"       report_day={chunk['report_day']}  report_month={chunk['report_month']}")
    print(f"       section={chunk['section']}  region={chunk['region']}  content_type={chunk['content_type']}")
    print(f"       topic={chunk['topic']}")
    print(f"       source={chunk['source']}")
    print(f"       text: {snippet}…")


def ingest_file(file_path: str | Path, collection_name: str = collectionName,
                skip_existing: bool = True, sample_every: int = 0) -> dict:
    """
    Ingest one PDF into the daily collection. Skips the file if its source is
    already present (idempotent re-runs). Returns the build report + insert count.

    sample_every: if > 0, print full metadata for every Nth chunk across the run
                  (useful for eyeballing that tagging is correct).
    """
    global sampleCounter
    path = Path(file_path)
    col = get_collection(collection_name)
    source = path.name

    if skip_existing and col.count_documents({"source": source}) > 0:
        return {"source": source, "skipped": True, "reason": "already present"}

    chunks, report = build_chunks_for_file(path)
    if not chunks:
        report["inserted"] = 0
        report["warnings"].append("No chunks produced.")
        return report

    # Spot-check sampling before embedding/insert
    if sample_every:
        for c in chunks:
            if sampleCounter % sample_every == 0:
                print_sample(c)
            sampleCounter += 1

    vectors = embedding.embed_documents([c["text"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["embedding"] = v
    col.insert_many(chunks)

    report["inserted"] = len(chunks)
    return report


def reingest_all(source_dir: str | Path, collection_name: str = collectionName,
                 skip_existing: bool = True, sample_every: int = 0) -> list[dict]:
    """Ingest every PDF under a folder. Returns a per-file report list."""
    global sampleCounter
    sampleCounter = 0  # reset so sampling is consistent each run
    folder = Path(source_dir)
    pdfs = sorted(folder.rglob("*.pdf"))
    results = []
    for pdf in pdfs:
        try:
            res = ingest_file(pdf, collection_name, skip_existing, sample_every=sample_every)
        except Exception as e:
            res = {"source": pdf.name, "error": str(e)}
        results.append(res)
        print_file_result(res)
    return results


# ============================================================
# Verify — coverage report (also surfaced to the user)
# ============================================================

def verify(collection_name: str = collectionName) -> dict:
    """Summarize what landed in the collection so tagging can be eyeballed."""
    col = get_collection(collection_name)
    total = col.count_documents({})
    with_day = col.count_documents({"report_day": {"$ne": None}})
    days = sorted(d for d in col.distinct("report_day") if d)
    months = sorted(m for m in col.distinct("report_month") if m)
    ctypes = {ct: col.count_documents({"content_type": ct}) for ct in col.distinct("content_type")}
    regions = {r: col.count_documents({"region": r}) for r in col.distinct("region")}
    topics = {t: col.count_documents({"topic": t}) for t in col.distinct("topic") if t}
    no_day_sources = [s for s in col.distinct("source", {"report_day": None})]

    return {
        "collection":        collection_name,
        "total_chunks":      total,
        "chunks_with_day":   with_day,
        "chunks_without_day": total - with_day,
        "distinct_days":     len(days),
        "day_range":         (days[0], days[-1]) if days else (None, None),
        "months":            months,
        "content_type_counts": ctypes,
        "region_counts":     regions,
        "topic_counts":      topics,
        "sources_without_day": no_day_sources,
    }


# ============================================================
# Pretty printing (for manual runs / surfacing to the user)
# ============================================================

def print_file_result(res: dict):
    if res.get("skipped"):
        print(f"  · {res['source']}: skipped (already present)")
    elif res.get("error"):
        print(f"  ✗ {res['source']}: ERROR {res['error']}")
    else:
        flag = "" if res.get("detected_daily") else "  ⚠ no daily headers"
        print(f"  ✓ {res['source']}: {res.get('inserted', 0)} chunks, "
              f"{len(res.get('days', []))} day(s){flag}")
        for w in res.get("warnings", []):
            print(f"      ! {w}")


def print_verify(collection_name: str = collectionName):
    v = verify(collection_name)
    print("\n=== VERIFY:", v["collection"], "===")
    print(f"  total chunks:        {v['total_chunks']:,}")
    print(f"  with report_day:     {v['chunks_with_day']:,}")
    print(f"  without report_day:  {v['chunks_without_day']:,}")
    print(f"  distinct days:       {v['distinct_days']}  range {v['day_range']}")
    print(f"  months:              {v['months']}")
    print(f"  content types:       {v['content_type_counts']}")
    print(f"  regions:             {v['region_counts']}")
    print(f"  topics:              {v['topic_counts']}")
    if v["sources_without_day"]:
        print(f"  ⚠ sources missing daily tags: {v['sources_without_day']}")


# ============================================================
# Manual CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python daily_data.py <source_dir> [sample_every]")
        print("  sample_every: print full metadata for every Nth chunk (default 20, 0 = off)")
        sys.exit(1)
    sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(f"Ingesting PDFs from: {sys.argv[1]} → {collectionName}  (sampling every {sample_every} chunks)")
    reingest_all(sys.argv[1], sample_every=sample_every)
    print_verify()
