"""
test_dates.py — Verify Excel serial dates were correctly converted after reindex.

Tests:
    1. test_raw_chunks()         — inspect raw text in pnl_vectors for ISO dates vs serial numbers
    2. test_vector_search_feb()  — query for Feb 2026 PnL and check dates in returned chunks
    3. test_convert_function()   — unit test the convert_excel_dates() function itself

Run: python test_dates.py
"""

import os
import re
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Excel serial numbers present in the PnL files
# 46052 = 2026-01-01, 46081 = 2026-02-01, 46112 = 2026-03-04
KNOWN_SERIALS = [46052, 46081, 46112]
SERIAL_PATTERN = re.compile(r"\b(4[0-9]{4}|5[0-9]{4})\b")
ISO_DATE_PATTERN = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def get_col():
    client = MongoClient(os.getenv("MONGO_URI_ADMIN"), tlsCAFile=certifi.where())
    return client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_vectors"]


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# TEST 1: Inspect raw chunk text in pnl_vectors
# ============================================================

def test_raw_chunks():
    """
    Scan every chunk in pnl_vectors and report:
    - How many still contain Excel serial numbers (bad)
    - How many contain ISO date strings like 2026-01-01 (good)
    """
    section("TEST 1: Raw Chunk Date Audit")

    col = get_col()
    total = col.count_documents({})
    print(f"  Total chunks in pnl_vectors: {total}\n")

    serial_chunks = []
    iso_chunks = []
    neither = []

    for doc in col.find({}, {"_id": 1, "text": 1, "source": 1}):
        text = doc.get("text", "")
        source = doc.get("source", "unknown")

        has_serial = bool(SERIAL_PATTERN.search(text))
        has_iso = bool(ISO_DATE_PATTERN.search(text))

        if has_serial:
            serial_chunks.append((source, text[:120].replace("\n", " ")))
        if has_iso:
            iso_chunks.append((source, text[:120].replace("\n", " ")))
        if not has_serial and not has_iso:
            neither.append(source)

    print(f"  Chunks with ISO dates (2026-XX-XX):  {len(iso_chunks)}  ← GOOD")
    print(f"  Chunks with Excel serial numbers:    {len(serial_chunks)}  ← BAD (conversion didn't apply)")
    print(f"  Chunks with neither:                 {len(neither)}  (headers, totals, etc.)")

    if serial_chunks:
        print("\n  --- Sample chunks still containing serial numbers ---")
        for src, preview in serial_chunks[:3]:
            print(f"  source: {src}")
            print(f"  text:   {preview}")
            print()

    if iso_chunks:
        print("\n  --- Sample chunks with ISO dates (confirming conversion worked) ---")
        for src, preview in iso_chunks[:3]:
            print(f"  source: {src}")
            print(f"  text:   {preview}")
            print()

    if len(serial_chunks) == 0 and len(iso_chunks) > 0:
        print("  RESULT: PASS — all date fields converted to ISO format.")
    elif len(serial_chunks) > 0:
        print("  RESULT: FAIL — serial numbers still present. Reindex may not have run, or conversion missed these.")
    else:
        print("  RESULT: INCONCLUSIVE — no date patterns found in any chunk.")


# ============================================================
# TEST 2: Vector search query for a specific month
# ============================================================

def test_vector_search_feb():
    """
    Run a vector search for February 2026 PnL data and verify
    the returned chunks contain readable ISO dates, not serial numbers.
    """
    section("TEST 2: Vector Search — February 2026 PnL")

    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

    from langchain_mongodb import MongoDBAtlasVectorSearch
    client = MongoClient(os.getenv("MONGO_URI_USER"), tlsCAFile=certifi.where())
    col = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_vectors"]
    store = MongoDBAtlasVectorSearch(
        collection=col,
        embedding=embedding,
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )

    query = "February 2026 PnL positions performance"
    docs = store.similarity_search(query, k=5)

    print(f"  Query: '{query}'")
    print(f"  Returned {len(docs)} chunks\n")

    for i, doc in enumerate(docs):
        text = doc.page_content
        has_serial = bool(SERIAL_PATTERN.search(text))
        has_iso = bool(ISO_DATE_PATTERN.search(text))
        iso_dates = ISO_DATE_PATTERN.findall(text)
        source = doc.metadata.get("source", "unknown")

        status = "GOOD (ISO dates)" if has_iso and not has_serial else \
                 "BAD (serial numbers)" if has_serial else \
                 "NEUTRAL (no dates)"

        print(f"  [{i+1}] {status}")
        print(f"       source:     {source}")
        if iso_dates:
            print(f"       ISO dates found: {', '.join(sorted(set(iso_dates)))}")
        print(f"       preview:    {text[:100].replace(chr(10), ' ')}...")
        print()


# ============================================================
# TEST 3: Unit test the conversion function directly
# ============================================================

def test_convert_function():
    """
    Directly test convert_excel_dates() against known values from the PnL files.
    """
    section("TEST 3: Unit Test — convert_excel_dates()")

    from build import convert_excel_dates

    cases = [
        # (input_text, expected_substring, should_contain)
        # Standalone date serials — should be converted
        ("| 46052 | 46081 | rate | Long TY |", "2026-01-30", True),
        ("| 46052 | 46081 | rate | Long TY |", "2026-02-28", True),
        ("| 46081 | 46112 | crypto | IBIT |", "2026-02-28", True),
        ("| 46081 | 46112 | crypto | IBIT |", "2026-03-31", True),
        # Decimal values whose integer part is in serial range — must NOT be converted
        ("DV01 | 51850.69071 | 46979.23827 |", "51850.69071", True),
        ("DV01 | 51850.69071 | 46979.23827 |", "46979.23827", True),
        ("DV01 | 26120.51579 | 14132.1035 |", "26120.51579", True),
        # Large notionals — must NOT be converted
        ("Notional | 176471011.6 | 164", "176471011.6", True),
    ]

    passed = 0
    for text, expected, should_contain in cases:
        result = convert_excel_dates(text)
        ok = (expected in result) == should_contain
        status = "PASS" if ok else "FAIL"
        verb = "in" if should_contain else "NOT in"
        print(f"  [{status}]  input:    {text}")
        print(f"          expected: '{expected}' {verb} output")
        print(f"          output:   {result}")
        print()
        if ok:
            passed += 1

    print(f"  {passed}/{len(cases)} tests passed.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    test_convert_function()
    test_raw_chunks()
    test_vector_search_feb()
