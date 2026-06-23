"""
test.py — Diagnostic tests for PnL retrieval issues.

Tests:
    1. audit_pnl_collection() — raw view of what's actually stored in pnl_vectors
    2. test_deletion_works() — verify delete_by_source removes all matching docs
    3. test_vector_search_sources() — see which source files vector search actually returns
    4. test_index_vs_collection() — detect stale Atlas index (deleted docs still returned)
    5. test_embedding_consistency() — confirm no mixed embedding dimensions in pnl_vectors

Run individual tests or run_all_diagnostics() to run everything.
"""

import os
import certifi
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

from dotenv import load_dotenv
import os
load_dotenv()
from anthropic import Anthropic

load_dotenv()

# ============================================================
# Helpers
# ============================================================

def get_admin_col(collection_name: str):
    client = MongoClient(os.getenv("MONGO_URI_ADMIN"), tlsCAFile=certifi.where())
    return client[os.getenv("MONGO_DB_NAME", "portfolio_rag")][collection_name]


def get_user_col(collection_name: str):
    """Use the read-only user URI — same as what the agents use."""
    client = MongoClient(os.getenv("MONGO_URI_USER"), tlsCAFile=certifi.where())
    return client[os.getenv("MONGO_DB_NAME", "portfolio_rag")][collection_name]


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# TEST 1: Raw audit of pnl_vectors
# What's actually stored? Are old files still in there?
# ============================================================

def audit_pnl_collection():
    """
    Print every unique source file in pnl_vectors and how many chunks
    it has. Reveals if old PnL files are still present after deletion.
    """
    section("TEST 1: Raw PnL Collection Audit")

    col = get_admin_col("pnl_vectors")
    total = col.count_documents({})
    print(f"  Total documents in pnl_vectors: {total}")

    if total == 0:
        print("  Collection is empty.")
        return

    # Check all possible source fields LangChain might use
    source_fields = ["source", "metadata.source", "metadata.original_filename"]
    print("\n  --- Source field breakdown ---")
    for field in source_fields:
        values = col.distinct(field)
        non_empty = [v for v in values if v]
        print(f"\n  Field '{field}' ({len(non_empty)} unique values):")
        for v in non_empty:
            count = col.count_documents({field: v})
            print(f"    [{count:>4} chunks]  {Path(str(v)).name}  ({v})")

    # Check for docs with NO source field at all
    no_source = col.count_documents({
        "source": {"$exists": False},
        "metadata": {"$exists": False},
    })
    print(f"\n  Docs with no source field at all: {no_source}")

    # Show a raw sample doc structure (keys only, not embedding)
    sample = col.find_one({})
    if sample:
        print("\n  --- Sample document structure (field names) ---")
        for key, val in sample.items():
            if key == "embedding":
                print(f"    embedding: [list of {len(val)} floats]")
            elif key == "_id":
                print(f"    _id: {val}")
            else:
                preview = str(val)[:80].replace("\n", " ")
                print(f"    {key}: {preview}")


# ============================================================
# TEST 2: Verify deletion actually works
# Does delete_by_source remove all chunks for a filename?
# ============================================================

def test_deletion_works(filename: str):
    """
    Check how many chunks match a filename using every possible
    field/pattern. Shows exactly what delete_by_source would (or
    would NOT) find, without deleting anything.

    Args:
        filename: just the filename, e.g. "march_2026_pnl.csv"
    """
    section(f"TEST 2: Deletion Match Audit for '{filename}'")

    col = get_admin_col("pnl_vectors")

    queries = {
        "exact match on 'source'":                  {"source": filename},
        "exact match on 'metadata.source'":          {"metadata.source": filename},
        "regex on 'source' (case-insensitive)":      {"source": {"$regex": filename, "$options": "i"}},
        "regex on 'metadata.source'":                {"metadata.source": {"$regex": filename, "$options": "i"}},
        "regex on 'metadata.original_filename'":     {"metadata.original_filename": {"$regex": filename, "$options": "i"}},
        "combined $or (what delete_by_source uses)": {
            "$or": [
                {"source": filename},
                {"metadata.source": filename},
                {"source": {"$regex": filename, "$options": "i"}},
                {"metadata.source": {"$regex": filename, "$options": "i"}},
                {"metadata.original_filename": filename},
            ]
        },
    }

    any_found = False
    for label, query in queries.items():
        count = col.count_documents(query)
        status = "FOUND" if count > 0 else "none"
        print(f"  {label}:")
        print(f"    → {count} documents ({status})")
        if count > 0:
            any_found = True
            # Show what source field the matched docs actually have
            sample = col.find_one(query, {"source": 1, "metadata": 1, "_id": 0})
            print(f"    Sample source:    {sample.get('source', '(not set)')}")
            print(f"    Sample metadata:  {sample.get('metadata', '(not set)')}")

    if not any_found:
        print(f"\n  WARNING: '{filename}' was not found under any field.")
        print("  Either it was never indexed, or it's stored under a different name.")
        print("  Run audit_pnl_collection() to see what IS stored.")


# ============================================================
# TEST 3: Vector search source audit
# What files are the retrieved chunks actually coming from?
# ============================================================

def test_vector_search_sources(query: str = "PnL portfolio performance March 2026", k: int = 20):
    """
    Run the same vector search the PortfolioPerformanceAgent uses and
    show exactly which source files the returned chunks came from.

    This reveals if old-period PnL chunks are being returned even
    after you thought they were deleted.

    Args:
        query: the search query (defaults to a typical PnL agent query)
        k:     number of results to retrieve
    """
    section(f"TEST 3: Vector Search Source Audit (k={k})")
    print(f"  Query: '{query}'")

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
    col = get_user_col("pnl_vectors")
    store = MongoDBAtlasVectorSearch(
        collection=col,
        embedding=embedding,
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )

    docs = store.similarity_search(query, k=k)

    print(f"\n  Retrieved {len(docs)} chunks:")
    source_counts: dict[str, int] = {}

    for i, doc in enumerate(docs):
        src = (
            doc.metadata.get("source")
            or doc.metadata.get("original_filename")
            or "UNKNOWN"
        )
        src_name = Path(str(src)).name
        source_counts[src_name] = source_counts.get(src_name, 0) + 1
        text_preview = doc.page_content[:80].replace("\n", " ")
        print(f"\n  [{i+1:02d}] source: {src_name}")
        print(f"       text:   {text_preview}...")

    print(f"\n  --- Source file summary ---")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3} chunks  ←  {src}")

    if len(source_counts) > 1:
        print("\n  WARNING: Chunks from multiple source files returned.")
        print("  If old-period files appear here, deletion did NOT fully work,")
        print("  OR the Atlas Search index is stale (see TEST 4).")
    else:
        print("\n  OK: All retrieved chunks come from a single source file.")


# ============================================================
# TEST 4: Atlas Search index staleness check
# Are deleted docs still showing up in vector search?
# ============================================================

def test_index_vs_collection(k: int = 20):
    """
    Cross-reference vector search results against what's actually in
    the collection. If a returned chunk's _id doesn't exist in the
    collection, the Atlas Search index is stale — it's returning
    results for documents that were already deleted.

    This is the 'funky persistent memory' your supervisor observed.
    """
    section("TEST 4: Atlas Search Index Staleness Check")

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
    from bson import ObjectId

    # Use admin URI here so we can do direct _id lookups
    client = MongoClient(os.getenv("MONGO_URI_ADMIN"), tlsCAFile=certifi.where())
    col = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_vectors"]

    store = MongoDBAtlasVectorSearch(
        collection=col,
        embedding=embedding,
        index_name="vector_index",
        text_key="text",
        embedding_key="embedding",
    )

    # Use a broad query to get a representative sample
    docs = store.similarity_search("portfolio PnL performance attribution", k=k)

    print(f"  Vector search returned {len(docs)} results.")
    print("  Checking each result's _id against the live collection...\n")

    stale_count = 0
    live_count = 0

    for doc in docs:
        # LangChain stores the MongoDB _id in metadata
        raw_id = doc.metadata.get("_id")
        source = Path(str(doc.metadata.get("source", "unknown"))).name

        if raw_id is None:
            print(f"  [?] No _id in metadata — cannot verify. source={source}")
            continue

        try:
            oid = ObjectId(str(raw_id))
            exists = col.count_documents({"_id": oid}) > 0
        except Exception:
            exists = col.count_documents({"_id": raw_id}) > 0

        if exists:
            live_count += 1
        else:
            stale_count += 1
            print(f"  [STALE] _id={raw_id} is in the index but NOT in the collection.")
            print(f"          source={source}")
            print(f"          text preview: {doc.page_content[:60].replace(chr(10),' ')}...")

    print(f"\n  Live documents (in collection):  {live_count}")
    print(f"  Stale documents (index only):    {stale_count}")

    if stale_count > 0:
        print("\n  DIAGNOSIS: Atlas Search index is stale.")
        print("  Deleted documents are still being returned by vector search.")
        print("  Fix: wait for Atlas to sync (~1-5 min), or re-create the search index.")
    else:
        print("\n  OK: No stale index entries detected.")
        print("  The Atlas index appears to be in sync with the collection.")


# ============================================================
# TEST 5: Embedding dimension consistency in pnl_vectors
# Mixed dimensions = corrupted vector search results
# ============================================================

def test_embedding_consistency():
    """
    Check all documents in pnl_vectors for consistent embedding
    dimensions. Mixed dimensions (e.g. 768 + 3072) from switching
    between Gemini and OpenAI will cause incorrect similarity scores.
    """
    section("TEST 5: Embedding Dimension Consistency")

    col = get_admin_col("pnl_vectors")
    total = col.count_documents({"embedding": {"$exists": True}})
    print(f"  Scanning {total} documents with embeddings...")

    dim_counts: dict[int, int] = {}
    dim_sources: dict[int, list[str]] = {}

    cursor = col.find(
        {"embedding": {"$exists": True}},
        {"embedding": 1, "source": 1, "metadata": 1},
        batch_size=200,
    )

    for doc in cursor:
        dim = len(doc.get("embedding", []))
        src = (
            doc.get("source")
            or (doc.get("metadata") or {}).get("source")
            or "unknown"
        )
        src_name = Path(str(src)).name
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
        if dim not in dim_sources:
            dim_sources[dim] = []
        if src_name not in dim_sources[dim]:
            dim_sources[dim].append(src_name)

    print("\n  --- Embedding dimensions found ---")
    for dim, count in sorted(dim_counts.items()):
        print(f"  {dim}d:  {count} documents")
        for src in dim_sources[dim][:5]:
            print(f"         from: {src}")

    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    expected = 3072 if provider == "openai" else 768

    if len(dim_counts) > 1:
        print(f"\n  WARNING: Mixed embedding dimensions detected!")
        print(f"  Current EMBEDDING_PROVIDER='{provider}' expects {expected}d.")
        print("  Documents with wrong dimensions will return garbage similarity scores.")
        print("  Fix: delete_collection('pnl_vectors') and re-ingest everything.")
    elif len(dim_counts) == 1:
        actual = list(dim_counts.keys())[0]
        if actual == expected:
            print(f"\n  OK: All embeddings are {actual}d — matches {provider} ({expected}d).")
        else:
            print(f"\n  WARNING: All embeddings are {actual}d but {provider} expects {expected}d.")
            print("  Fix: delete_collection('pnl_vectors') and re-ingest with correct provider.")
    else:
        print("\n  No embeddings found in collection.")


# ============================================================
# Run all diagnostics
# ============================================================

def run_all_diagnostics(pnl_filename: str = None):
    """
    Run all 5 diagnostic tests in sequence.

    Args:
        pnl_filename: optional specific filename to test deletion matching,
                      e.g. "march_2026_pnl.csv". If None, skips TEST 2.
    """
    print("\n" + "="*60)
    print("  PnL RETRIEVAL DIAGNOSTIC SUITE")
    print("="*60)

    audit_pnl_collection()

    if pnl_filename:
        test_deletion_works(pnl_filename)
    else:
        print("\n  [TEST 2 SKIPPED] Pass pnl_filename= to run deletion audit.")

    test_embedding_consistency()
    test_vector_search_sources()
    test_index_vs_collection()

    print("\n" + "="*60)
    print("  DIAGNOSTICS COMPLETE")
    print("="*60)

# ============================================================
# Check for models
# ============================================================

def check_models():
    client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
    models = client.models.list()
    for m in models.data:
        print(m.id)

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Option A: run everything (replace filename with your actual PnL file)
    # run_all_diagnostics(pnl_filename=None)

    # Option B: run individual tests
    # audit_pnl_collection()
    # test_deletion_works("march_2026_pnl.csv")
    # test_embedding_consistency()
    # test_vector_search_sources("PnL portfolio performance March 2026")
    # test_index_vs_collection()

    check_models()



