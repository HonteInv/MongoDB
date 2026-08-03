# MongoDB

## Cluster Creation and Set Up
1. Pre-set up info
    - Access to the existing Atlas cluster (portfolio-cluster)
    - Database: portfolio_rag
    - Atlas → cluster → Collections → Create Collection

2. Search Index Config
    - Regular Index: Exact lookups and does not allow for vector similarity search. Created through the Indexes Tab.
    - Altas Vector Search: Sorts on fields and similarity search over embeddings. 

**NOTE**
    - numDimensions field: 3072 for OpenAI text-embedding-3-large, 768 if switch to gemini 
        - must match the embedding model or else will cause an index failure 
    - standard search is cosine for embeddings 

3. Current Fields Used
    - report_day, report_month, region, context_type, section, source
    - content_type -> separates narrative from numbers (deliberate desig)
        - Daily reports mix commentary with embedded tables (exhibits), so they are now tagged separately so we don't embed numerical data as prose. (similar set up to pnl) 

## Set Up 
1. set up virtual environment (Python version 3.11 and up)
2. install requirements 
    * `brew install cmake` if failure building wheel 
3. 

## PNL 
This is stored in a separate collection with different logic for accuracy.

## Data Audit (read-only)
`audit_data.py` scans every collection for residue of the ingestion bugs fixed
on 2026-08-02 (date-corrupted position sizes, literal "None" cells, rows filed
under the wrong period, missing/mismatched pnl_summary docs, zero-confidence
vision tables, wrong-schema context docs, invalid user roles) and reconciles
`pnl_table` row counts + AUM against the source files in `data/pnl/`.

```
python audit_data.py                 # console report, exit 1 if errors found
python audit_data.py --json out.json # also write machine-readable findings
```

It NEVER writes to MongoDB — every finding names the admin-app action that
fixes it (usually a Reindex-mode re-upload). Needs `MONGO_URI_USER` (plus
`MONGO_URI_ADMIN` for the users check) in `.env`; run it wherever the app runs.

## Wayfound Agent Supervision (optional)
`query_app.py` can send every run to Wayfound (app.wayfound.ai) so the supervisor
can review agent behavior: the user query, the planner's routing decision, each
agent's output (with self-critique rounds), and any revision / validation_agent
loop. One Wayfound session per query; revisions append to the same session.

**Enable** — add to `.env` (off silently when unset):
```
WAYFOUND_API_KEY=...        # ask a Wayfound admin to create one
WAYFOUND_AGENT_ID=...       # optional; defaults to the registered "MongoDB" agent
```

**Notes**
- All uploads are fire-and-forget on background threads — a Wayfound outage can
  never block or break the app. Failures disable recording for that run only.
- Engine lives in `wayfound_monitor.py`; `multiagent.py` is untouched (agent
  result dicts already carry per-agent timestamps, which preserve the run timeline).
- Sessions are tagged with visitor = login username and metadata = plan intent /
  period / agents, so supervisor recordings are filterable per user and intent. 