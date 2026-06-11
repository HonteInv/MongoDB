# Validation / Revision Agent — Design Doc

1. Interprets a complaint + diagnoses the likely cause
2. Rewrites as targetted instruction 
3. Replays the pipeline over the affected agents

Ex.
```
ONE complaint box: "the Feb numbers look off"
        ↓
VALIDATION AGENT · PHASE A (plan)
  interpret → diagnose → rewrite → materiality → dirty set
        ↓
CONFIRM STEP
  "I'll re-run Performance, then refresh Risk + Newsletter to
   stay consistent. Market / Weekly untouched."   [Confirm] [Cancel]
        ↓ (on confirm)
snapshot current result → push to version stack
        ↓
VALIDATION AGENT · PHASE B (execute)
  replay ordered pipeline over dirty set → reassemble
        ↓
updated result + plain-English explanation of what changed
```

## Two-phase design (plan / execute)

Ensures user confirmation before anything runs

Two entry points:
- **`plan_revision(complaint, current_result)`** → returns a plan
- **`execute_revision(plan, current_result)`** → runs the plan, returns the updated result + explanation.

### The plan Object attributes
- `dirty_set` — which agents will be re-run, in dependency order
- `targeted_queries` — the rewritten, specific instruction for each agent
- `diagnosis` — data issue vs generation/wording issue 
- `materiality_reasoning` — why downstream agents are / aren't included
- `summary` — the concise human-readable plan shown at the confirm step
