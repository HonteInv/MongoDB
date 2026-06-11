# Validation / Revision Agent — Design Doc

**Status:** Scoped, not yet built
**Author:** (internship project)
**Purpose:** Replace the current per-section revise flow with a single, intelligent
revision layer that interprets a free-text complaint, decides what actually needs
to change, confirms with the user, and re-runs only the responsible agents.

---

## 1. Problem

The current `revise_section()` flow has two structural weaknesses:

1. **The user must pick the section themselves.** The UI shows a "Revise this
   section" box under each expander. If the user types *"the numbers are off"*
   under the **Newsletter** box, it re-runs the *newsletter writer* — but the
   writer doesn't own the numbers. They come from `performance` → `pnl_summary`.
   The newsletter re-renders with the **same underlying numbers**, so nothing
   meaningfully changes. This is the most likely cause of the reported
   "I sent a request but nothing came out."

2. **Re-prompting can't fix a data problem.** If a number is genuinely wrong, it
   is wrong in `pnl_table` / `pnl_summary`, not in the prompt. Asking the model to
   "recheck" reproduces the same output, because it is faithfully reading the same
   data. No error, no visible change.

A validation/revision layer that **interprets** the complaint and **routes** it to
the correct underlying agent fixes weakness 1 directly, and sets up the structure
to address weakness 2 later (see Tiers).

---

## 2. Concept

A second planner — parallel to the existing `_plan_execution()` — but for *fixes*
instead of fresh queries. It interprets a complaint, diagnoses the likely cause,
rewrites it as a targeted instruction, judges how far the change propagates, and
replays the existing pipeline over only the affected agents.

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

---

## 3. Two-phase design (plan / execute)

Because the user confirms before anything runs, the agent **cannot be a single
call**. It has two entry points:

- **`plan_revision(complaint, current_result)`** → returns a *plan*, runs no agents.
- **`execute_revision(plan, current_result)`** → runs the plan, returns the updated
  result + explanation.

This seam also benefits Tier 3 (below): the database diagnosis naturally lives in
Phase A, so the user confirms with real information in front of them
(*"the stored Feb return is +4.36%, which matches the database — this looks like a
wording issue, not a data error. Proceed?"*).

### The plan object carries
- `dirty_set` — which agents will be re-run, in dependency order
- `targeted_queries` — the rewritten, specific instruction for each agent
- `diagnosis` — data issue vs generation/wording issue (LLM guess in Tier 2,
  DB-backed in Tier 3) — **kept as its own field from day one**
- `materiality_reasoning` — why downstream agents are / aren't included
- `summary` — the concise human-readable plan shown at the confirm step

---

## 4. Internal steps

1. **Interpret** — which section owns this complaint?

   | Complaint type | Owner |
   |---|---|
   | "numbers are off / return is wrong" | performance (data) |
   | "missed the CPI print / wrong macro fact" | market |
   | "risk section ignored the DV01 concentration" | risk |
   | "tone too defensive / restructure the outlook" | newsletter writer |

2. **Diagnose** — data issue or generation issue?
   - Tier 2: LLM's best guess.
   - Tier 3: actual DB lookup against `pnl_summary` / `pnl_table`.

3. **Rewrite** — turn the vague complaint into a targeted instruction. e.g.
   > "I think the numbers are off"
   > → to performance: *"Re-verify the February return figure and total P&L against
   > pnl_summary; the user believes the reported return is incorrect."*

4. **Materiality** — is the change material to downstream agents?
   (Type-based: numeric/factual = material; tone/wording = not material to numbers.)

5. **Dirty set** — mark the owning agent + any downstream agents the change
   materially affects.

---

## 5. Smart cascade

Agents are not independent. Dependency map:

```
performance ─┐
             ├─→ risk ─→ newsletter
market ──────┘            ↑
weekly ──────────────────┘
newsletter → (leaf, feeds nothing)
```

**Cascade is transitive and order-sensitive.** If a complaint changes
`performance`:
- `risk` must re-run (it read performance), then
- `newsletter` must re-run **after** risk (it read both).

So the cascade is "re-run dependents **in dependency order**" — not ad-hoc
re-runs. The execute phase therefore **reuses `run()`'s existing ordered pipeline**
over the dirty set, keeping one source of truth for ordering.

**Materiality test (Tier 2):** reuse the interpret-step classification.
- numeric / data / factual change → material → auto-include downstream
- tone / structure / wording change → not material to numbers → leave downstream out

---

## 6. Confirm step

Phase A returns a plan; the UI shows what *will* happen and waits:

```
I'll re-run Performance (you flagged the numbers).
Then refresh Risk + Newsletter to stay consistent.
Market and Weekly will be left untouched.
                                   [Confirm]   [Cancel]
```

- **Confirm** → snapshot current result, then run Phase B.
- **Cancel** → discard the plan but **leave the complaint text** so the user can
  edit and re-plan (friendlier than wiping it).

---

## 7. Version stack (within-result history)

Separate from the existing per-query `st.session_state.history`. This is a second
axis: versions **within one query's result**.

- **Structure:** a `version_stack` attached to the current history entry — *not*
  overloaded onto the existing `history` list.
- **What gets snapshotted:** the whole `result` dict, pushed **before** each
  revision executes.
- **Granularity:** one snapshot per revision (even if it touched 3 agents) — matches
  how the user thinks ("go back to before I asked for that change").
- **Label:** a **concise summary of what changed** — no timestamps.
  e.g. *"numbers revision — performance, risk, newsletter"*.
- **Lifespan:** session only (RAM), consistent with current app behavior.
- **Restore = new snapshot** (so the user gets free redo), rather than a destructive
  rollback.

---

## 8. UI changes

- **Top of the response:** version dropdown + single complaint box.
- **Remove** the per-section "Revise this section" boxes from the screen.
- **Confirm panel** between plan and execute.
- **Explanation** shown after execute (*what was re-run and why*), so the user always
  sees that something happened — directly addressing the "nothing came out" problem.

---

## 9. Scope tiers

| Tier | Adds | Effort |
|---|---|---|
| **Tier 1** | interpret + route to existing `revise_section` | small |
| **Tier 2** *(start here)* | + rewrite → targeted query, + explanation, + two-phase confirm, + version stack, + smart cascade | medium |
| **Tier 3** *(later)* | + DB read access: validation agent verifies numerical claims against `pnl_summary` / `pnl_table` and reports data-vs-wording | small *if Tier 2 built right* |

### Why Tier 3 is a clean add-on, not a rewrite
Tier 3 inserts **one step** (`diagnose` backed by a DB lookup). Everything around it
is unchanged. To keep it clean, Tier 2 must:
- keep `diagnosis` as its own output field (LLM guess now, DB-backed later), and
- give the validation agent its **own DB handle from day one** (dormant in Tier 2 —
  one line, same as the other agents already do).

---

## 10. What this reuses (low net-new code)

- **Planner pattern** (`_plan_execution`) — same structure, for revisions.
- **Ordered pipeline** (`run()`) — cascade replays it over the dirty set.
- **Agent `feedback` params** — every agent already accepts targeted feedback.
- **Session-state model** — version stack lives alongside existing `history`.

Net new: one `ValidationAgent` class (two methods) + one orchestrator method that
replays the pipeline over a dirty set + the UI swap to a single box and version
dropdown.

---

## 11. Open micro-decisions (deferrable)

1. **Version labels** — confirmed: concise summary of what changed, no timestamps.
2. **Restore semantics** — confirmed: restore counts as a new snapshot (free redo).
3. **Cancel behavior** — confirmed: keep complaint text for edit-and-retry.
4. **(Open)** Should the confirm panel show the *rewritten targeted query*, or just
   the plain-English plan? (Plain-English is less intimidating; targeted query could
   be tucked behind a "details" expander.)

---

## 12. Build checklist (for when work starts — Tier 2)

- [ ] `ValidationAgent` with `plan_revision()` and `execute_revision(plan)`
- [ ] Plan object: dirty set, targeted queries, `diagnosis` field, materiality reasoning, summary
- [ ] Dormant DB handle on the agent (Tier 3 readiness)
- [ ] Execute phase replays `run()`'s ordered pipeline over the dirty set
- [ ] Per-result `version_stack` in session_state, separate from `history`
- [ ] Snapshot pushed before execute; restore = new snapshot
- [ ] UI: single complaint box + version dropdown at top of response
- [ ] UI: confirm panel between plan and execute
- [ ] UI: remove per-section revise boxes
- [ ] Explanation string returned and displayed after execute
