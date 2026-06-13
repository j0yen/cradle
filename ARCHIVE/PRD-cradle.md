# PRD: Self-Trained Embedded Models (codename: *cradle*)

**Author:** Claude (Opus 4.7), for me
**Status:** Draft v0.1
**Date:** 2026-05-22
**Depends on:** [[morsel]] (the Rust ML substrate — primitives + bake CLI)
**Consumed by:** [[episode]], [[spool]], [[self-evaluator]], [[recall]], `/self-review`
**Worked example:** `morsel-redirect` — detect user-redirects in the next 3 turns

---

## TL;DR

`morsel` (sibling PRD) gives me a way to bake a trained model into a Rust source file and consume it as a pure function. `cradle` is what feeds it: a small pipeline that **harvests labeled training data from my own session transcripts**, trains small models against those labels, runs them through `morsel bake`, and emits a Rust crate the rest of my tools can `cargo add`. The labels are nearly free because I'm both labeler and consumer — my JSONL transcripts already contain the signals (redirects, productive sessions, memory-saves, playbook firings) that downstream classifiers want to learn. v0.1 ships three models: `morsel-redirect` (highest-value, smallest scope), `morsel-session-productivity` (uses the same transcripts), and `morsel-playbook-match` (Phase B.5 acceleration). The training receipt becomes one of autobuilder's 7-receipt gate items, which makes the whole "PRD → labeled dataset → trained model → baked crate" loop reproducible and gated.

---

## 1. Why this exists

1. **The labels are already in my transcripts; I'm just not extracting them.** Every time a user wrote "wait, actually" within 3 turns of an action I took, that's a labeled example of user-redirect. Every time I wrote a recall memory after a session, that's a labeled positive for "session was productive enough to remember." Today these signals get matched by regex, missed by regex, or ignored. They should train a model.

2. **`morsel` without a label pipeline is useless to me.** The cat-RNN example in [[morsel]] is a demo; the actual workloads I want to embed are mine, and they need a way to get from "labels in JSONL files" to "weights in a Rust source file." Without `cradle`, `morsel` is a tool with no factory feeding it.

3. **The closed loop is the leverage.** I am the labeler, the trainer, the consumer, *and* the source of more training data. Every session I run produces more labels for the next training cycle. This is what makes small task-specific models viable here in a way they aren't in most settings — annotation cost is normally the killer, and here it's near zero.

4. **Heuristics rot; models drift gracefully.** A regex for "user pushback" misses creative phrasings forever; a model trained on my actual transcripts gets better as my transcripts pile up. The replacement isn't "smarter regex" — it's a different shape.

5. **It slots into autobuilder cleanly.** Autobuilder's 7-receipt gate already proves "the code compiles and the tests pass." Adding "the model achieves ≥ threshold accuracy on held-out data" is one more receipt of the same shape. Training becomes a normal autobuilder run; tuning becomes an autobuilder loop.

---

## 2. Who this is for

- **Me, building the rest of the personal-tools stack.** `episode`, `spool`, `/self-review`, `recall`, `mirror` all have ML-shaped subproblems they're currently solving with regex or punting on.
- **Future-me reading committed weight files.** Const-baked models are *more* inspectable than runtime-loaded models — they live in git, they're versioned, and `git blame` works.
- **Not** for: anyone but me, until the privacy story is solved. Models trained on jsy's communication patterns leak distributional information about jsy. v0.1 keeps everything local; no crates.io publishing.

---

## 3. What I'd use it for (concretely)

The catalog. Each row is a candidate model. v0.1 ships the three at the top; the rest are future work.

| Model | Input | Output | Label source (self-supervised) | Consumer |
| --- | --- | --- | --- | --- |
| **`morsel-redirect`** | a `(prev_assistant_turn, user_turn)` pair, tokenized | P(redirect) | turns where user phrasing + next-assistant-turn behavioral diff both indicate course-correction | `episode`, `spool`, `/self-review` Phase A |
| **`morsel-session-productivity`** | a session-summary feature vector (turn count, tool calls, file edits, recall writes, errors) | P(memorable) | sessions I wrote a recall memory about within 24h | `recall` (prioritize what to keep), `/self-review` |
| **`morsel-playbook-match`** | an anomaly state vector (32 features from `/self-review` Phase A) | k-NN over baked anomaly embeddings → playbook name or "novel" | past `/self-review` runs where a playbook fired and the user accepted the outcome | `/self-review` Phase B.5 dispatch |
| `morsel-journal-paragraph` | a paragraph from `~/brain/journal/YYYY-MM-DD.md` | one of {finding, applied, pending, notable} | the existing section headers in my journal already partition this | `/self-review` Phase E journaling |
| `morsel-memory-dedup` | a candidate memory string | nearest existing memory + cosine sim | implicit: cosine over existing recall memories | `recall save` (warn-before-save) |
| `morsel-topic-boundary` | sliding window over assistant turns | P(this is a topic boundary) | hand-labeled 200 boundaries; bootstrap | `episode` for session segmentation |
| `morsel-receipt-precheck` | proposed code change + receipt config | P(receipt passes) | past autobuilder receipts (pass/fail) | `autobuilder` for early-exit |

v0.1 commits to the bolded three. The rest live in this PRD as a reminder, not a roadmap.

---

## 4. Functional requirements

### 4.1 The pipeline shape

```
~/.claude/projects/**/*.jsonl                          (transcripts, ongoing)
        │
        ▼
   cradle harvest <model_name>                         (extract labeled examples)
        │
        ▼
   data/<model_name>/{train,val,test}.jsonl            (deterministic split by session_id hash)
        │
        ▼
   cradle train <model_name>                           (PyTorch; outputs safetensors)
        │
        ▼
   models/<model_name>/checkpoint.safetensors
        │
        ▼
   morsel bake --in checkpoint.safetensors             (from [[morsel]])
        │
        ▼
   crates/morsel-<model_name>/src/weights.rs           (const arrays; the artifact)
        │
        ▼
   autobuilder receipts gate                           (model accuracy receipt = 1 of 7)
        │
        ▼
   crates/morsel-<model_name>/                         (published or local; consumed by downstream)
```

The whole pipeline is one `cradle build <model_name>` invocation. Each stage is also separately invokable for debugging.

### 4.2 The harvest step

`cradle harvest` reads transcript JSONLs and emits labeled examples per the model spec. The spec lives in `models/<name>/spec.toml`:

```toml
# models/redirect/spec.toml
name = "redirect"
input_shape = "turn_pair_v1"        # references a shared featurization in cradle-harvest
label_source = "redirect_v1"        # references a label-extraction strategy

[label_extractor.redirect_v1]
positive_keywords = ["wait", "no", "actually", "stop", "go back", "different"]
require_behavioral_change_next_turn = true   # next assistant turn must show diff: revert, switch tool, change file
min_session_age_days = 1                     # don't train on very recent sessions to allow user redirects to surface
holdout_session_fraction = 0.15
```

Outputs `data/redirect/train.jsonl`, `val.jsonl`, `test.jsonl`. Each row: `{features: [...], label: 0|1, source_session: ..., source_turn: ...}`. Session IDs determine the split (so train/test never share a session); turn-level shuffle is fine within sessions.

### 4.3 The train step

`cradle train` invokes PyTorch under `uv run`. v0.1 picks PyTorch over candle because the training story is better understood; the trained weights end up as safetensors and `morsel bake` doesn't care which framework produced them. Training script lives at `models/<name>/train.py` and is hand-written per model in v0.1 (no autogen yet). Output: `models/<name>/checkpoint.safetensors` + `metrics.json` (val/test accuracy, AUC, confusion matrix).

### 4.4 The bake step

Delegated to `morsel bake` from [[morsel]]. `cradle` calls it with the right `--arch`, `--quant`, `--out` flags. No new code here.

### 4.5 The receipt step

The output crate is a normal Rust crate. autobuilder's receipt gate (existing) checks: compiles, clippy clean, tests pass, docs build, fingerprint matches. `cradle` adds **one** receipt:

```
receipt 7: model held-out accuracy ≥ spec.threshold
  - reads models/<name>/metrics.json
  - asserts metrics.test_accuracy >= spec.threshold
  - asserts metrics.test_auc >= spec.auc_threshold (if classifier)
  - on fail: emits failure receipt with confusion matrix as evidence
```

The 7-receipt threshold is unchanged; this is a normal receipt that gates the same way.

### 4.6 The consumer integration

Once `morsel-redirect` is published (or local-pathed), consumers add it to `Cargo.toml` and call its single function. Three concrete consumers:

```rust
// in episode/src/turn_observer.rs
use morsel_redirect::redirect_probability;
if redirect_probability(prev_assistant, user_turn) > 0.7 {
    emit_event("redirect_detected", session_id, turn_index);
}

// in spool/src/backfill.rs  
// (replaces today's keyword-match logic)
use morsel_redirect::redirect_probability;

// in self-review Phase A
// (just reads episode's emitted events; doesn't import morsel directly)
```

### 4.7 Storage layout

```
~/projects/cradle/
├── PRD-cradle.md                       (this file, mirrored)
├── crates/
│   ├── cradle-cli/                     (the `cradle` binary)
│   ├── cradle-harvest/                 (transcript → labeled dataset)
│   └── cradle-features/                (shared featurization: turn_pair_v1, session_v1, etc.)
├── models/
│   ├── redirect/
│   │   ├── spec.toml
│   │   ├── train.py
│   │   ├── data/                       (gitignored — large, regenerable)
│   │   ├── checkpoint.safetensors      (gitignored)
│   │   └── metrics.json                (committed — small, useful for diffing across runs)
│   ├── session-productivity/
│   └── playbook-match/
└── output/                             (gitignored)
    └── morsel-redirect/                (the baked Rust crate, regenerable)
```

The baked output crates live *outside* `cradle/` — they're regenerable artifacts. Their canonical home is `~/projects/morsel-<name>/` or wherever the consumer chooses to path-dep.

---

## 5. Architecture

Cradle is one Rust binary (`cradle-cli`) plus a thin Python training shell per model. The binary handles harvest, orchestration, and the bake invocation; Python handles training (because that's where the ecosystem is for v0.1).

```
~/.local/bin/cradle
  harvest <model_name>                  # transcripts → data/<name>/{train,val,test}.jsonl
  train   <model_name>                  # invokes models/<name>/train.py under uv
  bake    <model_name>                  # invokes morsel-bake on the safetensors
  build   <model_name>                  # harvest → train → bake → receipt-check in sequence
  status                                # which models built, when, what accuracy
  rebuild-all                           # for after a featurization bump
```

`cradle build` is the autobuilder-callable entry point. Phasing-wise (§7), `cradle` itself can be built by autobuilder once `morsel` ships — meta-bootstrapping the model factory.

---

## 6. Non-goals

1. **Training in production / online learning.** Models are trained offline, baked, and shipped. The consumer never trains. No drift detection at runtime (v0.1).
2. **Multi-user data.** Trained only on jsy's transcripts. Models stay local. No publishing.
3. **Big models.** Same envelope as [[morsel]]: ≤1MB of weights per model. Anything bigger goes to `fastembed-rs` (BGE in [[recall]] v0.2) or `candle`.
4. **A general AutoML system.** Each model has a hand-written train.py in v0.1. Hyperparameter search is autobuilder's iterate loop, not a sweep framework.
5. **Replacing rule-based systems wholesale.** Where regex is clearly fine (exact match of a known string), it stays. Models replace regex only where the labels exist and the regex is missing signal.
6. **Cross-task model sharing in v0.1.** Bespoke embedder per task for the first three. The shared-embedder hypothesis is a v0.2 experiment (§9).

---

## 7. Phasing

| Phase | Scope |
| --- | --- |
| **0** | `cradle-harvest` for one label spec (`redirect_v1`). Outputs labeled JSONL. Sanity-check label quality by spot-reading 50 positive examples. |
| **1** | `cradle train` for `redirect` model. PyTorch, ~1K parameters (LogReg or tiny MLP). Validate held-out accuracy ≥ 0.85. Hand-bake via `morsel bake`. Publish as local-path crate. Wire `episode` (or build a stub if `episode` isn't built yet) to consume it. |
| **2** | `cradle build redirect` end-to-end one-shot. Same model, no behavior change — just automation. autobuilder receipt 7 (accuracy) gates the build. |
| **3** | Add `session-productivity` and `playbook-match` models, each with their own spec + train.py + baked crate. Test that the orchestrator handles multiple models. |
| **4** | Shared-embedder experiment: replace the bespoke featurization for two of the three models with a single 16-dim turn-embedder + linear head per task. Compare accuracy-per-parameter. Decide whether to migrate the third model. |
| **5** | The rest of the catalog (§3) gets PRDs of its own as I feel the need. |

Phase 0–1 should take a day. Phase 2 is the autobuilder integration; that's where the interesting work is.

---

## 8. Risks

- **Label noise.** Self-supervised labels are heuristic — the redirect-keyword + behavioral-diff rule has false positives. The model trained on noisy labels will inherit that noise. *Mitigation:* spot-check labels (Phase 0); use behavioral-diff as the *required* signal so keyword-only matches don't dominate; treat the first model as a v0.1 that I'll re-train against cleaner labels later.
- **Distributional shift.** I am the only source of training data. If I change how I write or interact, models trained on past me misfire on future me. *Mitigation:* re-bake monthly; track `metrics.json` diffs across runs to catch drift early; if the model gets worse over time, that's data, not just a bug.
- **Label leakage / overfitting.** Splitting by session is the right primitive (turns within a session are non-iid). But sessions still cluster — a week of intense Rust work could overrepresent that domain. *Mitigation:* stratify holdout by week, not just session.
- **Embedded model invisible until it misfires.** Once baked, the model is silent inside whatever crate consumes it. A regression is harder to notice than a regex regression. *Mitigation:* every consumer logs `{model_version, input_hash, output}` for one of every N invocations to `~/.claude/spool/<YYYY-MM>.jsonl` (per [[spool]]). Closes the observability loop.
- **Privacy.** Trained-on-my-transcripts models leak distributional facts about jsy. *Mitigation:* models are local; never published; the baked weight files live in a private repo; PII filter in the harvest step strips obvious identifiers (emails, paths) before training.
- **Autobuilder receipt is gameable.** A model could hit ≥0.85 accuracy by predicting the majority class on an imbalanced dataset. *Mitigation:* the receipt also checks per-class recall (or AUC for binary), not just accuracy. Spec.toml encodes the bar.
- **PyTorch dep is heavy and Python-y.** Goes against the "Rust-everywhere" instinct. *Mitigation:* training is a one-shot offline step run under `uv`; consumers never see Python. If candle's training story matures, swap it in without touching the bake / consume sides.

---

## 9. Open questions

1. **Bespoke features vs shared embedder.** Phase 4 is the cheap experiment. I'd guess shared-embedder wins above ~3 tasks, but it's a guess. Don't pre-commit to the shared embedder architecture; let the data decide.
2. **How fresh should training data be?** `spec.toml` has `min_session_age_days = 1` for redirect — too low, and the user's not-yet-surfaced corrections are missing; too high, and the model is stale. Tune per-model.
3. **One crate per model, or one crate with many functions?** v0.1 says one crate per model (`morsel-redirect`, `morsel-session-productivity`, …). Cleaner versioning, cleaner consumer dependency graphs. Reconsider if I end up with 15 tiny crates.
4. **Does `cradle` train its own embedder, or use [[recall]]'s BGE?** For *some* tasks (turn-topic-classifier) the BGE embeddings might be great features. But BGE is 130MB and lives in [[recall]]; pulling it into the harvest step couples the two systems. v0.1: train bespoke features. v0.2 question: can a consumer crate that already has BGE loaded (because it uses [[recall]]) reuse those embeddings as input to a `morsel-*` head?
5. **Should `cradle build` be allowed to *update* a baked crate that's already in use, or only emit new versions?** Probably new versions (semver bump). The consumer pins. Re-baking in place would silently change behavior, which is the opposite of the "models live in git" property I want.
6. **Co-located with autobuilder, or its own repo?** Right now this PRD lives in `~/projects/autobuilder/`. The implementation probably wants its own repo (`~/projects/cradle/`) because it accumulates training datasets and that bloats autobuilder's git history. Decide at build time.

---

## 10. Relationship to other PRDs

- **[[morsel]]** — the substrate. Cradle produces; morsel hosts. No overlap.
- **[[recall]]** — uses fastembed-rs/BGE for *general* embeddings. Cradle uses task-specific small models. Possible v0.2 cooperation (§9.4).
- **[[episode]]** — the canonical consumer for `morsel-redirect`. The PRD-episode redirect detector currently described as "keyword matching" should reference this PRD instead once both exist.
- **[[spool]]** — consumes `morsel-redirect` for skill-invocation backfill (currently described as keyword matching in PRD-skill-telemetry §4.3); also observes baked-model behavior over time (§8 mitigation above).
- **[[self-evaluator]]** (mirror) — could consume `morsel-session-productivity` to decide which sessions to write reflective memories about.
- **`/self-review`** — Phase A consumes `morsel-playbook-match`; Phase E consumes `morsel-journal-paragraph` (future).
- **autobuilder** — both the build pipeline for cradle's outputs *and* a downstream consumer (`morsel-receipt-precheck`, future).

---

## Appendix: Acceptance Criteria (AC1–AC12, v0.1)

The full criteria with predicates live in `agent/intent-card.json`. The IDs are
referenced by `tests/acceptance_*.rs`:

- AC1 — workspace builds clean (cargo build --workspace)
- AC2 — `cradle --help` lists subcommands; unknown returns nonzero
- AC3 — `cradle harvest` emits train/val/test JSONL with stable schema
- AC4 — split is deterministic and stable across runs
- AC5 — cradle-features registry surfaces turn_pair_v1; unknown returns error
- AC6 — redirect_v1 label extractor matches PRD §4.2 semantics
- AC7 — `cradle train` shells out and surfaces runner failures
- AC8 — `cradle bake` is documented as deferred to PRD-cradle-bake-integration.md
- AC9 — `cradle status` prints text and `--json` modes
- AC10 — `cradle build` chains harvest -> train with skip semantics
- AC11 — workspace passes clippy -D warnings + `cargo deny check bans licenses sources`
- AC12 — harvest emits grep-parseable stats line on stderr
