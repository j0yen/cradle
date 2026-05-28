# cradle

> Self-trained models for my personal-tools stack. Harvest labels from
> Claude transcripts; orchestrate train and bake into Rust crates.

`cradle` is a workspace of three Rust crates that turn `~/.claude/projects/**/*.jsonl`
transcripts into labeled training data, drive per-model `train.py`
shellouts, and (eventually) emit baked Rust crates via [`morsel`][morsel].

This is the **v0.1** release. The Rust harvest + orchestration core is
complete and gated under `/autobuilder`'s 25-receipt rigor; the
`morsel`-bake integration is deferred to a focused follow-on PRD.

[morsel]: https://github.com/j0yen/morsel

## What v0.1 ships

```
cradle/
├── crates/
│   ├── cradle-harvest/      transcript JSONL → labeled examples + split
│   └── cradle-features/     shared featurization registry (turn_pair_v1)
├── src/                     cradle binary (cli)
│   ├── cli.rs               clap-based subcommands
│   └── orchestrator.rs      harvest -> train shellout -> (deferred) bake
└── models/
    ├── redirect/            ← only model with a real label extractor in v0.1
    ├── session-productivity/  spec only; extractor deferred
    └── playbook-match/        spec only; extractor deferred
```

### Subcommands

| Command | What it does | v0.1 status |
| --- | --- | --- |
| `cradle harvest <model>` | Walk transcripts, apply the named label extractor, write `data/<model>/{train,val,test}.jsonl` | shipped (redirect model only) |
| `cradle train <model>` | Shell out to `uv run python models/<model>/train.py` with stable env vars | shipped (orchestration only — train.py is user-supplied) |
| `cradle bake <model>` | Bake the trained checkpoint via `morsel bake` | **deferred** — see [`PRD-cradle-bake-integration.md`](https://github.com/j0yen/autobuilder/blob/main/PRD-cradle-bake-integration.md) |
| `cradle build <model>` | harvest → train → (skipped) bake | shipped |
| `cradle status` | Print per-model on-disk status. `--json` for machine-readable | shipped |

## Acceptance criteria (intent-card)

12 MUST-level + 1 SHOULD-level criteria, all green at gate time. The
intent-card lives at `agent/intent-card.json`; the acceptance tests
that prove each AC live at `tests/acceptance_acN.rs`.

## Why a Python shellout for `cradle train`?

PyTorch's training story is mature and the user-supplied `train.py`
runs once, offline, per (re-)bake cycle. The trained safetensors leave
the Python world entirely and never reach the consumer. Switching to a
pure-Rust trainer (candle, burn) is a v0.2 question — the harvest /
features / orchestrate surfaces don't change either way.

## Build / run

```
cargo build --workspace
cargo test --workspace
cargo run --bin cradle -- status
cargo run --bin cradle -- harvest redirect --models-dir models --transcripts-dir ~/.claude/projects
```

## Workspace lint posture

- `unsafe_code = "deny"` at the workspace level.
- clippy `pedantic + nursery` as warn; BAD_RUST patterns (`unwrap`, `expect`,
  `panic`, `todo`, `unimplemented`, `dbg!`) as deny in production code.
- `cargo deny check bans licenses sources` is the supported subset (the
  full `cargo deny check` errors against cargo-deny 0.18.3 on a CVSS 4.0
  advisory entry — fixed upstream in 0.18.4+).

## Origin

Built via [`/autobuilder`][autobuilder] from
[`PRD-cradle.md`][prd] on 2026-05-27. The
hand-built prototype it replaces is preserved at
`~/wintermute/cradle-2026-05-27-handbuilt-bak/`.

[autobuilder]: https://github.com/j0yen/autobuilder
[prd]: https://github.com/j0yen/autobuilder/blob/main/PRD-cradle.md

## License

Dual-licensed under MIT OR Apache-2.0. See [`LICENSE-MIT`](LICENSE-MIT)
and [`LICENSE-APACHE`](LICENSE-APACHE).
