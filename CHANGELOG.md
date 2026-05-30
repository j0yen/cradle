# Changelog

## v0.1.1 — 2026-05-30

Implements `cradle bake`: shells out to `morsel bake` with arch/quant from spec.toml, gates on receipt 7 (test_accuracy >= spec.threshold), generates the output Rust crate under `output/morsel-<model>/` with Cargo.toml + src/lib.rs + src/weights.rs. `cradle build <model>` now executes the full harvest → train → bake pipeline end-to-end.
