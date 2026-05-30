//! Acceptance test for AC10 (MUST) — updated by PRD-cradle-bake-integration.
//!
//! Project: cradle (cli)
//! AC description: `cradle build <model>` runs harvest -> train -> bake in
//! sequence and returns nonzero if any stage fails. When [bake] is absent
//! from spec.toml or metrics.json is missing, bake is skipped with a notice.
//! When train is skipped via --skip-train, bake is also skipped (no metrics).
//!
//! Ownership split: body owned by the edit-agent.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown, clippy::indexing_slicing, clippy::panic, clippy::missing_panics_doc, clippy::float_cmp, clippy::missing_const_for_fn, clippy::similar_names, clippy::redundant_clone, clippy::option_if_let_else, clippy::needless_collect, clippy::bool_assert_comparison, clippy::large_stack_arrays)]

use std::fs;
use std::process::Command;

fn bin() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push("debug");
    p.push("cradle");
    p
}

fn ensure_bin_built() {
    let status = Command::new(env!("CARGO"))
        .args(["build", "--bin", "cradle", "--quiet"])
        .status()
        .unwrap();
    assert!(status.success(), "cargo build failed");
}

fn spec_toml(loose: bool) -> String {
    format!(
        r#"name = "redirect"
input_shape = "turn_pair_v1"
label_source = "redirect_v1"
holdout_session_fraction = 15.0
test_session_fraction = 15.0

[label_extractor.redirect_v1]
positive_keywords = ["wait", "no", "actually"]
require_behavioral_change_next_turn = {}
"#,
        !loose
    )
}

fn make_transcript(path: &std::path::Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let lines = [
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"running alpha"}]}}"#,
        r#"{"type":"user","message":{"role":"user","content":[{"type":"text","text":"wait, do something different"}]}}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"running beta entirely different"}]}}"#,
    ];
    fs::write(path, lines.join("\n")).unwrap();
}

#[test]
fn acceptance_ac10() {
    ensure_bin_built();
    let exe = bin();
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let models = root.join("models");
    let model_dir = models.join("redirect");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(model_dir.join("spec.toml"), spec_toml(true)).unwrap();
    let transcripts = root.join("transcripts");
    make_transcript(&transcripts.join("s.jsonl"));

    // Case A: --skip-train with no [bake] in spec.toml (smoke build).
    // With no [bake] table and no metrics.json, bake is skipped with a notice.
    let out = Command::new(&exe)
        .args(["build", "redirect", "--skip-train", "--models-dir"])
        .arg(&models)
        .arg("--transcripts-dir")
        .arg(&transcripts)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "skip-train build should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Bake is skipped because [bake] section is absent; a clear notice appears.
    assert!(
        stdout.contains("bake skipped"),
        "build stdout must mention bake skip; got: {stdout}"
    );

    // Case B: with [bake] but no metrics.json (train skipped) → bake skipped with notice.
    let spec_with_bake = format!(
        "{}\n[bake]\narch = \"logreg\"\nquant = \"q8\"\ncrate_name = \"morsel-redirect\"\n",
        spec_toml(true)
    );
    fs::write(model_dir.join("spec.toml"), &spec_with_bake).unwrap();
    let out = Command::new(&exe)
        .args(["build", "redirect", "--skip-train", "--models-dir"])
        .arg(&models)
        .arg("--transcripts-dir")
        .arg(&transcripts)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "skip-train build with [bake] but no metrics.json should succeed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("bake skipped"),
        "build stdout must note bake skip (no metrics.json); got: {stdout}"
    );

    // Case C: failure propagation — bogus model name → nonzero.
    let out = Command::new(&exe)
        .args(["build", "does-not-exist", "--skip-train", "--models-dir"])
        .arg(&models)
        .arg("--transcripts-dir")
        .arg(&transcripts)
        .output()
        .unwrap();
    assert!(!out.status.success(), "build of missing model must fail");
}
