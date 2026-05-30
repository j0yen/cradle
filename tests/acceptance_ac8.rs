//! Acceptance test for AC8 (MUST) — updated by PRD-cradle-bake-integration.
//!
//! Project: cradle (cli)
//! AC description: `cradle bake <model>` shells out to `morsel bake` with
//! arch/quant from spec; when the `[bake]` table is absent from spec.toml,
//! returns a clear `BakeSpecMissing` error naming the missing field. When
//! `metrics.json` is absent, returns `MetricsMissing`. When accuracy is
//! below threshold, returns `AccuracyBelowThreshold`.
//!
//! Ownership split: body owned by the edit-agent.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::doc_markdown, clippy::indexing_slicing, clippy::panic, clippy::missing_panics_doc, clippy::float_cmp, clippy::missing_const_for_fn, clippy::similar_names, clippy::redundant_clone, clippy::option_if_let_else, clippy::needless_collect, clippy::bool_assert_comparison, clippy::large_stack_arrays)]

use std::fs;
use std::path::Path;

use cradle::orchestrator::{OrchestrationError, run_bake};

fn write_spec(model_dir: &Path, with_bake: bool, threshold: f32) {
    let bake_section = if with_bake {
        r#"
[bake]
arch = "logreg"
quant = "q8"
crate_name = "morsel-redirect"
"#
    } else {
        ""
    };
    let content = format!(
        r#"name = "redirect"
input_shape = "turn_pair_v1"
label_source = "redirect_v1"
threshold = {threshold}
auc_threshold = {threshold}
{bake_section}"#
    );
    fs::write(model_dir.join("spec.toml"), content).unwrap();
}

fn write_metrics(model_dir: &Path, accuracy: f32) {
    let content = format!(r#"{{"test_accuracy": {accuracy}}}"#);
    fs::write(model_dir.join("metrics.json"), content).unwrap();
}

/// AC8a: missing [bake] table returns BakeSpecMissing.
#[test]
fn acceptance_ac8_missing_bake_spec() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let model_dir = models.join("redirect");
    fs::create_dir_all(&model_dir).unwrap();
    write_spec(&model_dir, false, 0.85);
    write_metrics(&model_dir, 0.91);

    let err = run_bake(&models, "redirect", None).unwrap_err();
    assert!(
        matches!(err, OrchestrationError::BakeSpecMissing(_)),
        "expected BakeSpecMissing; got: {err}"
    );
    assert!(
        err.to_string().contains("redirect"),
        "error must name the model; got: {err}"
    );
}

/// AC8b: metrics.json absent returns MetricsMissing.
#[test]
fn acceptance_ac8_metrics_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let model_dir = models.join("redirect");
    fs::create_dir_all(&model_dir).unwrap();
    write_spec(&model_dir, true, 0.85);
    // No metrics.json written.

    let err = run_bake(&models, "redirect", None).unwrap_err();
    assert!(
        matches!(err, OrchestrationError::MetricsMissing(_)),
        "expected MetricsMissing; got: {err}"
    );
}

/// AC8c: accuracy below threshold returns AccuracyBelowThreshold.
#[test]
fn acceptance_ac8_accuracy_below_threshold() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let model_dir = models.join("redirect");
    fs::create_dir_all(&model_dir).unwrap();
    write_spec(&model_dir, true, 0.85);
    write_metrics(&model_dir, 0.70); // below threshold

    let err = run_bake(&models, "redirect", None).unwrap_err();
    assert!(
        matches!(err, OrchestrationError::AccuracyBelowThreshold { .. }),
        "expected AccuracyBelowThreshold; got: {err}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("0.70") || msg.contains("0.7"),
        "error must show actual accuracy; got: {msg}"
    );
    assert!(
        msg.contains("0.85"),
        "error must show threshold; got: {msg}"
    );
}

/// AC8d: happy path — writes output crate.
#[test]
fn acceptance_ac8_happy_path_generates_crate() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let model_dir = models.join("redirect");
    fs::create_dir_all(&model_dir).unwrap();
    write_spec(&model_dir, true, 0.85);
    write_metrics(&model_dir, 0.91); // above threshold

    // Use an override out_dir so we don't need a real checkpoint.
    let out_dir = tmp.path().join("out");
    let result = run_bake(&models, "redirect", Some(&out_dir)).unwrap();
    assert_eq!(result.model, "redirect");
    assert!((result.test_accuracy - 0.91_f32).abs() < 1e-3);
    assert!(out_dir.join("Cargo.toml").exists(), "Cargo.toml must be generated");
    assert!(out_dir.join("src").join("lib.rs").exists(), "src/lib.rs must be generated");
    assert!(out_dir.join("src").join("weights.rs").exists(), "src/weights.rs must be generated");

    // Cargo.toml must declare the right crate name.
    let cargo = fs::read_to_string(out_dir.join("Cargo.toml")).unwrap();
    assert!(cargo.contains("morsel-redirect"), "Cargo.toml must name the crate");
}
