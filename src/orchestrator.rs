//! Orchestrate the cradle pipeline: harvest -> train -> bake.
//!
//! Each step is invokable independently; `build` chains them with
//! early exit on failure.

#![allow(
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::struct_excessive_bools
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use cradle_harvest::{BakeSpec, HarvestStats, ModelSpec, harvest};
use serde::Serialize;

/// Orchestration errors.
#[derive(Debug, thiserror::Error)]
pub enum OrchestrationError {
    /// Underlying I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Harvest stage error.
    #[error("harvest: {0}")]
    Harvest(#[from] cradle_harvest::HarvestError),

    /// Model directory or spec.toml missing.
    #[error("model {0:?} not found at {1}")]
    ModelMissing(String, PathBuf),

    /// `models/<model>/train.py` missing.
    #[error("train.py missing for model {0} at {1}")]
    TrainScriptMissing(String, PathBuf),

    /// The training runner exited nonzero.
    #[error("train runner exited {0:?}; stderr: {1}")]
    TrainRunnerFailed(Option<i32>, String),

    /// `[bake]` section is missing from the model's spec.toml.
    #[error("model {0:?} spec.toml is missing a [bake] table — add arch/quant/crate_name")]
    BakeSpecMissing(String),

    /// Held-out accuracy is below the threshold declared in spec.toml.
    #[error(
        "model {model:?}: test_accuracy={actual:.4} is below threshold={threshold:.4} (receipt 7 failed)"
    )]
    AccuracyBelowThreshold {
        /// Model name.
        model: String,
        /// Observed accuracy.
        actual: f32,
        /// Required threshold.
        threshold: f32,
    },

    /// `metrics.json` is missing or unparseable; can't gate on accuracy.
    #[error("model {0:?}: metrics.json missing or invalid — run `cradle train` first")]
    MetricsMissing(String),

    /// The bake shell-out (to `morsel bake`) exited nonzero.
    #[error("bake runner exited {0:?}; stderr: {1}")]
    BakeRunnerFailed(Option<i32>, String),

    /// Code generation for the output crate failed.
    #[error("bake codegen failed: {0}")]
    BakeCodegenFailed(String),
}

/// Result of the harvest stage.
#[derive(Debug, Serialize)]
pub struct HarvestResult {
    /// Model name.
    pub model: String,
    /// Stats emitted by the harvest run.
    pub stats: HarvestStats,
    /// Where the JSONL files landed.
    pub data_dir: PathBuf,
}

/// Default transcripts root: `$HOME/.claude/projects`.
#[must_use]
pub fn default_transcripts_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".claude").join("projects");
    }
    PathBuf::from(".claude/projects")
}

/// Load a model spec from `<models_dir>/<model>/spec.toml`.
///
/// # Errors
///
/// Returns [`OrchestrationError::ModelMissing`] if the directory or
/// spec.toml is absent; propagates `cradle_harvest::SpecError` via
/// `HarvestError` on parse failure.
pub fn load_model_spec(
    models_dir: &Path,
    model: &str,
) -> Result<ModelSpec, OrchestrationError> {
    let model_dir = models_dir.join(model);
    let spec_path = model_dir.join("spec.toml");
    if !spec_path.exists() {
        return Err(OrchestrationError::ModelMissing(
            model.to_string(),
            spec_path,
        ));
    }
    let spec = ModelSpec::load(&spec_path).map_err(cradle_harvest::HarvestError::from)?;
    Ok(spec)
}

/// Run the harvest stage.
///
/// # Errors
///
/// Propagates [`OrchestrationError`] on any sub-stage failure.
pub fn run_harvest(
    models_dir: &Path,
    model: &str,
    transcripts_dir: Option<&Path>,
    out_dir: Option<&Path>,
) -> Result<HarvestResult, OrchestrationError> {
    let spec = load_model_spec(models_dir, model)?;
    let default_root = default_transcripts_dir();
    let transcripts_root = transcripts_dir.unwrap_or(&default_root);
    let default_data = models_dir.join(model).join("data");
    let data_dir = out_dir.unwrap_or(&default_data).to_path_buf();
    let stats = harvest(transcripts_root, &spec, &data_dir)?;
    Ok(HarvestResult {
        model: model.to_string(),
        stats,
        data_dir,
    })
}

/// Run the train stage by shelling out to the model's `train.py`.
///
/// The runner is invoked as `<runner> run python <train.py>` (default
/// runner is `uv`). Environment passed:
///
/// - `CRADLE_MODEL_NAME` — the model name
/// - `CRADLE_DATA_DIR` — absolute path to the model's data dir
/// - `CRADLE_OUTPUT_DIR` — absolute path to the model's output dir
///   (the spec dir; train.py is expected to write `checkpoint.safetensors`
///   and `metrics.json` there)
///
/// # Errors
///
/// Returns [`OrchestrationError::TrainScriptMissing`] if `train.py`
/// doesn't exist; [`OrchestrationError::TrainRunnerFailed`] if the
/// runner exits nonzero.
pub fn run_train(
    models_dir: &Path,
    model: &str,
    runner: &str,
) -> Result<(), OrchestrationError> {
    let model_dir = models_dir.join(model);
    let train_py = model_dir.join("train.py");
    if !train_py.exists() {
        return Err(OrchestrationError::TrainScriptMissing(
            model.to_string(),
            train_py,
        ));
    }
    let data_dir = model_dir.join("data").canonicalize().unwrap_or_else(|_| {
        model_dir.join("data")
    });
    let output_dir = model_dir
        .canonicalize()
        .unwrap_or_else(|_| model_dir.clone());
    let mut cmd = Command::new(runner);
    cmd.arg("run")
        .arg("python")
        .arg(&train_py)
        .env("CRADLE_MODEL_NAME", model)
        .env("CRADLE_DATA_DIR", &data_dir)
        .env("CRADLE_OUTPUT_DIR", &output_dir);
    let out = cmd.output()?;
    if out.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    Err(OrchestrationError::TrainRunnerFailed(
        out.status.code(),
        stderr,
    ))
}

/// Path of the output crate root for a given model.
///
/// Result: `<cradle_root>/output/morsel-<model>/`
#[must_use]
pub fn bake_output_root(models_dir: &Path, model: &str) -> PathBuf {
    // `models_dir` is typically `./models`; sibling `output/` is at the
    // same level as `models/`.
    let cradle_root = models_dir
        .parent()
        .unwrap_or(models_dir);
    cradle_root.join("output").join(format!("morsel-{model}"))
}

/// Read and parse `models/<model>/metrics.json`, returning `test_accuracy`.
///
/// # Errors
///
/// Returns [`OrchestrationError::MetricsMissing`] when the file is absent
/// or does not contain a numeric `test_accuracy` field.
pub fn read_test_accuracy(models_dir: &Path, model: &str) -> Result<f32, OrchestrationError> {
    let path = models_dir.join(model).join("metrics.json");
    if !path.exists() {
        return Err(OrchestrationError::MetricsMissing(model.to_string()));
    }
    let text = std::fs::read_to_string(&path)?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|_| OrchestrationError::MetricsMissing(model.to_string()))?;
    let acc = v
        .get("test_accuracy")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| OrchestrationError::MetricsMissing(model.to_string()))?;
    #[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
    Ok(acc as f32)
}

/// Generate the output Rust crate for a baked model.
///
/// Creates `output/morsel-<model>/` with:
/// - `Cargo.toml` declaring `morsel-<crate_name>` depending on the `morsel` library
/// - `src/lib.rs` with a stub `pub mod weights;` and the public `arch`/`quant` constants
/// - `src/weights.rs` with a placeholder (to be filled by the actual bake binary)
///
/// This is the Rust-native codegen path for when no `morsel bake` binary
/// is on PATH; it generates the crate skeleton so the consumer can `cargo add`.
///
/// # Errors
///
/// Returns [`OrchestrationError::BakeCodegenFailed`] on I/O failure.
pub fn generate_bake_crate(
    out_dir: &Path,
    model: &str,
    bake: &BakeSpec,
    checkpoint_path: &Path,
) -> Result<(), OrchestrationError> {
    let src_dir = out_dir.join("src");
    std::fs::create_dir_all(&src_dir)
        .map_err(|e| OrchestrationError::BakeCodegenFailed(e.to_string()))?;

    // Cargo.toml
    let cargo_toml = format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT OR Apache-2.0"
publish = false
description = "Baked weights for model '{model}' (arch={arch} quant={quant})"

[dependencies]
morsel = {{ path = "{morsel_path}" }}
"#,
        crate_name = bake.crate_name,
        model = model,
        arch = bake.arch,
        quant = bake.quant,
        morsel_path = morsel_lib_path(),
    );
    std::fs::write(out_dir.join("Cargo.toml"), cargo_toml)
        .map_err(|e| OrchestrationError::BakeCodegenFailed(e.to_string()))?;

    // src/lib.rs
    let lib_rs = format!(
        r#"//! Baked weights for model `{model}` — arch={arch}, quant={quant}.
//!
//! Generated by `cradle bake {model}`. Do not edit by hand.
//! Re-generate by running `cradle bake {model}` after retraining.

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]

pub mod weights;

/// Inference architecture used when generating these weights.
pub const ARCH: &str = "{arch}";
/// Quantization level used when generating these weights.
pub const QUANT: &str = "{quant}";
/// Original model name.
pub const MODEL_NAME: &str = "{model}";
/// Checkpoint path used to generate this crate (for provenance).
pub const CHECKPOINT_PATH: &str = "{checkpoint_path}";
"#,
        model = model,
        arch = bake.arch,
        quant = bake.quant,
        checkpoint_path = checkpoint_path.display(),
    );
    std::fs::write(src_dir.join("lib.rs"), lib_rs)
        .map_err(|e| OrchestrationError::BakeCodegenFailed(e.to_string()))?;

    // src/weights.rs — placeholder; a real morsel bake binary would fill this
    // with actual const arrays deserialized from the safetensors checkpoint.
    let weights_rs = format!(
        "//! Weight constants for model `{model}` (arch={arch}, quant={quant}).\n\
         //!\n\
         //! This placeholder is generated by `cradle bake` when no `morsel bake`\n\
         //! binary is available. Replace with the output of `morsel bake --in\n\
         //! {checkpoint_path}` once the morsel CLI ships.\n\
         \n\
         /// Number of input features (from cradle-features `{input_shape}`).\n\
         pub const INPUT_DIM: usize = 0; // TODO: filled by morsel bake\n\
         \n\
         /// Number of output classes.\n\
         pub const OUTPUT_DIM: usize = 2; // binary classifier\n",
        model = model,
        arch = bake.arch,
        quant = bake.quant,
        checkpoint_path = checkpoint_path.display(),
        input_shape = "turn_pair_v1",
    );
    std::fs::write(src_dir.join("weights.rs"), weights_rs)
        .map_err(|e| OrchestrationError::BakeCodegenFailed(e.to_string()))?;

    Ok(())
}

/// Resolve the path to the morsel library crate relative to a standard
/// wintermute layout (`~/wintermute/morsel`).
fn morsel_lib_path() -> String {
    if let Ok(home) = std::env::var("HOME") {
        let p = std::path::Path::new(&home)
            .join("wintermute")
            .join("morsel");
        if p.exists() {
            return p.display().to_string();
        }
    }
    // Fallback: relative path assuming co-located checkouts.
    "../morsel".to_string()
}

/// Result of a successful `cradle bake` run.
#[derive(Debug, Serialize)]
pub struct BakeResult {
    /// Model name.
    pub model: String,
    /// Observed test accuracy (from `metrics.json`).
    pub test_accuracy: f32,
    /// Required threshold (from `spec.toml`).
    pub threshold: f32,
    /// Output crate path.
    pub output_path: PathBuf,
}

/// Run the bake stage.
///
/// Steps:
/// 1. Load model spec; require `[bake]` table (else `BakeSpecMissing`).
/// 2. Read `metrics.json`; assert `test_accuracy >= spec.threshold`
///    (receipt 7 gate — `AccuracyBelowThreshold` on failure).
/// 3. Attempt to shell out to `morsel bake`; if not found, fall back
///    to the Rust-native code generator (`generate_bake_crate`).
/// 4. Write the output crate to `output/morsel-<model>/`.
///
/// # Errors
///
/// Returns [`OrchestrationError`] on spec errors, accuracy gate failure,
/// I/O failure, or codegen failure.
pub fn run_bake(
    models_dir: &Path,
    model: &str,
    out_dir_override: Option<&Path>,
) -> Result<BakeResult, OrchestrationError> {
    let spec = load_model_spec(models_dir, model)?;
    let bake = spec
        .bake
        .as_ref()
        .ok_or_else(|| OrchestrationError::BakeSpecMissing(model.to_string()))?;

    // Receipt 7: accuracy gate.
    let accuracy = read_test_accuracy(models_dir, model)?;
    let threshold = spec.threshold.unwrap_or(0.0);
    if accuracy < threshold {
        return Err(OrchestrationError::AccuracyBelowThreshold {
            model: model.to_string(),
            actual: accuracy,
            threshold,
        });
    }

    let checkpoint = models_dir.join(model).join("checkpoint.safetensors");
    let default_out = bake_output_root(models_dir, model);
    let out_dir = out_dir_override.unwrap_or(&default_out);

    // Try shelling out to `morsel bake` first; fall back to Rust codegen.
    let used_morsel_cli = try_morsel_bake_cli(model, bake, &checkpoint, out_dir)?;
    if !used_morsel_cli {
        generate_bake_crate(out_dir, model, bake, &checkpoint)?;
    }

    Ok(BakeResult {
        model: model.to_string(),
        test_accuracy: accuracy,
        threshold,
        output_path: out_dir.to_path_buf(),
    })
}

/// Attempt to invoke `morsel bake` as an external binary.
///
/// Returns `Ok(true)` on success, `Ok(false)` if the binary is not on PATH,
/// and `Err(...)` if the binary is found but exits nonzero.
fn try_morsel_bake_cli(
    _model: &str,
    bake: &BakeSpec,
    checkpoint: &Path,
    out_dir: &Path,
) -> Result<bool, OrchestrationError> {
    // Check if `morsel` binary exists on PATH.
    let found = Command::new("which")
        .arg("morsel")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !found {
        return Ok(false);
    }
    let out = Command::new("morsel")
        .args([
            "bake",
            "--in",
            &checkpoint.display().to_string(),
            "--arch",
            &bake.arch,
            "--quant",
            &bake.quant,
            "--out",
            &out_dir.display().to_string(),
        ])
        .output()?;
    if out.status.success() {
        return Ok(true);
    }
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    Err(OrchestrationError::BakeRunnerFailed(out.status.code(), stderr))
}

/// Status of one model on disk.
#[derive(Debug, Serialize)]
pub struct ModelStatus {
    /// Model name (directory basename).
    pub name: String,
    /// True if `spec.toml` exists and parses.
    pub spec_present: bool,
    /// True if `data/train.jsonl` exists (harvest has been run).
    pub data_present: bool,
    /// True if `checkpoint.safetensors` exists (train has been run).
    pub checkpoint_present: bool,
    /// True if `train.py` exists (training is wired up).
    pub train_script_present: bool,
}

impl ModelStatus {
    /// One-line human-readable summary.
    #[must_use]
    pub fn one_line(&self) -> String {
        format!(
            "{:24}  spec={} data={} train.py={} checkpoint={}",
            self.name,
            yesno(self.spec_present),
            yesno(self.data_present),
            yesno(self.train_script_present),
            yesno(self.checkpoint_present),
        )
    }
}

const fn yesno(b: bool) -> &'static str {
    if b { "yes" } else { "no " }
}

/// List the models in `models_dir` and assess each one.
///
/// # Errors
///
/// Returns [`OrchestrationError::Io`] on directory read failure.
pub fn collect_statuses(models_dir: &Path) -> Result<Vec<ModelStatus>, OrchestrationError> {
    let mut out = Vec::new();
    if !models_dir.exists() {
        return Ok(out);
    }
    let mut entries: Vec<_> = std::fs::read_dir(models_dir)?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(std::fs::DirEntry::path);
    for entry in entries {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let spec = p.join("spec.toml").exists();
        let data = p.join("data").join("train.jsonl").exists();
        let ckpt = p.join("checkpoint.safetensors").exists();
        let train = p.join("train.py").exists();
        out.push(ModelStatus {
            name: name.to_string(),
            spec_present: spec,
            data_present: data,
            train_script_present: train,
            checkpoint_present: ckpt,
        });
    }
    Ok(out)
}
