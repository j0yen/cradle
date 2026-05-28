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

use cradle_harvest::{HarvestStats, ModelSpec, harvest};
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

    /// Bake is deferred in v0.1.
    #[error(
        "cradle bake: not yet implemented in v0.1 — see PRD-cradle-bake-integration.md \
         (follow-on PRD queued)"
    )]
    BakeDeferred,
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

/// Run the bake stage. v0.1 returns [`OrchestrationError::BakeDeferred`].
///
/// # Errors
///
/// Always returns [`OrchestrationError::BakeDeferred`] until the
/// follow-on PRD lands.
pub const fn run_bake(_model: &str) -> Result<(), OrchestrationError> {
    Err(OrchestrationError::BakeDeferred)
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
