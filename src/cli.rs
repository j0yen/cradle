//! CLI definition for `cradle`.
//!
//! Public so `tests/acceptance_ac2.rs` can drive parse-only checks via
//! `clap::CommandFactory`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Top-level CLI struct.
#[derive(Debug, Parser)]
#[command(
    name = "cradle",
    version,
    about = "Harvest labeled training data from Claude transcripts; orchestrate train/bake"
)]
pub struct Cli {
    /// Subcommand.
    #[command(subcommand)]
    pub command: Command,
}

/// Subcommands. v0.1 surface: harvest, train, bake, build, status.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Walk transcripts and emit labeled train/val/test JSONLs for a model.
    Harvest(HarvestArgs),
    /// Shell out to a per-model train.py under uv to produce a checkpoint.
    Train(TrainArgs),
    /// Bake a checkpoint into a Rust source file (deferred in v0.1; see PRD-cradle-bake-integration.md).
    Bake(BakeArgs),
    /// Run harvest -> train -> bake in sequence; bake is skipped in v0.1.
    Build(BuildArgs),
    /// Print a summary of which models exist on disk and their stage status.
    Status(StatusArgs),
}

/// Args for `cradle harvest`.
#[derive(Debug, clap::Args)]
pub struct HarvestArgs {
    /// Model name (matches a directory under `models/`).
    pub model: String,

    /// Models directory (default: `./models`).
    #[arg(long, default_value = "models")]
    pub models_dir: PathBuf,

    /// Where to scan for transcript JSONLs. Defaults to
    /// `~/.claude/projects` (resolved at runtime).
    #[arg(long)]
    pub transcripts_dir: Option<PathBuf>,

    /// Output root. Defaults to `<models_dir>/<model>/data`.
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
}

/// Args for `cradle train`.
#[derive(Debug, clap::Args)]
pub struct TrainArgs {
    /// Model name.
    pub model: String,

    /// Models directory (default: `./models`).
    #[arg(long, default_value = "models")]
    pub models_dir: PathBuf,

    /// Runner command, default `uv`. The CLI invokes
    /// `<runner> run python <models_dir>/<model>/train.py`.
    #[arg(long, default_value = "uv")]
    pub runner: String,
}

/// Args for `cradle bake`.
#[derive(Debug, clap::Args)]
pub struct BakeArgs {
    /// Model name.
    pub model: String,
}

/// Args for `cradle build`.
#[derive(Debug, clap::Args)]
pub struct BuildArgs {
    /// Model name.
    pub model: String,

    /// Models directory (default: `./models`).
    #[arg(long, default_value = "models")]
    pub models_dir: PathBuf,

    /// Where to scan for transcript JSONLs.
    #[arg(long)]
    pub transcripts_dir: Option<PathBuf>,

    /// Skip the train step (harvest only). Useful for CI smoke tests
    /// that don't have Python on PATH.
    #[arg(long)]
    pub skip_train: bool,
}

/// Args for `cradle status`.
#[derive(Debug, clap::Args)]
pub struct StatusArgs {
    /// Models directory.
    #[arg(long, default_value = "models")]
    pub models_dir: PathBuf,

    /// Emit machine-readable JSON instead of human-readable text.
    #[arg(long)]
    pub json: bool,
}
