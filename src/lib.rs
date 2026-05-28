//! cradle — the user-facing library surface for the cradle binary.
//!
//! The bulk of the logic lives in two sibling crates:
//!
//! - [`cradle_harvest`] — transcript JSONL → labeled examples + split
//! - [`cradle_features`] — shared featurization registry
//!
//! This crate re-exports the public surface so the binary can `use
//! cradle::...` and so external consumers (autobuilder, tests) have
//! one place to import from.

#![forbid(unsafe_code)]

pub use cradle_features as features;
pub use cradle_harvest as harvest;

pub mod cli;
pub mod orchestrator;

/// Version string for the crate. Kept manually in sync with `Cargo.toml`'s
/// `package.version`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
