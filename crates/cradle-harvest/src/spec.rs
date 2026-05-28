//! Model `spec.toml` parsing.
//!
//! Each model directory contains a `spec.toml` describing the
//! featurization (`input_shape`) and the label extraction strategy
//! (`label_source` + `[label_extractor.<source>]`).

use std::path::Path;

use serde::{Deserialize, Serialize};

/// Errors parsing `spec.toml`.
#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    /// I/O failure reading the file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// TOML parse failure.
    #[error("toml parse: {0}")]
    Toml(#[from] toml::de::Error),

    /// The spec referenced a `label_source` with no matching
    /// `[label_extractor.<source>]` table.
    #[error("spec missing label_extractor.{0} table")]
    MissingExtractorTable(String),
}

/// Parsed model spec.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelSpec {
    /// Model name (must match the directory basename).
    pub name: String,
    /// Featurization registry key (e.g. `turn_pair_v1`).
    pub input_shape: String,
    /// Label extractor registry key (e.g. `redirect_v1`).
    pub label_source: String,
    /// Optional held-out fraction (val %), default 15.
    #[serde(default = "default_holdout")]
    pub holdout_session_fraction: f32,
    /// Optional test fraction (%), default 15.
    #[serde(default = "default_test")]
    pub test_session_fraction: f32,
    /// Optional acceptance accuracy threshold for receipt 7 (not used
    /// in v0.1 harvest, but parsed so it's not lost on round-trip).
    #[serde(default)]
    pub threshold: Option<f32>,
    /// Optional AUC threshold (binary classifiers).
    #[serde(default)]
    pub auc_threshold: Option<f32>,
    /// Per-source extractor configs. Keyed by the same string as
    /// `label_source`; the harvest looks up `label_extractor.<source>`.
    #[serde(default, rename = "label_extractor")]
    pub label_extractors: std::collections::BTreeMap<String, LabelExtractorConfig>,
}

#[allow(clippy::unnecessary_wraps)]
const fn default_holdout() -> f32 {
    15.0
}

#[allow(clippy::unnecessary_wraps)]
const fn default_test() -> f32 {
    15.0
}

/// Configuration for one label extractor.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct LabelExtractorConfig {
    /// Keywords that signal a positive label when present in the user turn.
    #[serde(default)]
    pub positive_keywords: Vec<String>,
    /// If true, a positive also requires the next assistant turn to
    /// show a behavioral change.
    #[serde(default)]
    pub require_behavioral_change_next_turn: bool,
    /// Don't train on sessions younger than this many days.
    #[serde(default)]
    pub min_session_age_days: u32,
    /// Override for the val split fraction (0–1.0). Falls back to the
    /// top-level `holdout_session_fraction` if absent.
    #[serde(default)]
    pub holdout_session_fraction: Option<f32>,
}

impl ModelSpec {
    /// Load and parse a `spec.toml`.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError::Io`] or [`SpecError::Toml`] on read/parse failure.
    pub fn load(path: &Path) -> Result<Self, SpecError> {
        let text = std::fs::read_to_string(path)?;
        let parsed: Self = toml::from_str(&text)?;
        Ok(parsed)
    }

    /// Look up the config for the spec's declared `label_source`.
    ///
    /// # Errors
    ///
    /// Returns [`SpecError::MissingExtractorTable`] if the `[label_extractor.<source>]`
    /// table is absent from the spec.
    pub fn label_extractor_config(&self) -> Result<LabelExtractorConfig, SpecError> {
        self.label_extractors
            .get(&self.label_source)
            .cloned()
            .ok_or_else(|| SpecError::MissingExtractorTable(self.label_source.clone()))
    }

    /// The val split percent as a `u8` in `[0, 100]`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn val_pct(&self) -> u8 {
        // Accept either a fraction (0–1) or a percent (0–100) — convert.
        let v = self.holdout_session_fraction;
        let pct = if v <= 1.0 { v * 100.0 } else { v };
        pct.clamp(0.0, 100.0) as u8
    }

    /// The test split percent as a `u8` in `[0, 100]`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn test_pct(&self) -> u8 {
        let v = self.test_session_fraction;
        let pct = if v <= 1.0 { v * 100.0 } else { v };
        pct.clamp(0.0, 100.0) as u8
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parses_redirect_spec() {
        let toml_src = r#"
            name = "redirect"
            input_shape = "turn_pair_v1"
            label_source = "redirect_v1"
            holdout_session_fraction = 15.0
            test_session_fraction = 15.0

            [label_extractor.redirect_v1]
            positive_keywords = ["wait", "no", "actually"]
            require_behavioral_change_next_turn = true
            min_session_age_days = 1
        "#;
        let s: ModelSpec = toml::from_str(toml_src).unwrap();
        assert_eq!(s.name, "redirect");
        assert_eq!(s.input_shape, "turn_pair_v1");
        let cfg = s.label_extractor_config().unwrap();
        assert_eq!(cfg.positive_keywords.len(), 3);
        assert!(cfg.require_behavioral_change_next_turn);
    }

    #[test]
    fn missing_extractor_table_errors() {
        let toml_src = r#"
            name = "redirect"
            input_shape = "turn_pair_v1"
            label_source = "redirect_v1"
        "#;
        let s: ModelSpec = toml::from_str(toml_src).unwrap();
        let err = s.label_extractor_config().unwrap_err();
        assert!(err.to_string().contains("redirect_v1"));
    }
}
