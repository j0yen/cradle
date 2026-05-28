//! Shared featurization registry for cradle.
//!
//! Each featurization is named (e.g. `turn_pair_v1`) and turns a
//! semantically-typed input into a fixed-length feature vector.
//!
//! A model's `spec.toml` references a featurization by name via the
//! `input_shape` field; the harvest step looks up the named featurizer
//! and applies it row-by-row. Unknown names return [`FeatureError::UnknownShape`].

// Featurization is float-heavy by construction; opt the whole crate
// into the categories that the workspace marks `warn` and which would
// otherwise trip `-D warnings` on every coefficient.
#![allow(
    clippy::float_arithmetic,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::suboptimal_flops
)]
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Errors returned by featurization lookup and application.
#[derive(Debug, thiserror::Error)]
pub enum FeatureError {
    /// `spec.toml` referenced an `input_shape` not in the registry.
    #[error("unknown input_shape: {0}")]
    UnknownShape(String),
}

/// A pair of consecutive turns: the previous assistant turn and the
/// next user turn. Featurizer input for `turn_pair_v1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnPair {
    /// The previous assistant turn text, if available.
    pub prev_assistant: String,
    /// The user turn that immediately followed.
    pub user_turn: String,
    /// The next assistant turn that followed the user turn, if any.
    /// Used for behavioral-diff signals; not encoded into features.
    #[serde(default)]
    pub next_assistant: Option<String>,
}

/// A fixed-length feature vector emitted by a registered featurizer.
pub type FeatureVec = Vec<f32>;

/// Apply the named featurizer to a turn pair. Returns the feature
/// vector, or [`FeatureError::UnknownShape`] if the name is not
/// registered.
///
/// # Errors
///
/// Returns [`FeatureError::UnknownShape`] when `shape` is not a known
/// featurizer name.
pub fn featurize_turn_pair(shape: &str, pair: &TurnPair) -> Result<FeatureVec, FeatureError> {
    match shape {
        "turn_pair_v1" => Ok(turn_pair_v1(pair)),
        other => Err(FeatureError::UnknownShape(other.to_string())),
    }
}

/// Return the list of featurizer names currently registered.
///
/// Stable across releases — adding a new featurizer is additive.
#[must_use]
pub const fn registered_shapes() -> &'static [&'static str] {
    &["turn_pair_v1"]
}

/// `turn_pair_v1`: a deterministic, hand-engineered featurization of a
/// `(prev_assistant, user_turn)` pair.
///
/// Returned vector layout (length 8, all f32):
///
/// | idx | feature                                     |
/// |-----|---------------------------------------------|
/// |  0  | log1p(prev_assistant char count) / 10       |
/// |  1  | log1p(user_turn char count) / 10            |
/// |  2  | user_turn contains a redirect keyword       |
/// |  3  | user_turn starts with a redirect keyword    |
/// |  4  | user_turn ends with `?` (clarification)     |
/// |  5  | prev_assistant contains a tool-use marker   |
/// |  6  | user_turn token count, log1p-normalized     |
/// |  7  | shared-token ratio (Jaccard, lowercase ASCII) |
///
/// All values are in roughly `[0, 1]` so a linear head can train
/// without aggressive normalization downstream.
#[must_use]
pub fn turn_pair_v1(pair: &TurnPair) -> FeatureVec {
    let prev = pair.prev_assistant.as_str();
    let user = pair.user_turn.as_str();
    let prev_len = log1p_norm(prev.chars().count());
    let user_len = log1p_norm(user.chars().count());
    let lower = user.to_lowercase();
    let has_redirect = f32::from(u8::from(contains_any(&lower, REDIRECT_HINTS)));
    let starts_redirect = f32::from(u8::from(starts_with_any(&lower, REDIRECT_HINTS)));
    let trailing_q = f32::from(u8::from(user.trim_end().ends_with('?')));
    let prev_tooluse = f32::from(u8::from(contains_any(
        &prev.to_lowercase(),
        TOOL_USE_HINTS,
    )));
    let user_tokens = log1p_norm(user.split_whitespace().count());
    let jaccard = jaccard_lower_ascii(prev, user);
    vec![
        prev_len,
        user_len,
        has_redirect,
        starts_redirect,
        trailing_q,
        prev_tooluse,
        user_tokens,
        jaccard,
    ]
}

/// Keywords used as a coarse signal for "the user wants to redirect"
/// in `turn_pair_v1`. The actual label extractor in `cradle-harvest`
/// owns the authoritative list; this constant only mirrors it for
/// featurization. Keep the two in sync — `redirect_v1` extractor
/// reads its own list from `spec.toml`.
const REDIRECT_HINTS: &[&str] = &[
    "wait", "no", "actually", "stop", "go back", "different", "instead",
];

/// Substrings that suggest the previous assistant turn invoked a tool.
const TOOL_USE_HINTS: &[&str] = &[
    "calling function",
    "tool_use",
    "running",
    "executing",
    "let me ",
];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

fn starts_with_any(haystack: &str, needles: &[&str]) -> bool {
    let trimmed = haystack.trim_start();
    needles.iter().any(|n| trimmed.starts_with(n))
}

#[allow(clippy::cast_precision_loss)]
fn log1p_norm(count: usize) -> f32 {
    // Saturating: counts above a few thousand chars all map to ~1.
    // Precision loss on the usize -> f64 cast is fine; we're only
    // computing a feature in [0, 1], not a count.
    let v = (count as f64).ln_1p() / 10.0;
    v.min(1.0) as f32
}

#[allow(clippy::cast_precision_loss)]
fn jaccard_lower_ascii(a: &str, b: &str) -> f32 {
    let toks_a = tokenize_lower(a);
    let toks_b = tokenize_lower(b);
    if toks_a.is_empty() && toks_b.is_empty() {
        return 0.0;
    }
    let inter = toks_a.iter().filter(|t| toks_b.contains(*t)).count();
    let union = toks_a.len() + toks_b.len() - inter;
    if union == 0 {
        return 0.0;
    }
    let v = inter as f64 / union as f64;
    v as f32
}

fn tokenize_lower(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn turn_pair_v1_returns_8_dims() {
        let pair = TurnPair {
            prev_assistant: "hello".into(),
            user_turn: "hi".into(),
            next_assistant: None,
        };
        assert_eq!(featurize_turn_pair("turn_pair_v1", &pair).unwrap().len(), 8);
    }

    #[test]
    fn unknown_shape_is_error() {
        let pair = TurnPair {
            prev_assistant: String::new(),
            user_turn: String::new(),
            next_assistant: None,
        };
        let err = featurize_turn_pair("does_not_exist", &pair).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("does_not_exist"), "got: {msg}");
    }

    #[test]
    fn redirect_keyword_lights_feature_2() {
        let pair = TurnPair {
            prev_assistant: "ran a command".into(),
            user_turn: "wait, do something different".into(),
            next_assistant: None,
        };
        let v = turn_pair_v1(&pair);
        assert!(v[2] > 0.5, "feature[2] redirect-contains should be 1");
        assert!(v[3] > 0.5, "feature[3] redirect-starts should be 1");
    }

    #[test]
    fn registered_shapes_lists_turn_pair_v1() {
        assert!(registered_shapes().contains(&"turn_pair_v1"));
    }
}
