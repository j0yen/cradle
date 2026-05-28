//! Transcript JSONL → labeled examples for cradle.
//!
//! Reads `~/.claude/projects/**/*.jsonl` (or a configurable root),
//! applies a named label extractor from a model `spec.toml`, applies
//! the featurization referenced by `input_shape`, and writes
//! `data/<model>/{train,val,test}.jsonl` with a deterministic
//! session-keyed split.
//!
//! v0.1 ships one extractor: `redirect_v1`.

#![forbid(unsafe_code)]
#![allow(
    clippy::float_arithmetic,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown
)]

pub mod spec;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use cradle_features::{FeatureError, TurnPair, featurize_turn_pair};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

pub use spec::{LabelExtractorConfig, ModelSpec, SpecError};

/// Errors returned during harvest.
#[derive(Debug, thiserror::Error)]
pub enum HarvestError {
    /// I/O failure while reading transcripts or writing data files.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// `spec.toml` failed to parse or was missing required fields.
    #[error("spec error: {0}")]
    Spec(#[from] SpecError),

    /// `input_shape` referenced an unknown featurizer.
    #[error("feature error: {0}")]
    Feature(#[from] FeatureError),

    /// `label_source` referenced an extractor cradle-harvest doesn't
    /// ship in v0.1.
    #[error("unknown label_source: {0} (v0.1 ships only redirect_v1)")]
    UnknownLabelSource(String),

    /// Walking the transcripts directory failed.
    #[error("walk error: {0}")]
    Walk(#[from] walkdir::Error),
}

/// One labeled example emitted by harvest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledExample {
    /// Feature vector produced by the named featurizer.
    pub features: Vec<f32>,
    /// 0 or 1 for the binary label.
    pub label: u8,
    /// Stable session identifier the example came from.
    pub source_session: String,
    /// Turn index within the session.
    pub source_turn: usize,
}

/// Aggregate stats from a harvest run, written to stderr.
#[derive(Debug, Default, Clone, Serialize)]
pub struct HarvestStats {
    /// Number of positive labels emitted.
    pub positive_count: usize,
    /// Number of negative labels emitted.
    pub negative_count: usize,
    /// Distinct sessions seen across all walked JSONL files.
    pub sessions_seen: usize,
    /// Distinct turns scanned (across all sessions).
    pub turns_scanned: usize,
    /// Distinct sessions placed in train.
    pub sessions_train: usize,
    /// Distinct sessions placed in val.
    pub sessions_val: usize,
    /// Distinct sessions placed in test.
    pub sessions_test: usize,
}

impl HarvestStats {
    /// Format the stats as a single-line grep-friendly summary, e.g.
    /// `cradle-harvest: pos=12 neg=34 sessions=5 turns=200 split=3/1/1`.
    #[must_use]
    pub fn one_line(&self) -> String {
        format!(
            "cradle-harvest: pos={} neg={} sessions={} turns={} split={}/{}/{}",
            self.positive_count,
            self.negative_count,
            self.sessions_seen,
            self.turns_scanned,
            self.sessions_train,
            self.sessions_val,
            self.sessions_test,
        )
    }
}

/// One side of a parsed transcript turn. `role` is one of
/// `"user"` / `"assistant"`. `content` is the text payload.
#[derive(Debug, Clone)]
pub struct Turn {
    /// `"user"` or `"assistant"`.
    pub role: String,
    /// Text payload (concatenated from message content blocks).
    pub content: String,
}

/// A single parsed transcript: an ordered sequence of turns.
#[derive(Debug, Clone)]
pub struct Transcript {
    /// Stable session identifier (path basename or recorded `session_id`).
    pub session_id: String,
    /// Ordered turns.
    pub turns: Vec<Turn>,
}

/// Walk `root` recursively, returning all `*.jsonl` files in a stable
/// (sorted) order so harvest output is reproducible.
///
/// # Errors
///
/// Returns [`HarvestError::Walk`] if the directory walk fails.
pub fn list_transcripts(root: &Path) -> Result<Vec<PathBuf>, HarvestError> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root).follow_links(false).sort_by_file_name() {
        let entry = entry?;
        if entry.file_type().is_file()
            && entry
                .path()
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"))
        {
            out.push(entry.path().to_path_buf());
        }
    }
    Ok(out)
}

/// Parse a single transcript JSONL file into a [`Transcript`].
///
/// The parser is lenient about transcript schema variations: it pulls
/// `role` and a concatenated text from any `message.content` array of
/// `{ "type": "text", "text": "..." }` blocks, falling back to a top-
/// level `text` field. Lines that don't parse are skipped.
///
/// # Errors
///
/// Returns [`HarvestError::Io`] only if the file can't be opened or read.
pub fn parse_transcript(path: &Path) -> Result<Transcript, HarvestError> {
    let session_id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut turns = Vec::new();
    for line in r.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        if let Some(turn) = parse_turn_value(&v) {
            turns.push(turn);
        }
    }
    Ok(Transcript { session_id, turns })
}

fn parse_turn_value(v: &serde_json::Value) -> Option<Turn> {
    // Shapes we accept:
    //   { "type": "user", "message": {"role": "user", "content": [...] } }
    //   { "role": "assistant", "content": [...] }
    //   { "role": "user", "content": "text" }
    let msg = v.get("message").unwrap_or(v);
    let role = msg
        .get("role")
        .and_then(serde_json::Value::as_str)
        .or_else(|| v.get("type").and_then(serde_json::Value::as_str))?
        .to_string();
    if role != "user" && role != "assistant" {
        return None;
    }
    let content = match msg.get("content") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(blocks)) => blocks
            .iter()
            .filter_map(extract_text_block)
            .collect::<Vec<_>>()
            .join("\n"),
        _ => return None,
    };
    Some(Turn { role, content })
}

fn extract_text_block(block: &serde_json::Value) -> Option<String> {
    let ty = block.get("type").and_then(serde_json::Value::as_str)?;
    if ty == "text" {
        block
            .get("text")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    } else {
        None
    }
}

/// Which split a session belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Split {
    /// Training set.
    Train,
    /// Validation (development) set.
    Val,
    /// Held-out test set.
    Test,
}

/// Decide a session's split deterministically. SHA-256(session_id) is
/// reduced to a value in `[0, 100)`; `[0, test_pct)` is test,
/// `[test_pct, test_pct + val_pct)` is val, the remainder is train.
///
/// # Panics
///
/// Never panics — the hash and modulo are total.
#[must_use]
pub fn split_for_session(session_id: &str, val_pct: u8, test_pct: u8) -> Split {
    debug_assert!(u16::from(val_pct) + u16::from(test_pct) <= 100);
    let mut hasher = Sha256::new();
    hasher.update(session_id.as_bytes());
    let digest = hasher.finalize();
    // Use first byte mod 100 — stable, monotone in the byte value.
    let Some(&first) = digest.as_slice().first() else {
        return Split::Train;
    };
    let bucket = first % 100;
    if bucket < test_pct {
        Split::Test
    } else if bucket < test_pct.saturating_add(val_pct) {
        Split::Val
    } else {
        Split::Train
    }
}

/// Build labeled examples from a transcript using the `redirect_v1`
/// extractor.
///
/// Positive: a user turn that contains a positive keyword from the
/// spec, *and* (when `require_behavioral_change_next_turn` is true)
/// the next assistant turn shows a behavioral signal — a tool-use
/// hint, a revert keyword, or markedly different content than the
/// prior assistant turn.
///
/// Negative: a user turn with no positive keywords and no behavioral
/// follow-on; sampled (every Nth such turn) until balanced with
/// positives.
fn extract_redirect_v1(
    transcript: &Transcript,
    spec: &ModelSpec,
    cfg: &LabelExtractorConfig,
) -> Result<Vec<LabeledExample>, HarvestError> {
    let mut out = Vec::new();
    let mut positives = 0usize;
    let mut neg_candidates: Vec<(usize, &Turn, &Turn)> = Vec::new();
    let turns = &transcript.turns;
    for i in 1..turns.len() {
        let prev = turns.get(i.saturating_sub(1));
        let curr = turns.get(i);
        let next = turns.get(i + 1);
        let (Some(prev), Some(curr)) = (prev, curr) else {
            continue;
        };
        if prev.role != "assistant" || curr.role != "user" {
            continue;
        }
        let lower = curr.content.to_lowercase();
        let has_keyword = cfg
            .positive_keywords
            .iter()
            .any(|k| lower.contains(&k.to_lowercase()));
        let behavioral = next.is_some_and(|n| {
            n.role == "assistant"
                && (contains_tool_marker(&n.content) || differs_meaningfully(prev, n))
        });
        let pair = TurnPair {
            prev_assistant: prev.content.clone(),
            user_turn: curr.content.clone(),
            next_assistant: next.map(|n| n.content.clone()),
        };
        let features = featurize_turn_pair(&spec.input_shape, &pair)?;
        if has_keyword && (!cfg.require_behavioral_change_next_turn || behavioral) {
            out.push(LabeledExample {
                features,
                label: 1,
                source_session: transcript.session_id.clone(),
                source_turn: i,
            });
            positives += 1;
        } else if !has_keyword {
            // Negative candidate: user turn with no positive keyword.
            // Behavioral signal is only meaningful as part of the
            // positive-class definition (it's a necessary condition
            // for a redirect to actually have *been* a redirect).
            // Non-keyword user turns are the negative pool regardless.
            neg_candidates.push((i, prev, curr));
        }
    }
    // Sample negatives to match positives 1:1 (or take all if fewer
    // candidates). Deterministic: take the first `positives` entries.
    let sample_n = positives.min(neg_candidates.len());
    for (i, prev, curr) in neg_candidates.into_iter().take(sample_n) {
        let pair = TurnPair {
            prev_assistant: prev.content.clone(),
            user_turn: curr.content.clone(),
            next_assistant: None,
        };
        let features = featurize_turn_pair(&spec.input_shape, &pair)?;
        out.push(LabeledExample {
            features,
            label: 0,
            source_session: transcript.session_id.clone(),
            source_turn: i,
        });
    }
    Ok(out)
}

fn contains_tool_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    ["tool_use", "calling function", "running", "executing", "let me "]
        .iter()
        .any(|m| lower.contains(m))
}

fn differs_meaningfully(prev: &Turn, next: &Turn) -> bool {
    // Coarse signal: char count differs by > 30% or the first 50 chars differ.
    let pa: usize = prev.content.chars().count();
    let pb: usize = next.content.chars().count();
    if pa == 0 && pb == 0 {
        return false;
    }
    #[allow(clippy::cast_precision_loss)]
    let ratio = (pa.abs_diff(pb)) as f64 / (pa.max(pb).max(1)) as f64;
    let head_a: String = prev.content.chars().take(50).collect();
    let head_b: String = next.content.chars().take(50).collect();
    ratio > 0.3 || head_a != head_b
}

/// Apply the spec's label extractor to a transcript.
///
/// # Errors
///
/// Returns [`HarvestError::UnknownLabelSource`] if the spec references
/// an extractor v0.1 doesn't ship, or [`HarvestError::Feature`] if the
/// featurization fails.
pub fn extract_labels(
    transcript: &Transcript,
    spec: &ModelSpec,
) -> Result<Vec<LabeledExample>, HarvestError> {
    let cfg = spec.label_extractor_config()?;
    match spec.label_source.as_str() {
        "redirect_v1" => extract_redirect_v1(transcript, spec, &cfg),
        other => Err(HarvestError::UnknownLabelSource(other.to_string())),
    }
}

/// Run the full harvest: walk transcripts, parse each, extract labels,
/// split by session hash, write `train/val/test.jsonl` to `out_dir`.
///
/// # Errors
///
/// Propagates I/O, spec, walk, and feature errors. See [`HarvestError`].
pub fn harvest(
    transcripts_root: &Path,
    spec: &ModelSpec,
    out_dir: &Path,
) -> Result<HarvestStats, HarvestError> {
    std::fs::create_dir_all(out_dir)?;
    let train_path = out_dir.join("train.jsonl");
    let val_path = out_dir.join("val.jsonl");
    let test_path = out_dir.join("test.jsonl");
    let mut train = BufWriter::new(File::create(&train_path)?);
    let mut val = BufWriter::new(File::create(&val_path)?);
    let mut test = BufWriter::new(File::create(&test_path)?);
    let mut stats = HarvestStats::default();
    let mut sessions_by_split: BTreeMap<String, Split> = BTreeMap::new();
    let val_pct = spec.val_pct();
    let test_pct = spec.test_pct();
    for path in list_transcripts(transcripts_root)? {
        let transcript = parse_transcript(&path)?;
        if transcript.turns.is_empty() {
            continue;
        }
        stats.sessions_seen += 1;
        stats.turns_scanned += transcript.turns.len();
        let split = split_for_session(&transcript.session_id, val_pct, test_pct);
        sessions_by_split.insert(transcript.session_id.clone(), split);
        let labels = extract_labels(&transcript, spec)?;
        for ex in labels {
            if ex.label == 1 {
                stats.positive_count += 1;
            } else {
                stats.negative_count += 1;
            }
            let line = serde_json::to_string(&ex)?;
            let writer: &mut BufWriter<File> = match split {
                Split::Train => &mut train,
                Split::Val => &mut val,
                Split::Test => &mut test,
            };
            writeln!(writer, "{line}")?;
        }
    }
    train.flush()?;
    val.flush()?;
    test.flush()?;
    for s in sessions_by_split.values() {
        match s {
            Split::Train => stats.sessions_train += 1,
            Split::Val => stats.sessions_val += 1,
            Split::Test => stats.sessions_test += 1,
        }
    }
    Ok(stats)
}

impl From<serde_json::Error> for HarvestError {
    fn from(value: serde_json::Error) -> Self {
        Self::Io(std::io::Error::other(value))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn split_is_deterministic() {
        let a = split_for_session("session-a", 15, 15);
        let b = split_for_session("session-a", 15, 15);
        assert_eq!(a, b);
    }

    #[test]
    fn split_distributes_across_buckets() {
        let mut tr = 0;
        let mut va = 0;
        let mut te = 0;
        for i in 0..200 {
            match split_for_session(&format!("s-{i}"), 15, 15) {
                Split::Train => tr += 1,
                Split::Val => va += 1,
                Split::Test => te += 1,
            }
        }
        // Loose bounds — distribution should be roughly 70/15/15.
        assert!(tr > 100);
        assert!(va > 5);
        assert!(te > 5);
    }

    #[test]
    fn parse_transcript_handles_simple_user_assistant() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"text","text":"hi"}}]}}}}"#
        )
        .unwrap();
        writeln!(
            f,
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"text","text":"hello"}}]}}}}"#
        )
        .unwrap();
        let t = parse_transcript(&path).unwrap();
        assert_eq!(t.turns.len(), 2);
        assert_eq!(t.turns[0].role, "user");
        assert_eq!(t.turns[1].content, "hello");
    }
}
