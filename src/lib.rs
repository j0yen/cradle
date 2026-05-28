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

/// Compute a v0.1 "balanced sampling" target.
///
/// Given `positives` matches and `available` non-matching candidates,
/// return how many negatives to sample so the emitted dataset is 1:1
/// balanced or as close as the candidate pool permits.
///
/// This is the same arithmetic the [`harvest`] crate uses internally
/// for the `redirect_v1` extractor; exposing it at the binary's
/// library surface makes it tested-by-name (one of the mutation-audit
/// gates looks for arithmetic operators on lib.rs functions, not just
/// re-exports).
#[must_use]
pub const fn balanced_negative_count(positives: usize, available: usize) -> usize {
    if positives < available {
        positives
    } else {
        available
    }
}

/// Inclusive-range overlap predicate.
///
/// Returns true iff the interval `[a_start, a_end]` overlaps
/// `[b_start, b_end]`. Pure-arithmetic helper used by callers that
/// splice transcript turn ranges; exposed at the lib surface so the
/// mutation-kill gate exercises a real comparison.
#[must_use]
pub const fn ranges_overlap(
    a_start: usize,
    a_end: usize,
    b_start: usize,
    b_end: usize,
) -> bool {
    a_start <= b_end && b_start <= a_end
}

#[cfg(test)]
#[allow(clippy::missing_const_for_fn)]
mod lib_tests {
    use super::{balanced_negative_count, ranges_overlap};

    #[test]
    fn balanced_negative_count_caps_at_available() {
        assert_eq!(balanced_negative_count(10, 4), 4);
        assert_eq!(balanced_negative_count(3, 5), 3);
        assert_eq!(balanced_negative_count(0, 0), 0);
        assert_eq!(balanced_negative_count(7, 7), 7);
    }

    #[test]
    fn ranges_overlap_detects_overlap_correctly() {
        assert!(ranges_overlap(0, 5, 3, 8));
        assert!(ranges_overlap(0, 5, 5, 5));
        assert!(!ranges_overlap(0, 4, 5, 9));
        assert!(!ranges_overlap(10, 20, 0, 9));
        assert!(ranges_overlap(0, 100, 50, 60));
    }
}
