//! Test-only introspection into launch-level decisions.
//!
//! Some launch decisions — like whether a fused kernel wrote its output in-place into an
//! input buffer — produce identical results either way, so value-based tests cannot catch
//! a silent regression. The counters here make those decisions observable.
//!
//! Counters are process-global: concurrent tests on the same process all increment them.
//! Assert on deltas (`>= n`), or serialize the test (e.g. `#[serial]`) when asserting that
//! a counter did *not* move.

use core::sync::atomic::{AtomicU64, Ordering};

static INPLACE_ALIAS_COUNT: AtomicU64 = AtomicU64::new(0);

/// The number of fused-kernel outputs that aliased (reused) an input buffer instead of
/// allocating their own, since process start.
pub fn inplace_alias_count() -> u64 {
    INPLACE_ALIAS_COUNT.load(Ordering::Relaxed)
}

pub(crate) fn record_inplace_alias() {
    INPLACE_ALIAS_COUNT.fetch_add(1, Ordering::Relaxed);
}
