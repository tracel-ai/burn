//! Test-only introspection into fusion runtime behavior.
//!
//! When a [`FusionInspector`] is installed for a given [`StreamId`], every execution
//! plan that runs on that stream is captured as a [`FusionReport`] describing which
//! [`OperationIr`]s ended up fused into a single kernel and which ran unfused.
//! Handle snapshots from the backing [`HandleContainer`](burn_ir::HandleContainer)
//! are also recorded at quiescent points, enabling leak detection.
//!
//! [`FusionBlock`] and [`BlockKind`] are the same types the logger renders into its
//! Full-level execution table — see [`crate::stream::execution::trace`].
//!
//! # Usage
//!
//! ```ignore
//! # use burn_fusion::inspect::FusionInspector;
//! # use burn_tensor::StreamId;
//! let stream = StreamId::current();
//! let inspector = FusionInspector::install(stream);
//! // ... run tensor ops with a fusion-wrapped backend, then sync ...
//! let reports = inspector.drain();
//! let block = reports[0].assert_single_fused_block();
//! assert_eq!(block.fuser_name(), Some("ElementWise"));
//! ```
//!
//! # Threading
//!
//! The fusion server runs on a background thread per device. Reports are captured on
//! that thread and deposited into a process-global registry keyed by [`StreamId`].
//! Multiple inspectors for *different* streams may coexist freely. Each only sees
//! reports from its own stream, so concurrent tests on different streams are naturally
//! isolated without any serialization.
//!
//! Installing two inspectors for the *same* stream will panic. Use `test_stream()`
//! (which returns a fresh unique [`StreamId`] per call) to avoid accidental sharing.
//!
//! ## Handle leak detection
//!
//! [`HandleContainer`](burn_ir::HandleContainer) is shared per-device across the
//! whole process, so handle snapshots are not stream-scoped. Every live handle on
//! the device appears in the snapshot regardless of which stream created it.
//! Tests that assert on handle counts via [`FusionInspector::new_handles_since_baseline`]
//! must therefore be serialized against each other with `#[serial]`
//! to prevent handles from concurrent tests from appearing as false leaks.

use burn_ir::{OperationIr, TensorId};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::stream::StreamId;
use crate::stream::execution::trace::format_table;
pub use crate::stream::execution::trace::{BlockKind, FusionBlock};

/// The list of fusion decisions captured for one execution plan.
#[derive(Debug, Clone)]
pub struct FusionReport {
    /// One entry per leaf of the executed strategy, in execution order.
    pub blocks: Vec<FusionBlock>,
}

impl FusionReport {
    /// Render the report as the same human-readable table that fusion logging emits at
    /// [`FusionLogLevel::Full`](burn_std::config::fusion::FusionLogLevel). Useful for
    /// rich panic messages from inspector-based tests.
    pub fn format_table(&self) -> String {
        format_table(&self.blocks)
    }

    /// Iterate over the fused blocks only.
    pub fn fused_blocks(&self) -> impl Iterator<Item = &FusionBlock> {
        self.blocks
            .iter()
            .filter(|b| matches!(b.kind, BlockKind::Fused { .. }))
    }

    /// Iterate over the unfused blocks only.
    pub fn unfused_blocks(&self) -> impl Iterator<Item = &FusionBlock> {
        self.blocks
            .iter()
            .filter(|b| matches!(b.kind, BlockKind::Unfused))
    }

    /// The total number of operations across all blocks.
    pub fn total_operations(&self) -> usize {
        self.blocks.iter().map(|b| b.operations.len()).sum()
    }

    /// Assert that exactly one block exists and that it is fused. Returns it.
    ///
    /// # Panics
    /// Panics if the report does not contain exactly one fused block.
    pub fn assert_single_fused_block(&self) -> &FusionBlock {
        assert_eq!(
            self.blocks.len(),
            1,
            "expected exactly one block, got {}: {:#?}",
            self.blocks.len(),
            self.blocks,
        );
        let block = &self.blocks[0];
        assert!(
            matches!(block.kind, BlockKind::Fused { .. }),
            "expected block to be fused, got {:?}",
            block.kind,
        );
        block
    }
}

/// Test-facing helpers on [`FusionBlock`]. Defined here so they aren't part of the
/// always-compiled logging data model.
impl FusionBlock {
    /// The fuser name if this block was fused, `None` otherwise.
    pub fn fuser_name(&self) -> Option<&'static str> {
        match self.kind {
            BlockKind::Fused { name, .. } => Some(name),
            BlockKind::Unfused => None,
        }
    }

    /// Returns `true` if the block's operations match the given matchers one-for-one,
    /// in order.
    pub fn ops_match(&self, matchers: &[OpMatcher]) -> bool {
        if self.operations.len() != matchers.len() {
            return false;
        }
        self.operations
            .iter()
            .zip(matchers.iter())
            .all(|(op, m)| m(op))
    }
}

/// A boxed predicate over an [`OperationIr`]. The matchers returned from
/// [`matchers`] have this type.
pub type OpMatcher = Box<dyn Fn(&OperationIr) -> bool + Send + Sync>;

/// A guard installed via [`FusionInspector::install`]. Holds the shared buffers and
/// clears the global sink on drop.
#[must_use = "the inspector is uninstalled as soon as this guard is dropped"]
pub struct FusionInspector {
    buffer: Arc<Mutex<Vec<FusionReport>>>,
    /// The full set of [`TensorId`]s alive in the
    /// [`HandleContainer`](burn_ir::HandleContainer) at the most recent quiescent
    /// point. Used to assert stream-isolated leak detection: a TensorId that appears
    /// here but wasn't in the baseline (see [`FusionInspector::set_baseline`]) was
    /// born during this test and still hasn't been freed.
    live_handles: Arc<Mutex<Option<HashSet<TensorId>>>>,
    /// Snapshot of `live_handles` captured by [`FusionInspector::set_baseline`]. Any
    /// handle here is assumed to belong to unrelated work (other running tests,
    /// pre-existing state) and excluded from the leak set.
    baseline_handles: Arc<Mutex<Option<HashSet<TensorId>>>>,
    /// Track which stream this inspector belongs to.
    stream_id: StreamId,
}

struct Sink {
    buffer: Arc<Mutex<Vec<FusionReport>>>,
    live_handles: Arc<Mutex<Option<HashSet<TensorId>>>>,
}

/// Maps each observed [`StreamId`] to its active [`Sink`].
fn registry() -> &'static Mutex<HashMap<StreamId, Sink>> {
    static REGISTRY: OnceLock<Mutex<HashMap<StreamId, Sink>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

impl FusionInspector {
    /// Install an inspector for `stream_id`.
    ///
    /// # Panics
    /// Panics if an inspector for the same stream is already active. Two inspectors
    /// for *different* streams may coexist freely.
    pub fn install(stream_id: StreamId) -> Self {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let live_handles = Arc::new(Mutex::new(None));
        let baseline_handles = Arc::new(Mutex::new(None));

        {
            let mut reg = registry().lock().unwrap_or_else(|p| p.into_inner());
            assert!(
                !reg.contains_key(&stream_id),
                "FusionInspector: a sink for stream {stream_id:?} is already installed. Drop the previous inspector before installing a new one.",
            );
            reg.insert(
                stream_id,
                Sink {
                    buffer: buffer.clone(),
                    live_handles: live_handles.clone(),
                },
            );
        }

        Self {
            stream_id,
            buffer,
            live_handles,
            baseline_handles,
        }
    }

    /// Take all captured reports, clearing the buffer.
    pub fn drain(&self) -> Vec<FusionReport> {
        let mut buf = self.buffer.lock().unwrap_or_else(|p| p.into_inner());
        core::mem::take(&mut *buf)
    }

    /// Peek at currently captured reports without clearing.
    pub fn reports(&self) -> Vec<FusionReport> {
        self.buffer
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// The most recent [`HandleContainer`](burn_ir::HandleContainer) size observed by
    /// the inspector, or `None` if no snapshot has been taken yet.
    ///
    /// Snapshots are emitted by the fusion runtime at the same quiescent points as
    /// `memory-checks` (after every registered op and after every drain), so calling
    /// this immediately after a `Backend::sync` reflects the post-drain state.
    pub fn last_handle_count(&self) -> Option<usize> {
        self.live_handles
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|s| s.len())
    }

    /// Snapshot of the [`TensorId`]s currently in the
    /// [`HandleContainer`](burn_ir::HandleContainer), or `None` if nothing has been
    /// observed yet. Mirrors [`Self::last_handle_count`] but returns the full ID set
    /// so callers can compute stream-isolated deltas.
    pub fn last_live_handles(&self) -> Option<HashSet<TensorId>> {
        self.live_handles
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Capture the most recent live-handle snapshot as a baseline. Handles present
    /// here are excluded from [`Self::new_handles_since_baseline`].
    ///
    /// Call this *after* `Backend::sync` so the fusion server has had a chance to
    /// emit a snapshot reflecting the pre-test quiescent state. Tests that use this
    /// method should be marked `#[serial]` to prevent handles from concurrent tests
    /// from polluting the baseline window.
    pub fn set_baseline(&self) {
        let current = self
            .live_handles
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .unwrap_or_default();
        *self
            .baseline_handles
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(current);
    }

    /// The set of [`TensorId`]s present in the most recent snapshot that were *not*
    /// in the baseline captured by [`Self::set_baseline`] — i.e., handles born
    /// during this test's window that are still alive.
    ///
    /// If no baseline was set, every live handle is considered "new". An empty set
    /// after the test syncs means no handles leaked.
    pub fn new_handles_since_baseline(&self) -> HashSet<TensorId> {
        let live = self
            .live_handles
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .unwrap_or_default();
        let baseline = self
            .baseline_handles
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .unwrap_or_default();
        live.difference(&baseline).copied().collect()
    }
}

impl Drop for FusionInspector {
    fn drop(&mut self) {
        if let Ok(mut reg) = registry().lock() {
            reg.remove(&self.stream_id);
        }
    }
}

/// Whether an inspector is currently installed. Called from the fusion server thread.
pub(crate) fn is_installed() -> bool {
    registry().lock().map(|r| !r.is_empty()).unwrap_or(false)
}

/// Push a report into the sink registered for `stream_id`, if any.
/// Called from the fusion server thread.
pub(crate) fn emit(stream_id: StreamId, blocks: &[FusionBlock]) {
    let reg = match registry().lock() {
        Ok(r) => r,
        Err(_) => return,
    };
    let Some(sink) = reg.get(&stream_id) else {
        return;
    };
    let report = FusionReport {
        blocks: blocks.to_vec(),
    };
    if let Ok(mut buf) = sink.buffer.lock() {
        buf.push(report);
    }
}

/// Record the live [`TensorId`]s for `stream_id` at a quiescent point.
pub(crate) fn emit_handle_snapshot(stream_id: StreamId, ids: impl IntoIterator<Item = TensorId>) {
    let reg = match registry().lock() {
        Ok(r) => r,
        Err(_) => return,
    };
    let Some(sink) = reg.get(&stream_id) else {
        return;
    };
    let set: HashSet<TensorId> = ids.into_iter().collect();
    if let Ok(mut slot) = sink.live_handles.lock() {
        *slot = Some(set);
    }
}

/// Small library of matchers for common operations, to keep tests readable.
pub mod matchers {
    use super::OpMatcher;
    use burn_backend::DType;
    use burn_ir::{FloatOperationIr, NumericOperationIr, OperationIr};

    /// Matches a float add (`a + b`) on the given dtype.
    pub fn is_add_float(dtype: DType) -> OpMatcher {
        Box::new(move |op| {
            matches!(
                op,
                OperationIr::NumericFloat(d, NumericOperationIr::Add(_)) if *d == dtype
            )
        })
    }

    /// Matches a float multiply (`a * b`) on the given dtype.
    pub fn is_mul_float(dtype: DType) -> OpMatcher {
        Box::new(move |op| {
            matches!(
                op,
                OperationIr::NumericFloat(d, NumericOperationIr::Mul(_)) if *d == dtype
            )
        })
    }

    /// Matches a float subtraction (`a - b`) on the given dtype.
    pub fn is_sub_float(dtype: DType) -> OpMatcher {
        Box::new(move |op| {
            matches!(
                op,
                OperationIr::NumericFloat(d, NumericOperationIr::Sub(_)) if *d == dtype
            )
        })
    }

    /// Matches a `sum_dim` reduction on float tensors of the given dtype.
    pub fn is_sum_dim_float(dtype: DType) -> OpMatcher {
        Box::new(move |op| {
            matches!(
                op,
                OperationIr::NumericFloat(d, NumericOperationIr::SumDim(_)) if *d == dtype
            )
        })
    }

    /// Matches `Exp` on the given float dtype.
    pub fn is_exp(dtype: DType) -> OpMatcher {
        Box::new(move |op| {
            matches!(
                op,
                OperationIr::Float(d, FloatOperationIr::Exp(_)) if *d == dtype
            )
        })
    }

    /// Matches `Log` on the given float dtype.
    pub fn is_log(dtype: DType) -> OpMatcher {
        Box::new(move |op| {
            matches!(
                op,
                OperationIr::Float(d, FloatOperationIr::Log(_)) if *d == dtype
            )
        })
    }

    /// Matches the memory-bookkeeping `Drop` op emitted when a tensor is deallocated.
    /// These aren't really operations — useful to filter out when counting ops in
    /// fusion-shape tests.
    pub fn is_drop() -> OpMatcher {
        Box::new(|op| matches!(op, OperationIr::Drop(_)))
    }
}
