//! Test-only observer for fusion decisions.
//!
//! When a [`FusionSpy`] is installed, every execution plan that runs through the fusion
//! server is captured as a [`FusionReport`] describing which [`OperationIr`]s ended up
//! fused into a single kernel and which ran unfused. This lets tests assert on the
//! *shape* of fusion — not just on numeric output.
//!
//! # Usage
//!
//! ```ignore
//! # use burn_fusion::spy::FusionSpy;
//! let spy = FusionSpy::install();
//! // ... run tensor ops with a fusion-wrapped backend, then sync ...
//! let reports = spy.drain();
//! let block = reports[0].assert_single_fused_block();
//! assert_eq!(block.fuser_name(), Some("ElementWise"));
//! ```
//!
//! # Threading
//!
//! The fusion server runs on a background thread per device. Reports are captured on
//! that thread and deposited into a process-global sink, so only one spy can be
//! active at a time. [`FusionSpy::install`] enforces this by blocking on a
//! process-wide mutex — concurrent calls from different tests wait rather than
//! racing. Marking spy tests `#[serial]` is still recommended as documentation, but
//! the mutex is what actually prevents interference.

use burn_ir::OperationIr;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::stream::execution::trace::{Section, SectionKind, format_table};

/// The list of fusion decisions captured for one execution plan.
#[derive(Debug, Clone)]
pub struct FusionReport {
    /// One entry per leaf of the executed strategy, in execution order.
    pub blocks: Vec<FusionBlock>,
}

/// A single contiguous run of operations executed together.
#[derive(Debug, Clone)]
pub struct FusionBlock {
    /// Whether this block ran fused or unfused.
    pub kind: BlockKind,
    /// The operations contained in the block, in execution order.
    pub operations: Vec<OperationIr>,
}

/// Whether a [`FusionBlock`] was fused into a single kernel or not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    /// The operations were fused. `name` is the fuser that produced the kernel,
    /// `score` is its reported score.
    Fused {
        /// The name of the fuser / optimization (e.g. `"ElementWise"`).
        name: &'static str,
        /// The score the fuser reported for this kernel.
        score: u64,
    },
    /// The operations ran individually; no fusion happened for this block.
    Unfused,
}

impl FusionReport {
    /// Render the report as the same human-readable table that fusion logging emits at
    /// [`FusionLogLevel::Full`](burn_std::config::fusion::FusionLogLevel). Useful for
    /// rich panic messages from spy-based tests.
    pub fn format_table(&self) -> String {
        let sections: Vec<Section> = self
            .blocks
            .iter()
            .map(|b| Section {
                kind: match b.kind {
                    BlockKind::Fused { name, score } => SectionKind::Fused { name, score },
                    BlockKind::Unfused => SectionKind::Operation,
                },
                ops: b.operations.clone(),
            })
            .collect();
        format_table(&sections)
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

/// A guard installed via [`FusionSpy::install`]. Holds the shared buffer and clears
/// the global sink on drop.
///
/// Only one spy can be active at a time; [`FusionSpy::install`] blocks on a
/// process-wide mutex to enforce this.
#[must_use = "the spy is uninstalled as soon as this guard is dropped"]
pub struct FusionSpy {
    buffer: Arc<Mutex<Vec<FusionReport>>>,
    // Held for the lifetime of the spy. Dropped *after* `buffer`, releasing the
    // process-wide install lock so the next spy can proceed.
    _install_guard: MutexGuard<'static, ()>,
}

struct Sink {
    buffer: Arc<Mutex<Vec<FusionReport>>>,
}

fn global() -> &'static Mutex<Option<Sink>> {
    static GLOBAL: OnceLock<Mutex<Option<Sink>>> = OnceLock::new();
    GLOBAL.get_or_init(|| Mutex::new(None))
}

/// Serializes `FusionSpy::install` across threads: only one spy can be active at a time.
fn install_mutex() -> &'static Mutex<()> {
    static INSTALL: OnceLock<Mutex<()>> = OnceLock::new();
    INSTALL.get_or_init(|| Mutex::new(()))
}

impl FusionSpy {
    /// Install a spy. Returns a guard that clears the global sink when dropped.
    ///
    /// Blocks if another [`FusionSpy`] is currently installed (whether in this
    /// thread or another) — the guard is released when that prior spy is dropped.
    /// Poisoned locks (e.g. from a previous test panic) are recovered silently.
    pub fn install() -> Self {
        let install_guard = install_mutex()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let buffer = Arc::new(Mutex::new(Vec::new()));
        {
            let mut slot = global().lock().unwrap_or_else(|p| p.into_inner());
            *slot = Some(Sink {
                buffer: buffer.clone(),
            });
        }

        Self {
            buffer,
            _install_guard: install_guard,
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
}

impl Drop for FusionSpy {
    fn drop(&mut self) {
        if let Ok(mut slot) = global().lock() {
            *slot = None;
        }
        // `_install_guard` is released after this, allowing the next install() to proceed.
    }
}

/// Whether a spy is currently installed. Called from the fusion server thread.
pub(crate) fn is_installed() -> bool {
    global().lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Push a report into the installed sink, if any. Called from the fusion server thread.
pub(crate) fn emit(sections: &[Section]) {
    let slot = match global().lock() {
        Ok(s) => s,
        Err(_) => return,
    };
    let Some(sink) = slot.as_ref() else {
        return;
    };
    let report = FusionReport {
        blocks: sections.iter().map(section_to_block).collect(),
    };
    if let Ok(mut buf) = sink.buffer.lock() {
        buf.push(report);
    }
}

fn section_to_block(section: &Section) -> FusionBlock {
    let kind = match section.kind {
        SectionKind::Fused { name, score } => BlockKind::Fused { name, score },
        SectionKind::Operation => BlockKind::Unfused,
    };
    FusionBlock {
        kind,
        operations: section.ops.clone(),
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
