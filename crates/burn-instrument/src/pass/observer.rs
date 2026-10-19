//! Forwarding the compute runtime's kernel launches into the component tree.
//!
//! The runtime calls a [`LaunchObserver`] synchronously, on the thread that
//! issued the launch, before it is submitted. The observer holds the collector
//! and hands each launch straight to it, which records it under that thread's
//! own attribution — moved in step with the ops the thread runs, and inside a
//! fusion block set to the components whose work the block carries out, with
//! nothing drained to make it so → [`LaunchCursor`](super::cursor::LaunchCursor).
//!
//! The runtime's slot is process-wide, and it has to be: a launch does not
//! arrive on the thread that entered the components — burn's fusion processes a
//! stream off the thread that recorded it — so an observer scoped to one thread
//! would see nothing at all. The cost is that a second measured run in one
//! process would write into the first's tree; nothing here starts two.
//!
//! # Measurements are read once the pass is over
//!
//! Each launch's measurement arrives unread and the collector keeps it, reading
//! them all back after the pass's last sync. Reading one waits for its kernel,
//! so reading each as it arrived would run the pass one kernel at a time — the
//! kernels would each be timed alone, and their sum would say nothing about a
//! pass that keeps the device fed.
//!
//! # The fusion server says whose work a block is
//!
//! A second observer watches the fusion server: every
//! operation it registers, and every block of them it runs. The collector maps
//! each operation to the component that registered it and, while a block
//! runs, records its launches under the closest component containing all of
//! them — see [`TreeCollector::block_starts`](super::tree::TreeCollector::block_starts). A
//! `Drop` is left out: it is registered by whoever let the tensor go, which
//! says nothing about the work, and fused into a block it would widen it to a
//! parent that did none of it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cubecl::logging::{
    Duration, LaunchObservation, LaunchObserver, ProfileDuration, TimingMethod, TimingRequest,
};

use super::tree::SharedCollector;

/// Hands every launch, and what it took, to one collector.
struct ToCollector(SharedCollector);

impl LaunchObserver for ToCollector {
    fn launched(&self, kernel: &'static str) {
        self.0.record_kernel(kernel);
    }

    /// Timed, and handed over unread — a breakdown whose kernel rows carried
    /// no time would say which kernels ran and nothing about which of them
    /// mattered, and reading each measurement at its launch waits for that
    /// kernel, which runs the pass one kernel at a time.
    fn timing(&self) -> TimingRequest {
        TimingRequest::Deferred
    }

    /// Kept. The collector reads them all once the pass is over.
    fn profiled(&self, kernel: &'static str, profile: ProfileDuration) {
        self.0.keep(kernel, profile);
    }

    /// Only while the runtime's profiling logger reads measurements too, which
    /// reads each at its launch and hands the observer the duration.
    fn timed(&self, kernel: &'static str, duration: Duration, method: TimingMethod) {
        self.0.record_duration(kernel, duration, method);
    }
}

/// Hands every registration and every block the fusion server runs to one
/// collector.
struct FusionToCollector(SharedCollector);

impl FusionToCollector {
    /// What an operation is known by: its first output, fresh for every
    /// operation the server registers. `None` for a `Drop` — see the module
    /// docs — and for the few operations that write no tensor.
    fn key(operation: &burn_ir::OperationIr) -> Option<u64> {
        match operation {
            burn_ir::OperationIr::Drop(_) => None,
            operation => operation.outputs().next().map(|output| output.id.value()),
        }
    }
}

impl burn_fusion::observer::FusionObserver for FusionToCollector {
    fn registered(&self, operation: &burn_ir::OperationIr) {
        if let Some(key) = Self::key(operation) {
            self.0.registered(key);
        }
    }

    fn block_starts(&self, operations: burn_fusion::observer::BlockOperations<'_>) {
        self.0.block_starts(operations.iter().filter_map(Self::key));
    }

    fn block_ran(&self) {
        self.0.block_ran();
    }
}

/// What watching a pass installed: the runtime's launch observer and the
/// fusion server's. Both come out when it drops.
pub struct Watching {
    _launches: LaunchObservation,
    _fusion: burn_fusion::observer::FusionObservation,
    /// Last, since fields drop in order: the claim is given up only once both
    /// observers above have been put back, so the next pass to claim it
    /// cannot have its own removed by this one's.
    _claim: Claim,
}

/// Whether a pass is being watched anywhere in the process.
static WATCHING: AtomicBool = AtomicBool::new(false);

/// This process's one watched pass. Given up on drop.
struct Claim;

impl Claim {
    /// # Panics
    ///
    /// If another pass holds it.
    fn take() -> Self {
        let already = WATCHING.swap(true, Ordering::AcqRel);
        assert!(
            !already,
            "a pass is already being collected in this process; the runtime's launch observer is \
             process-wide, so collections run one at a time"
        );
        Claim
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        WATCHING.store(false, Ordering::Release);
    }
}

/// Hand every launch, and every operation and block the
/// fusion server runs, to `collector` for as long as the returned guard
/// lives.
///
/// The guards are the runtimes' own: each holds one process-wide slot,
/// replaces whatever was installed there for its lifetime, and puts it back on
/// drop.
///
/// # Panics
///
/// If another pass is being watched: the slots are process-wide, so a second
/// collection would record the first's kernels and the first would lose its
/// own.
pub fn watch_launches(collector: &SharedCollector) -> Watching {
    // Before the observers, so a second pass is refused before it replaces
    // the first's.
    let claim = Claim::take();
    Watching {
        _launches: LaunchObservation::new(Arc::new(ToCollector(collector.handle()))),
        _fusion: burn_fusion::observer::FusionObservation::new(Arc::new(FusionToCollector(
            collector.handle(),
        ))),
        _claim: claim,
    }
}
