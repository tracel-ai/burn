//! The collector as the runtimes' observer: every kernel launch, and every
//! operation and block the fusion server runs.
//!
//! The runtime calls a [`LaunchObserver`] synchronously, on the thread that
//! issued the launch, before it is submitted. The collector is the observer,
//! and records each launch under that thread's own attribution — moved in step with the ops the thread runs, and inside a
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
//! The collector watches the fusion server too: every operation it registers,
//! and every block of them it runs. It maps
//! each operation to the component that registered it and, while a block
//! runs, records its launches under the closest component containing all of
//! them — see [`TreeCollector::block_starts`](super::tree::TreeCollector::block_starts). A
//! `Drop` is left out: it is registered by whoever let the tensor go, which
//! says nothing about the work, and fused into a block it would widen it to a
//! parent that did none of it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use burn_fusion::observer::{BlockOperations, FusionObservation, FusionObserver};
use cubecl::logging::{
    Duration, LaunchObservation, LaunchObserver, ProfileDuration, TimingMethod, TimingRequest,
};

use super::tree::TreeCollector;

/// Every launch, and what it took, goes to the collector.
impl LaunchObserver for TreeCollector {
    fn launched(&self, kernel: &'static str) {
        self.record_kernel(kernel);
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
        self.keep(kernel, profile);
    }

    /// Only while the runtime's profiling logger reads measurements too, which
    /// reads each at its launch and hands the observer the duration.
    fn timed(&self, kernel: &'static str, duration: Duration, method: TimingMethod) {
        self.record_duration(kernel, duration, method);
    }
}

/// Every registration and every block the fusion server runs goes to the
/// collector, each operation known by [`operation_key`].
impl FusionObserver for TreeCollector {
    fn registered(&self, operation: &burn_ir::OperationIr) {
        if let Some(key) = operation_key(operation) {
            self.registered(key);
        }
    }

    fn block_starts(&self, operations: BlockOperations<'_>) {
        self.block_starts(operations.iter().filter_map(operation_key));
    }

    fn block_ran(&self) {
        self.block_ran();
    }
}

/// What watching a pass installed: the runtime's launch observer and the
/// fusion server's. Both come out when it drops.
pub(crate) struct Watching {
    _launches: LaunchObservation,
    _fusion: FusionObservation,
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

impl Watching {
    /// Hand every launch, and every operation and block the fusion server
    /// runs, to `collector` for as long as the returned guard lives.
    ///
    /// The guards are the runtimes' own: each holds one process-wide slot,
    /// replaces whatever was installed there for its lifetime, and puts it
    /// back on drop.
    ///
    /// # Panics
    ///
    /// If another pass is being watched: the slots are process-wide, so a
    /// second collection would record the first's kernels and the first would
    /// lose its own.
    pub(crate) fn new(collector: &Arc<TreeCollector>) -> Self {
        // Before the observers, so a second pass is refused before it
        // replaces the first's.
        let claim = Claim::take();
        Self {
            _launches: LaunchObservation::new(collector.clone()),
            _fusion: FusionObservation::new(collector.clone()),
            _claim: claim,
        }
    }
}

/// What an operation is known by: its first output, fresh for every operation
/// the server registers. `None` for a `Drop` — see the module docs — and for
/// the few operations that write no tensor.
fn operation_key(operation: &burn_ir::OperationIr) -> Option<u64> {
    match operation {
        burn_ir::OperationIr::Drop(_) => None,
        operation => operation.outputs().next().map(|output| output.id.value()),
    }
}
