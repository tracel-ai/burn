//! Watching the fusion server register operations and run blocks of them.
//!
//! Fusion decouples *when* an operation is recorded from *when* its work runs:
//! an operation is registered as the program reaches it and executed later,
//! inside a block — one fused kernel's worth of operations, or one operation
//! run unfused — once something seals the segment it sits in. A block can run
//! long after its operations were registered, out of registration order, and
//! it can hold operations the program registered in quite different places. A
//! caller attributing device work to what the program was doing cannot read
//! that off the kernel launches alone.
//!
//! A [`FusionObserver`] sees both halves, on the server's thread:
//! [`registered`](FusionObserver::registered) for every operation as it is
//! registered, before anything it triggers runs, and
//! [`block_starts`](FusionObserver::block_starts) /
//! [`block_ran`](FusionObserver::block_ran) around every block, with the
//! operations it covers. Every kernel a block launches is issued on that
//! thread between the two, so an observer that pairs each operation with its
//! own state at registration can say, for any launch, which operations it
//! carried out.
//!
//! # Cost
//!
//! One relaxed atomic load per registration and per block when nothing is
//! installed, which is every ordinary run. The operations of a block are only
//! gathered for an installed observer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use burn_ir::OperationIr;

/// Notified of every operation the fusion server registers and every block of
/// them it runs, on the server's thread.
///
/// Implementations must be cheap and must not register operations or install
/// or drop a [`FusionObservation`]: they run inside the server, which is
/// processing the call that reached them.
pub trait FusionObserver: Send + Sync {
    /// `operation` was registered, in registration order, before anything its
    /// registration triggers runs.
    fn registered(&self, operation: &OperationIr);

    /// A block is about to run: the operations one fused kernel replaces, or
    /// the one operation an unfused block runs. Every kernel it launches is
    /// issued on this thread before [`block_ran`](Self::block_ran).
    ///
    /// The order is the block's, which is not registration order.
    fn block_starts(&self, operations: &[&OperationIr]);

    /// The block that last started has finished — run, skipped because an
    /// input carried a failure, or failed itself.
    fn block_ran(&self);
}

/// Watches the fusion server for as long as it lives, then puts back whatever
/// it replaced.
///
/// Process-wide, like the server it watches: operations reach it from every
/// thread that records them. A guard rather than an install/stop pair, so the
/// scope observed is the scope the guard lives for.
#[must_use = "an observation stops as soon as it is dropped"]
pub struct FusionObservation {
    previous: Option<Arc<dyn FusionObserver>>,
}

impl FusionObservation {
    /// Installs `observer` until the guard drops.
    pub fn new(observer: Arc<dyn FusionObserver>) -> Self {
        let previous = slot().replace(observer);
        // Last, so the flag is never set over an empty slot.
        OBSERVING.store(true, Ordering::Relaxed);
        Self { previous }
    }
}

impl Drop for FusionObservation {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let still_observed = previous.is_some();
        *slot() = previous;
        // Last, so the flag is never cleared while an observer is installed.
        OBSERVING.store(still_observed, Ordering::Relaxed);
    }
}

/// Tell the installed observer, if there is one, that `operation` was
/// registered.
pub(crate) fn notify_registered(operation: &OperationIr) {
    if let Some(observer) = installed() {
        observer.registered(operation);
    }
}

/// Tell the installed observer, if there is one, that a block covering
/// `operations` is about to run. The operations are gathered only for an
/// installed observer.
pub(crate) fn notify_block_starts<'a>(operations: impl FnOnce() -> Vec<&'a OperationIr>) {
    if let Some(observer) = installed() {
        observer.block_starts(&operations());
    }
}

/// Tell the installed observer, if there is one, that the block that last
/// started has finished.
pub(crate) fn notify_block_ran() {
    if let Some(observer) = installed() {
        observer.block_ran();
    }
}

/// The installed observer, cloned out of the slot so no call into it holds
/// the lock — or `None` on the one relaxed load an unobserved run pays.
fn installed() -> Option<Arc<dyn FusionObserver>> {
    if !OBSERVING.load(Ordering::Relaxed) {
        return None;
    }
    OBSERVER
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// The slot, for writing, recovering a lock a panicking observer poisoned.
fn slot() -> std::sync::RwLockWriteGuard<'static, Option<Arc<dyn FusionObserver>>> {
    OBSERVER
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether anything is watching — the unobserved path's one relaxed load.
static OBSERVING: AtomicBool = AtomicBool::new(false);

static OBSERVER: RwLock<Option<Arc<dyn FusionObserver>>> = RwLock::new(None);

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
    use burn_std::{DType, Shape};
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recorder(Mutex<Vec<String>>);

    impl FusionObserver for Recorder {
        fn registered(&self, operation: &OperationIr) {
            self.0.lock().unwrap().push(format!("+{}", id(operation)));
        }
        fn block_starts(&self, operations: &[&OperationIr]) {
            let ids: Vec<String> = operations.iter().map(|op| id(op)).collect();
            self.0.lock().unwrap().push(format!("[{}", ids.join(",")));
        }
        fn block_ran(&self) {
            self.0.lock().unwrap().push("]".to_string());
        }
    }

    fn id(operation: &OperationIr) -> String {
        match operation {
            OperationIr::Drop(tensor) => tensor.id.value().to_string(),
            _ => "?".to_string(),
        }
    }

    fn drop_of(value: u64) -> OperationIr {
        OperationIr::Drop(TensorIr {
            id: TensorId::new(value),
            shape: Shape::new([1]),
            status: TensorStatus::ReadWrite,
            dtype: DType::F32,
        })
    }

    /// An installed observer sees registrations and blocks in the order they
    /// happen, gathered only while it is installed; the guard puts back the
    /// empty slot it found.
    #[test]
    fn an_observer_sees_registrations_and_blocks_while_installed() {
        let recorder = Arc::new(Recorder::default());
        let (one, two) = (drop_of(1), drop_of(2));
        {
            let _watching = FusionObservation::new(recorder.clone());
            notify_registered(&one);
            notify_registered(&two);
            notify_block_starts(|| vec![&two, &one]);
            notify_block_ran();
        }
        notify_registered(&one);
        notify_block_starts(|| panic!("no observer, so nothing is gathered"));

        assert_eq!(*recorder.0.lock().unwrap(), ["+1", "+2", "[2,1", "]"]);
    }
}
