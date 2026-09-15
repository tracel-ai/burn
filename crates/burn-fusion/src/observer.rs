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
//! A [`FusionObserver`], installed with a [`FusionObservation`], sees both
//! halves, on the server's thread: [`registered`] for every operation as it is
//! registered, before anything it triggers runs, and [`block_starts`] /
//! [`block_ran`] around every block, with the operations it covers. Every
//! kernel a block launches is issued on that thread between the two, so an
//! observer that pairs each operation with its own state at registration can
//! say, for any launch, which operations it carried out.
//!
//! # Cost
//!
//! One relaxed atomic load per registration and per block when nothing is
//! installed, which is every ordinary run. The operations of a block are only
//! gathered for an installed observer.
//!
//! [`FusionObserver`]: crate::observer::FusionObserver
//! [`FusionObservation`]: crate::observer::FusionObservation
//! [`registered`]: crate::observer::FusionObserver::registered
//! [`block_starts`]: crate::observer::FusionObserver::block_starts
//! [`block_ran`]: crate::observer::FusionObserver::block_ran

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use burn_ir::OperationIr;

/// Notified of every operation the fusion server registers and every block of
/// them it runs, on the server's thread.
///
/// Implementations must be cheap and must not register operations or install
/// or drop a [`FusionObservation`]: they run inside the server, which is
/// processing the call that reached them. The same goes for their `Drop`, which
/// runs on the server's thread when a notification in flight holds the last
/// reference.
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

/// Watches the fusion server for as long as it lives.
///
/// Process-wide, like the server it watches: operations reach it from every
/// device and every thread that records them. Several observations can be
/// live at once — each observer sees every notification while it is installed,
/// and each guard removes only its own, whatever order they drop in.
///
/// The server notifies an observer as it *processes* a call, which lags the
/// call that recorded it: an operation recorded just before the guard drops
/// can be processed after, and one recorded before the guard was created can
/// be processed while it lives. To bracket a region of the program, sync the
/// devices it runs on (`Backend::sync`) before creating the guard and again
/// before dropping it.
#[must_use = "an observation stops as soon as it is dropped"]
pub struct FusionObservation {
    id: u64,
}

impl FusionObservation {
    /// Installs `observer` until the guard drops.
    pub fn new(observer: Arc<dyn FusionObserver>) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        edit_installed(|observers| observers.push(Installed { id, observer }));
        Self { id }
    }
}

impl Drop for FusionObservation {
    fn drop(&mut self) {
        edit_installed(|observers| observers.retain(|installed| installed.id != self.id));
    }
}

/// An installed observer, and the id of the guard that removes it.
#[derive(Clone)]
struct Installed {
    id: u64,
    observer: Arc<dyn FusionObserver>,
}

/// Replaces the installed observers with a `change`d copy.
///
/// The copy is changed and swapped in, and the flag set to match it, all under
/// the write lock, so no install or drop is lost and the flag matches the list
/// whatever order they race in. The list it replaces drops once the lock is
/// released: it can hold the last reference to a removed observer, whose
/// `Drop` must not run under the lock.
fn edit_installed(change: impl FnOnce(&mut Vec<Installed>)) {
    let replaced = {
        // Recovering a lock a panicking `change` poisoned: the slot is only
        // written once the copy is complete.
        let mut slot = OBSERVERS
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut observers = slot.as_deref().map(<[_]>::to_vec).unwrap_or_default();
        change(&mut observers);
        OBSERVING.store(!observers.is_empty(), Ordering::Relaxed);
        std::mem::replace(
            &mut *slot,
            (!observers.is_empty()).then(|| observers.into()),
        )
    };
    drop(replaced);
}

/// Tell the installed observers that `operation` was registered.
pub(crate) fn notify_registered(operation: &OperationIr) {
    for installed in installed().iter().flat_map(|list| list.iter()) {
        installed.observer.registered(operation);
    }
}

/// Tell the installed observers that a block covering `operations` is about
/// to run. The operations are gathered only when something is installed.
pub(crate) fn notify_block_starts<'a>(operations: impl FnOnce() -> Vec<&'a OperationIr>) {
    if let Some(list) = installed() {
        let operations = operations();
        for installed in list.iter() {
            installed.observer.block_starts(&operations);
        }
    }
}

/// Tell the installed observers that the block that last started has
/// finished.
pub(crate) fn notify_block_ran() {
    for installed in installed().iter().flat_map(|list| list.iter()) {
        installed.observer.block_ran();
    }
}

/// The installed observers, cloned out of the slot so no call into them
/// holds the lock — or `None` on the one relaxed load an unobserved run pays.
fn installed() -> Option<Arc<[Installed]>> {
    if !OBSERVING.load(Ordering::Relaxed) {
        return None;
    }
    OBSERVERS
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Whether anything is watching — the unobserved path's one relaxed load.
static OBSERVING: AtomicBool = AtomicBool::new(false);

/// The installed observers, in install order; `None` rather than empty.
static OBSERVERS: RwLock<Option<Arc<[Installed]>>> = RwLock::new(None);

/// The id of the next guard.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
    use burn_std::{DType, Shape};
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::thread::ThreadId;

    /// Records what reaches it from the thread that made it, and nothing else:
    /// other tests in this binary register operations on their own threads
    /// while it is installed.
    struct Recorder {
        thread: ThreadId,
        events: Mutex<Vec<String>>,
    }

    impl Recorder {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                thread: std::thread::current().id(),
                events: Mutex::default(),
            })
        }

        fn record(&self, event: impl FnOnce() -> String) {
            if std::thread::current().id() == self.thread {
                self.events.lock().unwrap().push(event());
            }
        }

        fn events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    impl FusionObserver for Recorder {
        fn registered(&self, operation: &OperationIr) {
            self.record(|| format!("+{}", id(operation)));
        }
        fn block_starts(&self, operations: &[&OperationIr]) {
            self.record(|| {
                let ids: Vec<String> = operations.iter().map(|op| id(op)).collect();
                format!("[{}", ids.join(","))
            });
        }
        fn block_ran(&self) {
            self.record(|| "]".to_string());
        }
    }

    struct Silent;

    impl FusionObserver for Silent {
        fn registered(&self, _operation: &OperationIr) {}
        fn block_starts(&self, _operations: &[&OperationIr]) {}
        fn block_ran(&self) {}
    }

    /// Serializes the tests that install observers: what is installed is
    /// process-wide, and each checks it.
    fn serial() -> MutexGuard<'static, ()> {
        static SERIAL: Mutex<()> = Mutex::new(());
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    /// happen, gathered only while it is installed.
    #[test]
    fn an_observer_sees_registrations_and_blocks_while_installed() {
        let _serial = serial();
        let recorder = Recorder::new();
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

        assert_eq!(recorder.events(), ["+1", "+2", "[2,1", "]"]);
    }

    /// Every installed observer is notified, and a guard removes only its own,
    /// even when an older guard drops before a newer one.
    #[test]
    fn guards_remove_only_their_own_observer_whatever_order_they_drop_in() {
        let _serial = serial();
        let (first, second) = (Recorder::new(), Recorder::new());

        let first_guard = FusionObservation::new(first.clone());
        let second_guard = FusionObservation::new(second.clone());
        notify_registered(&drop_of(1));
        drop(first_guard);
        notify_registered(&drop_of(2));
        drop(second_guard);
        notify_registered(&drop_of(3));

        assert_eq!(first.events(), ["+1"]);
        assert_eq!(second.events(), ["+1", "+2"]);
        assert!(installed().is_none());
        assert!(!OBSERVING.load(Ordering::Relaxed));
    }

    /// A removed observer holding its last reference in the slot is dropped
    /// once the lock is released, so its `Drop` can reach the slot.
    #[test]
    fn a_removed_observer_drops_outside_the_lock() {
        struct CheckLockOnDrop(Arc<OnceLock<bool>>);

        impl FusionObserver for CheckLockOnDrop {
            fn registered(&self, _operation: &OperationIr) {}
            fn block_starts(&self, _operations: &[&OperationIr]) {}
            fn block_ran(&self) {}
        }

        impl Drop for CheckLockOnDrop {
            fn drop(&mut self) {
                self.0.set(OBSERVERS.try_read().is_ok()).unwrap();
            }
        }

        let _serial = serial();
        let lock_was_free = Arc::new(OnceLock::new());
        drop(FusionObservation::new(Arc::new(CheckLockOnDrop(
            lock_was_free.clone(),
        ))));

        // Another test's notification can briefly hold the last reference, in
        // which case the observer drops on that thread once it is done.
        let lock_was_free = loop {
            if let Some(free) = lock_was_free.get() {
                break *free;
            }
            std::thread::yield_now();
        };
        assert!(lock_was_free);
    }

    /// Installs and drops racing from many threads neither lose an observer
    /// nor leave the flag disagreeing with what is installed.
    #[test]
    fn racing_installs_and_drops_keep_every_observer_installed_while_its_guard_lives() {
        let _serial = serial();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..200 {
                        let guard = FusionObservation::new(Arc::new(Silent));
                        let list = installed().expect("installed while its guard lives");
                        assert!(list.iter().any(|installed| installed.id == guard.id));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        assert!(installed().is_none());
        assert!(!OBSERVING.load(Ordering::Relaxed));
    }
}
