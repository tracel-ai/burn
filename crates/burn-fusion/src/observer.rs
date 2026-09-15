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
//! halves, on the server's thread: [`registered`](FusionObserver::registered)
//! for every operation as it is registered, before anything it triggers runs,
//! and [`block_starts`](FusionObserver::block_starts) /
//! [`block_ran`](FusionObserver::block_ran) around every block, with the
//! operations it covers. Every kernel a block launches is issued on that
//! thread between the two, so an observer that pairs each operation with its
//! own state at registration can say, for any launch, which operations it
//! carried out.
//!
//! # Cost
//!
//! One relaxed atomic load per registration and per block when nothing is
//! installed, which is every ordinary run. An observed block hands over the
//! operations where the server already holds them, so nothing is gathered for
//! it.

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

    /// `block` is about to run. Every kernel it launches is issued on this
    /// thread before [`block_ran`](Self::block_ran).
    fn block_starts(&self, block: BlockOperations<'_>);

    /// The block that last started has finished — run, skipped because an
    /// input carried a failure, or failed itself.
    fn block_ran(&self);
}

/// Watches the fusion server for as long as it lives.
///
/// Process-wide, like the server it watches: operations reach it from every
/// device and every thread that records them. Several observations can be
/// live at once — each observer sees every notification while it is installed,
/// and each guard removes only its own, whatever order they drop in. A block's
/// start and end reach the same observers: one installed while a block runs
/// first hears of the next one, and one removed while a block runs still hears
/// it end, after its guard has dropped.
///
/// The server notifies an observer as it *processes* a call, which lags the
/// call that recorded it: an operation recorded just before the guard drops
/// can be processed after, and one recorded before the guard was created can
/// be processed while it lives. To bracket a region of the program, sync the
/// devices it runs on ([`Backend::sync`](burn_backend::Backend::sync)) before
/// creating the guard and again before dropping it.
#[derive(Debug)]
#[must_use = "an observation stops as soon as it is dropped"]
pub struct FusionObservation {
    id: ObservationId,
}

/// The operations one block covers, in the order it runs them: the operations
/// one fused kernel replaces, or the one operation an unfused block runs.
///
/// A view over the server's own queue rather than a list, so an observed block
/// costs no allocation. The order is the block's, which is not registration
/// order.
#[derive(Debug, Clone, Copy)]
pub struct BlockOperations<'a> {
    operations: &'a [OperationIr],
    ordering: &'a [usize],
}

impl FusionObservation {
    /// Installs `observer` until the guard drops.
    pub fn new(observer: Arc<dyn FusionObserver>) -> Self {
        Self {
            id: OBSERVERS.install(observer),
        }
    }
}

impl Drop for FusionObservation {
    fn drop(&mut self) {
        OBSERVERS.remove(self.id);
    }
}

impl<'a> BlockOperations<'a> {
    /// The operations of `operations` at the indices in `ordering`, which the
    /// caller has checked are in range.
    pub(crate) fn new(operations: &'a [OperationIr], ordering: &'a [usize]) -> Self {
        Self {
            operations,
            ordering,
        }
    }

    /// The block's operations, in the order it runs them.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'a OperationIr> + 'a {
        let operations = self.operations;
        self.ordering.iter().map(move |index| &operations[*index])
    }
}

/// Tell the installed observers that `operation` was registered.
pub(crate) fn notify_registered(operation: &OperationIr) {
    let snapshot = OBSERVERS.snapshot();
    for installed in snapshot.as_deref().unwrap_or_default() {
        installed.observer.registered(operation);
    }
}

/// Runs `run`, the block covering `block`, between telling the installed
/// observers it starts and that it ran — the same observers both times.
pub(crate) fn observe_block<T>(block: BlockOperations<'_>, run: impl FnOnce() -> T) -> T {
    let snapshot = OBSERVERS.snapshot();
    let observers = snapshot.as_deref().unwrap_or_default();
    for installed in observers {
        installed.observer.block_starts(block);
    }
    let output = run();
    for installed in observers {
        installed.observer.block_ran();
    }
    output
}

/// Tells one guard's observer apart from the others, whatever they point to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObservationId(u64);

/// An observer, and the guard that removes it.
#[derive(Clone)]
struct InstalledObserver {
    id: ObservationId,
    observer: Arc<dyn FusionObserver>,
}

/// Every installed observer, behind the flag an unobserved run reads instead.
///
/// One type because the flag and the list must agree: every edit sets both
/// under the write lock, so no install or removal is lost and the flag matches
/// the list whatever order they race in.
struct Observers {
    observing: AtomicBool,
    /// In install order; `None` rather than empty.
    installed: RwLock<Option<Arc<[InstalledObserver]>>>,
    next_id: AtomicU64,
}

impl Observers {
    const fn new() -> Self {
        Self {
            observing: AtomicBool::new(false),
            installed: RwLock::new(None),
            next_id: AtomicU64::new(0),
        }
    }

    fn install(&self, observer: Arc<dyn FusionObserver>) -> ObservationId {
        let id = ObservationId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.edit(|observers| observers.push(InstalledObserver { id, observer }));
        id
    }

    fn remove(&self, id: ObservationId) {
        self.edit(|observers| observers.retain(|installed| installed.id != id));
    }

    /// The installed observers, cloned out so no call into them holds the
    /// lock — or `None` on the one relaxed load an unobserved run pays.
    fn snapshot(&self) -> Option<Arc<[InstalledObserver]>> {
        if !self.observing.load(Ordering::Relaxed) {
            return None;
        }
        self.installed
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Replaces the installed observers with a `change`d copy.
    ///
    /// The list it replaces drops once the lock is released: it can hold the
    /// last reference to a removed observer, and every notification on the
    /// server's thread would wait out that observer's `Drop` under the lock.
    fn edit(&self, change: impl FnOnce(&mut Vec<InstalledObserver>)) {
        let replaced = {
            // Recovering a lock a panicking `change` poisoned: the list is
            // only replaced once the copy is complete.
            let mut installed = self
                .installed
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut observers = installed.as_deref().map(<[_]>::to_vec).unwrap_or_default();
            change(&mut observers);
            self.observing
                .store(!observers.is_empty(), Ordering::Relaxed);
            std::mem::replace(
                &mut *installed,
                (!observers.is_empty()).then(|| observers.into()),
            )
        };
        drop(replaced);
    }
}

static OBSERVERS: Observers = Observers::new();

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
    use burn_std::{DType, Shape};
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use std::thread::ThreadId;

    /// An observer sees registrations and blocks in the order they happen,
    /// and only while it is installed.
    ///
    /// Attribution rests on the order: an observer pairs each block's
    /// operations with what it recorded at their registration, and one that
    /// kept hearing after its guard dropped would attribute work outside the
    /// region it was bracketing.
    #[test]
    fn an_observer_sees_registrations_and_blocks_while_installed() {
        let _serial = serial();
        let recorder = Recorder::new();
        let queue = [drop_of(1), drop_of(2)];
        {
            let _watching = FusionObservation::new(recorder.clone());
            notify_registered(&queue[0]);
            notify_registered(&queue[1]);
            observe_block(BlockOperations::new(&queue, &[1, 0]), || ());
        }
        notify_registered(&queue[0]);
        observe_block(BlockOperations::new(&queue, &[0]), || ());

        assert_eq!(recorder.events(), ["+1", "+2", "[2,1", "]"]);
    }

    /// A block's end reaches the observers its start did, whatever is
    /// installed or removed while it runs.
    ///
    /// An observer tracking open blocks would otherwise pop a block it never
    /// saw start, or keep one open forever — and blocks on other streams and
    /// devices run whenever a guard is created or dropped, however carefully
    /// the caller syncs its own.
    #[test]
    fn a_block_ends_for_the_observers_it_started_for() {
        let _serial = serial();
        let (before, during) = (Recorder::new(), Recorder::new());
        let queue = [drop_of(1)];

        let before_guard = FusionObservation::new(before.clone());
        let during_guard = observe_block(BlockOperations::new(&queue, &[0]), || {
            let during_guard = FusionObservation::new(during.clone());
            drop(before_guard);
            during_guard
        });
        drop(during_guard);

        assert_eq!(before.events(), ["[1", "]"]);
        assert!(during.events().is_empty());
    }

    /// Every installed observer is notified, and a guard removes only its own,
    /// even when an older guard drops before a newer one.
    ///
    /// Guards are process-wide and can drop on any thread, so nothing orders
    /// them: restoring what a guard replaced would silence a live observer
    /// and reinstall a dropped one for good.
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
        assert!(OBSERVERS.snapshot().is_none());
        assert!(!OBSERVERS.observing.load(Ordering::Relaxed));
    }

    /// A removed observer whose last reference is in the list is dropped once
    /// the lock is released.
    ///
    /// Its `Drop` is the observer's own teardown — writing out what it
    /// recorded, say — and under the lock, every notification on the server's
    /// thread would wait it out.
    #[test]
    fn a_removed_observer_drops_outside_the_lock() {
        struct CheckLockOnDrop(Arc<OnceLock<bool>>);

        impl FusionObserver for CheckLockOnDrop {
            fn registered(&self, _operation: &OperationIr) {}
            fn block_starts(&self, _block: BlockOperations<'_>) {}
            fn block_ran(&self) {}
        }

        impl Drop for CheckLockOnDrop {
            fn drop(&mut self) {
                self.0.set(OBSERVERS.installed.try_read().is_ok()).unwrap();
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
    ///
    /// A flag cleared over a live observer silences it with no error, for as
    /// long as its guard lives.
    #[test]
    fn racing_installs_and_drops_keep_every_observer_installed_while_its_guard_lives() {
        let _serial = serial();
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..200 {
                        let guard = FusionObservation::new(Arc::new(Silent));
                        let list = OBSERVERS
                            .snapshot()
                            .expect("installed while its guard lives");
                        assert!(list.iter().any(|installed| installed.id == guard.id));
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        assert!(OBSERVERS.snapshot().is_none());
        assert!(!OBSERVERS.observing.load(Ordering::Relaxed));
    }

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
        fn block_starts(&self, block: BlockOperations<'_>) {
            self.record(|| {
                let ids: Vec<String> = block.iter().map(id).collect();
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
        fn block_starts(&self, _block: BlockOperations<'_>) {}
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
}
