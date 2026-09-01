//! Drops that cannot be registered where they are raised.
//!
//! Registering a drop re-enters the client, which can drain the stream and run
//! queued work. A thread that does that while it is unwinding raises a second
//! panic inside the first, which aborts the process — so a drop raised during
//! a panic is set aside here instead, and replayed by the next registration on
//! that thread, which is a normal call stack with nothing unwinding through it.
//!
//! Setting it aside rather than abandoning it is what keeps the tensor's entry
//! in the handle container releasable: an abandoned drop leaves the entry, and
//! any claim on it, outliving every tensor that could report it.

use core::cell::{Cell, RefCell};
use std::collections::VecDeque;

thread_local! {
    static PENDING: RefCell<VecDeque<Box<dyn FnOnce()>>> =
        const { RefCell::new(VecDeque::new()) };
    static REPLAYING: Cell<bool> = const { Cell::new(false) };
}

/// Set a drop aside until this thread is somewhere it can be registered.
pub(crate) fn defer(drop: impl FnOnce() + 'static) {
    PENDING.with(|pending| pending.borrow_mut().push_back(Box::new(drop)));
}

/// Register everything set aside. Called wherever the client is reached, so the
/// check rides along with work the caller was doing anyway.
pub(crate) fn flush() {
    // The common case, on the hot path: nothing was ever deferred.
    if PENDING.with(|pending| pending.borrow().is_empty()) {
        return;
    }

    let Some(_replaying) = Replaying::enter() else {
        return;
    };

    // One at a time, off the front, rather than draining into a batch:
    // registering a drop can defer another, so the queue is still growing while
    // this runs, and a panic out of one must leave the rest where the next
    // flush will find them rather than carry them off in a batch it owns.
    while let Some(drop) = PENDING.with(|pending| pending.borrow_mut().pop_front()) {
        drop();
    }
}

/// Marks this thread as replaying for as long as it is held.
///
/// The flag says the outer flush owns the queue, so the inner flush that every
/// replayed registration triggers has nothing left to do. It is a guard rather
/// than a pair of assignments because a panic out of a replayed drop would skip
/// the reset: the flag would be set for the life of the thread, every later
/// flush would return early, and the entries this module exists to release
/// would be stranded — silently, and only on the failing path.
struct Replaying;

impl Replaying {
    /// `None` when this thread is already inside a flush.
    fn enter() -> Option<Self> {
        match REPLAYING.replace(true) {
            true => None,
            false => Some(Self),
        }
    }
}

impl Drop for Replaying {
    fn drop(&mut self) {
        REPLAYING.set(false);
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::deferred;
    use std::cell::Cell;
    use std::rc::Rc;

    /// A drop set aside during an unwind still runs: the entry it releases is
    /// the only thing that can release a claim on that tensor.
    #[test]
    fn a_deferred_drop_runs_at_the_next_flush() {
        let ran = Rc::new(Cell::new(0));

        let counter = ran.clone();
        deferred::defer(move || counter.set(counter.get() + 1));
        assert_eq!(ran.get(), 0, "not until something flushes");

        deferred::flush();
        assert_eq!(ran.get(), 1);

        deferred::flush();
        assert_eq!(ran.get(), 1, "and only once");
    }

    /// Registering a deferred drop can defer another — a tensor released by the
    /// first going out of scope. The flush has to reach those too, which is why
    /// it pops from a queue that is still growing rather than iterating a batch.
    #[test]
    fn a_flush_reaches_drops_deferred_by_the_flush() {
        let ran = Rc::new(Cell::new(0));

        let counter = ran.clone();
        deferred::defer(move || {
            counter.set(counter.get() + 1);
            let inner = counter.clone();
            deferred::defer(move || inner.set(inner.get() + 1));
        });

        deferred::flush();
        assert_eq!(ran.get(), 2, "both the deferred drop and the one it caused");
    }

    /// A replayed drop that panics must not wedge the thread. This is the path
    /// the whole module serves — the drops arrive from unwinding threads — so a
    /// flag left set here would strand every later drop on that thread, and the
    /// leak it exists to prevent would come back permanently and unannounced.
    #[test]
    fn a_panicking_replay_leaves_the_thread_able_to_flush_again() {
        let ran = Rc::new(Cell::new(0));

        deferred::defer(|| panic!("registering this one failed"));
        let counter = ran.clone();
        deferred::defer(move || counter.set(counter.get() + 1));

        let escaped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(deferred::flush));
        assert!(escaped.is_err(), "the panic reaches the caller");
        assert_eq!(ran.get(), 0, "and stopped the flush where it was");

        deferred::flush();
        assert_eq!(
            ran.get(),
            1,
            "the drop behind it is still queued, and the next flush reaches it"
        );
    }
}
