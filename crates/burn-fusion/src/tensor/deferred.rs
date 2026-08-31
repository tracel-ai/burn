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

use core::cell::RefCell;

thread_local! {
    static PENDING: RefCell<Vec<Box<dyn FnOnce()>>> = const { RefCell::new(Vec::new()) };
}

/// Set a drop aside until this thread is somewhere it can be registered.
pub(crate) fn defer(drop: impl FnOnce() + 'static) {
    PENDING.with(|pending| pending.borrow_mut().push(Box::new(drop)));
}

/// Register everything set aside. Called where a registration already
/// happens, so the check rides along with work the caller was doing anyway.
pub(crate) fn flush() {
    // The common case, on the hot path: nothing was ever deferred.
    if PENDING.with(|pending| pending.borrow().is_empty()) {
        return;
    }

    // Taken rather than iterated in place: registering a drop can defer
    // another one, and holding the borrow across that would panic.
    loop {
        let batch: Vec<_> = PENDING.with(|pending| core::mem::take(&mut *pending.borrow_mut()));

        if batch.is_empty() {
            return;
        }

        for drop in batch {
            drop();
        }
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
    /// it takes the queue rather than iterating it in place.
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
}
