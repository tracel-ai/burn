//! The test double for the [`ComponentObserver`] seam: what one run of a module declared, read as
//! the component tree and the floors in it.

use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::Mutex;

use burn_core::instrument::{Bounds, ComponentObservation, ComponentObserver, Declaration};

/// One entry, as it was declared.
struct Entry {
    depth: usize,
    label: String,
    bounds: Option<Bounds>,
}

/// Every entry of one run, in the order entered.
#[derive(Default)]
pub(crate) struct Recorded {
    entries: Mutex<Vec<Entry>>,
    depth: Mutex<usize>,
}

impl ComponentObserver for Recorded {
    fn enter(&self, declared: Declaration<'_>) {
        let mut depth = self.depth.lock().unwrap();
        self.entries.lock().unwrap().push(Entry {
            depth: *depth,
            label: declared.label.to_string(),
            bounds: declared.bounds,
        });
        *depth += 1;
    }

    fn exit(&self) {
        *self.depth.lock().unwrap() -= 1;
    }
}

impl Recorded {
    /// The tree, one indented line per entry, each marked `(unpriced)` where the component
    /// declared no floor.
    pub(crate) fn lines(&self) -> Vec<String> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .map(|entry| {
                let priced = match entry.bounds.is_some() {
                    true => "",
                    false => " (unpriced)",
                };
                alloc::format!("{}{}{priced}", "  ".repeat(entry.depth), entry.label)
            })
            .collect()
    }

    /// The floor the first entry declared: the module the run called.
    pub(crate) fn declared(&self) -> Option<Bounds> {
        self.entries
            .lock()
            .unwrap()
            .first()
            .and_then(|entry| entry.bounds)
    }
}

/// Record what `run` enters, with the observation closed before it is read.
pub(crate) fn recorded(run: impl FnOnce()) -> Arc<Recorded> {
    let recorded = Arc::new(Recorded::default());
    let observation = ComponentObservation::new(alloc::boxed::Box::new(recorded.clone()));
    run();
    drop(observation);
    recorded
}
