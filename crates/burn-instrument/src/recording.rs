//! The test double for the [`Collector`] seam: one recording of a pass, read as the exact
//! sequence of entries and exits.

use std::sync::Mutex;

use crate::{Bounds, Collecting, Collector, Declaration, ObservedBounds, collect};

/// One entry, as it was declared.
struct Entry {
    label: String,
    variant: Option<String>,
    bounds: Option<Bounds>,
}

/// What a pass did at the seam, in order. Exits are kept as well as entries, because "children
/// close before their parent" is a property of the sequence and not of the rows it produced.
enum Event {
    Enter(Entry),
    Exit,
    /// Traffic observed under the open component, priced on arrival.
    Observed(Bounds),
}

/// A recording of one pass.
#[derive(Default)]
pub(crate) struct Entered {
    events: Mutex<Vec<Event>>,
}

impl Collector for &'static Entered {
    fn enter(&self, declared: Declaration<'_>) {
        self.events.lock().unwrap().push(Event::Enter(Entry {
            label: declared.label.to_string(),
            variant: declared.variant.map(str::to_string),
            bounds: declared.bounds,
        }));
    }

    fn exit(&self) {
        self.events.lock().unwrap().push(Event::Exit);
    }

    fn observe(&self, observed: ObservedBounds) {
        let priced = observed.resolve();
        self.events.lock().unwrap().push(Event::Observed(priced));
    }
}

impl Entered {
    /// Every entry as `+label:variant:bytes`, every exit as `-`, and observed traffic as
    /// `?bytes`.
    pub(crate) fn events(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|event| match event {
                Event::Exit => "-".to_string(),
                Event::Observed(bounds) => format!("?{}", bounds.bytes()),
                Event::Enter(entry) => format!(
                    "+{}:{}:{}",
                    entry.label,
                    entry.variant.as_deref().unwrap_or("-"),
                    entry
                        .bounds
                        .map_or("-".to_string(), |bounds| bounds.bytes().to_string())
                ),
            })
            .collect()
    }
}

/// A recorder installed on this thread, and the guard holding it open.
pub(crate) fn recording() -> (&'static Entered, Collecting) {
    // Leaked: the tests read the recorder after the run, so it outlives every borrow by
    // construction.
    let recorder: &'static Entered = Box::leak(Box::new(Entered::default()));
    (recorder, collect(Box::new(recorder)))
}
