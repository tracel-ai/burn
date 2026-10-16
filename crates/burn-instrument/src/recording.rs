//! The test double for the [`Collector`] seam: one recording of a pass, read as the exact
//! sequence of entries and exits.

use std::sync::Mutex;

use crate::{Bounds, Collecting, Collector, Declaration, ObservedBounds, collect};

/// One entry, as it was declared.
struct Entry {
    depth: usize,
    label: String,
    variant: Option<String>,
    site: &'static str,
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
    depth: Mutex<usize>,
}

impl Collector for &'static Entered {
    fn enter(&self, declared: Declaration<'_>) {
        let mut depth = self.depth.lock().unwrap();
        self.events.lock().unwrap().push(Event::Enter(Entry {
            depth: *depth,
            label: declared.label.to_string(),
            variant: declared.variant.map(str::to_string),
            site: declared.site,
            bounds: declared.bounds,
        }));
        *depth += 1;
    }

    fn exit(&self) {
        self.events.lock().unwrap().push(Event::Exit);
        *self.depth.lock().unwrap() -= 1;
    }

    fn observe(&self, observed: ObservedBounds) {
        let priced = observed.resolve();
        self.events.lock().unwrap().push(Event::Observed(priced));
    }
}

impl Entered {
    /// The tree as indented rows, each marked `(unpriced)` where the component declared no
    /// floor.
    pub(crate) fn rows(&self) -> Vec<String> {
        self.with_entries(|entries| {
            entries
                .map(|entry| {
                    let priced = match entry.bounds.is_some() {
                        true => "",
                        false => " (unpriced)",
                    };
                    format!("{}{}{priced}", "  ".repeat(entry.depth), entry.label)
                })
                .collect()
        })
    }

    /// The distinct call sites that entered `label`, in first-seen order.
    pub(crate) fn sites(&self, label: &str) -> Vec<&'static str> {
        self.with_entries(|entries| {
            let mut sites: Vec<&'static str> = Vec::new();
            for entry in entries.filter(|entry| entry.label == label) {
                if !sites.contains(&entry.site) {
                    sites.push(entry.site);
                }
            }
            sites
        })
    }

    fn with_entries<T>(&self, read: impl FnOnce(&mut dyn Iterator<Item = &Entry>) -> T) -> T {
        let events = self.events.lock().unwrap();
        read(&mut events.iter().filter_map(|event| match event {
            Event::Enter(entry) => Some(entry),
            Event::Exit | Event::Observed(_) => None,
        }))
    }

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

/// Record what `run` enters, with the run closed before the recording is read.
pub(crate) fn entered(run: impl FnOnce()) -> &'static Entered {
    let (recorder, collecting) = recording();
    run();
    drop(collecting);
    recorder
}

/// A recorder installed on this thread, and the guard holding it open.
pub(crate) fn recording() -> (&'static Entered, Collecting) {
    // Leaked: the tests read the recorder after the run, so it outlives every borrow by
    // construction.
    let recorder: &'static Entered = Box::leak(Box::new(Entered::default()));
    (recorder, collect(Box::new(recorder)))
}
