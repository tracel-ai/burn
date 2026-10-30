//! The test double for the [`ComponentObserver`] seam: one observation of a run, read as the exact
//! sequence of entries and exits.

use std::sync::Mutex;

use crate::{Bounds, ComponentObservation, ComponentObserver, Declaration};

/// One entry, as it was declared.
struct Entry {
    depth: usize,
    label: String,
    variant: Option<String>,
    site: &'static str,
    bounds: Option<Bounds>,
}

/// What a run did at the seam, in order. Exits are kept as well as entries, because "children
/// close before their parent" is a property of the sequence and not of the components it produced.
enum Event {
    Enter(Entry),
    Exit,
}

/// What one run entered.
#[derive(Default)]
pub(crate) struct Entered {
    events: Mutex<Vec<Event>>,
    depth: Mutex<usize>,
}

impl ComponentObserver for &'static Entered {
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
}

impl Entered {
    /// The tree, one indented line per component, each marked `(unpriced)` where the component declared no
    /// floor.
    pub(crate) fn lines(&self) -> Vec<String> {
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
            Event::Exit => None,
        }))
    }

    /// Every entry as `+label:variant:bytes`, and every exit as `-`.
    pub(crate) fn events(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .map(|event| match event {
                Event::Exit => "-".to_string(),
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

/// Observe what `run` enters, with the observation closed before it is read.
pub(crate) fn entered(run: impl FnOnce()) -> &'static Entered {
    let (seen, observation) = installed();
    run();
    drop(observation);
    seen
}

/// A fresh observer installed on this thread, and the guard holding it open.
pub(crate) fn installed() -> (&'static Entered, ComponentObservation) {
    // Leaked: the tests read it after the run, so it outlives every borrow by
    // construction.
    let seen: &'static Entered = Box::leak(Box::new(Entered::default()));
    (seen, ComponentObservation::new(Box::new(seen)))
}
