use alloc::boxed::Box;
use alloc::vec::Vec;

use burn_tensor::{Int, Tensor};

use crate::Bounds;

/// What a measured run does with the components a pass enters.
///
/// Implemented by whoever is measuring; the seam knows nothing about reports, devices or
/// kernels.
// Neither `Send` nor `Sync`: a collector lives in the slot of the thread that installed it and is
// never shared, so requiring either would force interior mutability on implementors that do not
// need it.
pub trait Collector {
    /// A component was entered, at the path formed by every tag still open.
    fn enter(&self, declared: Declaration<'_>);
    /// The most recently entered component was left.
    fn exit(&self);
    /// The component most recently entered moves more traffic than its declared floor, by an
    /// amount only the pass's values tell. A run that reads the pass back adds it to that
    /// component and the ones around it; any other run ignores it.
    fn observe(&self, _observed: ObservedBounds) {}
}

/// Traffic a component moves beyond its declared floor that depends on values the pass
/// computes, such as which experts a sparse layer's routes select.
///
/// [`Bounds`] are pure, so a declared floor charges only what the shapes force. This keeps the
/// values on the device with their price, so a run that reads the pass back prices them
/// afterwards and synchronizes nothing during it.
pub struct ObservedBounds(Box<dyn FnOnce() -> Bounds + Send>);

impl ObservedBounds {
    /// `values`, priced by `price` once read back.
    pub fn new<const D: usize>(
        values: Tensor<D, Int>,
        price: impl FnOnce(&[i64]) -> Bounds + Send + 'static,
    ) -> Self {
        Self(Box::new(move || {
            let values: Vec<i64> = values.into_data().iter::<i64>().collect();
            price(&values)
        }))
    }

    /// Read the values back and price them. Waits on the device, so a run calls it after the
    /// pass is timed.
    ///
    /// # Panics
    ///
    /// If the work producing the values failed on the device.
    pub fn resolve(self) -> Bounds {
        (self.0)()
    }
}

/// What one entry of a component declares about itself.
///
/// A struct rather than a parameter list: [`label`](Self::label) and [`site`](Self::site) are
/// both strings, and a call that swapped them would compile.
#[derive(Clone, Copy, Debug)]
pub struct Declaration<'a> {
    /// What the component is called, which is the row this entry lands on.
    ///
    /// Rows merge on this alone among siblings, so a stack of identical layers is one row
    /// entered once per layer rather than a subtree per layer.
    pub label: &'a str,
    /// Which of several executions served this entry, where that is chosen at call time.
    ///
    /// Not part of the row's identity: a component that routed two ways is one row with a
    /// histogram of variants.
    pub variant: Option<&'a str>,
    /// Where this entry is written: the module and the function, as
    /// `my_model::attention::forward`.
    ///
    /// Not part of the row's identity either. Two entry points merged under one label is what a
    /// reader wants for a stack, and also how a row comes to hold two different jobs, so a
    /// report can name a row's sites without undoing the merge. Two types in one module with a
    /// method of the same name share a site.
    pub site: &'static str,
    /// The floor for running this entry alone, where the component declares one. `None` is a
    /// component that is timed and not scored.
    pub bounds: Option<Bounds>,
}

#[cfg(feature = "std")]
std::thread_local! {
    /// Whether this thread is measuring, kept apart from the collector so the disarmed path is one
    /// read rather than a borrow.
    ///
    /// Per thread, not process-wide: a collector belongs to the thread that installed it, and a
    /// global flag would let one thread arm the annotations of another holding no collector, or
    /// let a run ending on one thread disarm a run still going on another.
    static ARMED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    static COLLECTOR: core::cell::RefCell<Option<Box<dyn Collector>>> =
        const { core::cell::RefCell::new(None) };
}

/// The run measuring this thread, or `None` when nothing is.
///
/// Called on every entry of an annotated method, so the unmeasured path is a single
/// thread-local read.
#[inline]
pub fn armed() -> Option<Run> {
    #[cfg(feature = "std")]
    return ARMED.with(core::cell::Cell::get).then_some(Run);
    #[cfg(not(feature = "std"))]
    return None;
}

/// Install `collector` for **this thread** and arm its annotated methods.
///
/// A forward pass runs on one thread, so a run installs its collector there and nothing else in
/// the process is measured or slowed. The collector is removed when the returned guard drops, so
/// a panicking pass cannot leave the thread armed against a collector nobody holds.
///
/// A run started inside another replaces it until its guard drops, and the outer run then
/// measures again: each guard puts back what it replaced, so guards drop in the reverse order
/// they were made, as scopes drop them.
#[cfg(feature = "std")]
pub fn collect(collector: Box<dyn Collector>) -> Collecting {
    let replaced = COLLECTOR.with(|slot| slot.replace(Some(collector)));
    ARMED.with(|armed| armed.set(true));
    Collecting {
        replaced,
        _thread: core::marker::PhantomData,
    }
}

/// Holds a measured run open. See [`collect`].
///
/// Neither `Send` nor `Sync`: it restores the slot of the thread that made it, and dropped on
/// another thread it would disarm one that holds no run.
#[cfg(feature = "std")]
#[must_use = "the collector is uninstalled as soon as this is dropped"]
pub struct Collecting {
    /// The run this one replaced, put back on drop.
    replaced: Option<Box<dyn Collector>>,
    _thread: core::marker::PhantomData<*const ()>,
}

#[cfg(feature = "std")]
impl Drop for Collecting {
    fn drop(&mut self) {
        let replaced = self.replaced.take();
        ARMED.with(|armed| armed.set(replaced.is_some()));
        // Dropped outside the slot's borrow, so a collector whose drop touches the slot does not
        // find it borrowed.
        let ended = COLLECTOR.with(|slot| slot.replace(replaced));
        drop(ended);
    }
}

/// A measured run, handed to an annotated method by [`armed`].
#[derive(Clone, Copy, Debug)]
pub struct Run;

impl Run {
    /// Open a tag for the component being entered. It closes when the returned guard drops, so
    /// an early return cannot leave one open.
    #[inline]
    pub fn enter(self, declared: Declaration<'_>) -> Tag {
        Self::with_collector(|collector| collector.enter(declared));
        Tag
    }

    /// Report traffic beyond the declared floor of the component most recently entered. See
    /// [`Collector::observe`].
    pub fn observe(self, observed: ObservedBounds) {
        Self::with_collector(|collector| collector.observe(observed));
    }

    #[cfg_attr(not(feature = "std"), allow(unused_variables))]
    fn with_collector(call: impl FnOnce(&dyn Collector)) {
        #[cfg(feature = "std")]
        COLLECTOR.with(|slot| {
            if let Some(collector) = slot.borrow().as_ref() {
                call(collector.as_ref());
            }
        });
    }
}

/// An open component tag. Closes on drop.
#[must_use = "a tag closes as soon as it is dropped, so it has to outlive the body"]
pub struct Tag;

impl Drop for Tag {
    fn drop(&mut self) {
        Run::with_collector(|collector| collector.exit());
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::recording::recording;

    #[test]
    fn nothing_is_collected_until_a_run_installs_a_collector() {
        assert!(armed().is_none());
    }

    #[test]
    fn a_tag_closes_when_its_guard_drops() {
        let (recorder, _collecting) = recording();

        let run = armed().expect("armed by `collect`");
        {
            let _outer = run.enter(Declaration {
                label: "attention",
                variant: Some("flash"),
                site: "test::attend",
                bounds: Some(Bounds {
                    read: 6,
                    write: 2,
                    flops: 2,
                }),
            });
            let _inner = run.enter(Declaration {
                label: "qkv",
                variant: None,
                site: "test::project",
                bounds: None,
            });
        }

        assert_eq!(
            recorder.events(),
            ["+attention:flash:8", "+qkv:-:-", "-", "-"],
            "children close before their parent"
        );
    }

    #[test]
    fn observed_traffic_reaches_the_collector_under_the_open_component() {
        let (recorder, _collecting) = recording();
        let run = armed().expect("armed by `collect`");
        {
            let _experts = run.enter(Declaration {
                label: "experts",
                variant: None,
                site: "test::route",
                bounds: None,
            });
            let named = Tensor::<1, Int>::from_ints([2, 5, 2], &burn_tensor::Device::flex());
            run.observe(ObservedBounds::new(named, |named| Bounds {
                read: named.iter().sum::<i64>() as u64,
                write: 0,
                flops: 0,
            }));
        }

        assert_eq!(recorder.events(), ["+experts:-:-", "?9", "-"]);
    }

    #[test]
    fn dropping_the_guard_disarms() {
        {
            let (_recorder, _collecting) = recording();
            assert!(armed().is_some());
        }
        assert!(
            armed().is_none(),
            "a pass that panicked must not leave the thread armed"
        );
    }

    #[test]
    fn a_nested_run_hands_the_thread_back_to_the_outer_one() {
        let declared = |label| Declaration {
            label,
            variant: None,
            site: "test::forward",
            bounds: None,
        };
        let (outer, _collecting) = recording();
        let block = armed()
            .expect("armed by `collect`")
            .enter(declared("block"));
        {
            let (inner, _collecting) = recording();
            drop(
                armed()
                    .expect("armed by `collect`")
                    .enter(declared("probe")),
            );
            assert_eq!(inner.events(), ["+probe:-:-", "-"]);
        }
        drop(block);

        assert_eq!(
            outer.events(),
            ["+block:-:-", "-"],
            "the outer run measures again once the inner one ends, so its open tag closes on it"
        );
        assert!(armed().is_some(), "and the thread is still armed for it");
    }

    #[test]
    fn a_run_on_another_thread_neither_arms_nor_disarms_this_one() {
        let (mine, _collecting) = recording();

        std::thread::spawn(|| {
            assert!(armed().is_none(), "another thread's run is not this one's");
            let (_theirs, _collecting) = recording();
        })
        .join()
        .expect("the other thread ran");

        let run = armed().expect("a run that ended elsewhere must not disarm this thread");
        drop(run.enter(Declaration {
            label: "attention",
            variant: None,
            site: "test::attend",
            bounds: None,
        }));
        assert_eq!(mine.events(), ["+attention:-:-", "-"]);
    }
}
