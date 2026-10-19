use alloc::boxed::Box;
use alloc::vec::Vec;

use burn_tensor::{Int, Tensor};

use crate::Bounds;

/// What a measured pass does with the components it enters.
///
/// Implemented by whoever is measuring; the seam knows nothing about reports, devices or
/// kernels. The component side of what a runtime's launch observer is to its kernels.
// Neither `Send` nor `Sync`: an observer lives in the slot of the thread that installed it and is
// never shared, so requiring either would force interior mutability on implementors that do not
// need it.
pub trait ComponentObserver {
    /// A component was entered, at the path formed by every tag still open.
    fn enter(&self, declared: Declaration<'_>);
    /// The most recently entered component was left.
    fn exit(&self);
    /// The component most recently entered moves more traffic than its declared floor, by an
    /// amount only the pass's values tell. An observer that reads the pass back adds it to that
    /// component and the ones around it; any other ignores it.
    fn observe(&self, _observed: ObservedBounds) {}
}

/// An observer shared with whoever reads it once the pass is over, installed as the observer
/// itself.
#[cfg(feature = "std")]
impl<O: ComponentObserver + ?Sized> ComponentObserver for std::sync::Arc<O> {
    fn enter(&self, declared: Declaration<'_>) {
        (**self).enter(declared);
    }
    fn exit(&self) {
        (**self).exit();
    }
    fn observe(&self, observed: ObservedBounds) {
        (**self).observe(observed);
    }
}

/// Traffic a component moves beyond its declared floor that depends on values the pass
/// computes, such as which experts a sparse layer's routes select.
///
/// [`Bounds`] are pure, so a declared floor charges only what the shapes force. This keeps the
/// values on the device with their price, so an observer that reads the pass back prices them
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

    /// Read the values back and price them. Waits on the device, so an observer calls it after
    /// the pass is timed.
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
    /// What the component is called.
    ///
    /// Components merge on this alone among siblings, so a stack of identical layers is one component
    /// entered once per layer rather than a subtree per layer.
    pub label: &'a str,
    /// Which of several executions served this entry, where that is chosen at call time.
    ///
    /// Not part of the component's identity: a component that routed two ways is still one
    /// component, with a histogram of variants.
    pub variant: Option<&'a str>,
    /// Where this entry is written: the module and the function, as
    /// `my_model::attention::forward`.
    ///
    /// Not part of the component's identity either. Two entry points merged under one label is what a
    /// reader wants for a stack, and also how a component comes to hold two different jobs, so a
    /// report can name a component's sites without undoing the merge. Two types in one module with a
    /// method of the same name share a site.
    pub site: &'static str,
    /// The floor for running this entry alone, where the component declares one. `None` is a
    /// component that is timed and not scored.
    pub bounds: Option<Bounds>,
}

#[cfg(feature = "std")]
std::thread_local! {
    /// Whether this thread is observed, kept apart from the observer so the unobserved path is
    /// one read rather than a borrow.
    ///
    /// Per thread, not process-wide: an observer belongs to the thread that installed it, and a
    /// global flag would let one thread arm the annotations of another holding no observer, or
    /// let a pass ending on one thread stop the observation of a pass still going on another.
    static OBSERVED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    static OBSERVER: core::cell::RefCell<Option<Box<dyn ComponentObserver>>> =
        const { core::cell::RefCell::new(None) };
}

/// The observation of this thread, or `None` when nothing observes it.
///
/// Called on every entry of an annotated method, so the unobserved path is a single
/// thread-local read.
#[inline]
pub fn observing() -> Option<Observing> {
    #[cfg(feature = "std")]
    return OBSERVED.with(core::cell::Cell::get).then_some(Observing);
    #[cfg(not(feature = "std"))]
    return None;
}

/// An observer installed for **this thread**, arming its annotated methods until it drops.
///
/// A forward pass runs on one thread, so an observer is installed there and nothing else in the
/// process is measured or slowed. It is removed when this guard drops, so a panicking pass cannot
/// leave the thread observed by an observer nobody holds.
///
/// Neither `Send` nor `Sync`: it restores the slot of the thread that made it, and dropped on
/// another thread it would stop observing one that holds no observer.
#[cfg(feature = "std")]
#[must_use = "the observer is uninstalled as soon as this is dropped"]
pub struct ComponentObservation {
    /// The observer this one replaced, put back on drop.
    replaced: Option<Box<dyn ComponentObserver>>,
    _thread: core::marker::PhantomData<*const ()>,
}

#[cfg(feature = "std")]
impl ComponentObservation {
    /// Install `observer` for this thread.
    ///
    /// An observation started inside another replaces it until its guard drops, and the outer
    /// observer then sees the components again: each guard puts back what it replaced, so guards
    /// drop in the reverse order they were made, as scopes drop them.
    pub fn new(observer: Box<dyn ComponentObserver>) -> Self {
        let replaced = OBSERVER.with(|slot| slot.replace(Some(observer)));
        OBSERVED.with(|observed| observed.set(true));
        Self {
            replaced,
            _thread: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "std")]
impl Drop for ComponentObservation {
    fn drop(&mut self) {
        let replaced = self.replaced.take();
        OBSERVED.with(|observed| observed.set(replaced.is_some()));
        // Dropped outside the slot's borrow, so an observer whose drop touches the slot does not
        // find it borrowed.
        let ended = OBSERVER.with(|slot| slot.replace(replaced));
        drop(ended);
    }
}

/// The observation of this thread, handed to an annotated method by [`observing`].
#[derive(Clone, Copy, Debug)]
pub struct Observing;

impl Observing {
    /// Open a tag for the component being entered. It closes when the returned guard drops, so
    /// an early return cannot leave one open.
    #[inline]
    pub fn enter(self, declared: Declaration<'_>) -> Tag {
        Self::with_observer(|observer| observer.enter(declared));
        Tag
    }

    /// Report traffic beyond the declared floor of the component most recently entered. See
    /// [`ComponentObserver::observe`].
    pub fn observe(self, observed: ObservedBounds) {
        Self::with_observer(|observer| observer.observe(observed));
    }

    #[cfg_attr(not(feature = "std"), allow(unused_variables))]
    fn with_observer(call: impl FnOnce(&dyn ComponentObserver)) {
        #[cfg(feature = "std")]
        OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow().as_ref() {
                call(observer.as_ref());
            }
        });
    }
}

/// An open component tag. Closes on drop.
#[must_use = "a tag closes as soon as it is dropped, so it has to outlive the body"]
pub struct Tag;

impl Drop for Tag {
    fn drop(&mut self) {
        Observing::with_observer(|observer| observer.exit());
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::observed::installed;

    /// A thread no observer was installed on is not observed, so annotated methods take the one-read
    /// path.
    #[test]
    fn nothing_is_observed_until_an_observer_is_installed() {
        assert!(observing().is_none());
    }

    /// A tag closes when its guard drops, so children close before their parent and an early return
    /// cannot leave one open.
    #[test]
    fn a_tag_closes_when_its_guard_drops() {
        let (seen, _observation) = installed();

        let observing = observing().expect("installed above");
        {
            let _outer = observing.enter(Declaration {
                label: "attention",
                variant: Some("flash"),
                site: "test::attend",
                bounds: Some(Bounds {
                    read: 6,
                    write: 2,
                    flops: 2,
                }),
            });
            let _inner = observing.enter(Declaration {
                label: "qkv",
                variant: None,
                site: "test::project",
                bounds: None,
            });
        }

        assert_eq!(
            seen.events(),
            ["+attention:flash:8", "+qkv:-:-", "-", "-"],
            "children close before their parent"
        );
    }

    /// Observed traffic reaches the observer between the entry and exit of the component that
    /// observed it, which is how the observer knows whose it is.
    #[test]
    fn observed_traffic_reaches_the_observer_under_the_open_component() {
        let (seen, _observation) = installed();
        let observing = observing().expect("installed above");
        {
            let _experts = observing.enter(Declaration {
                label: "experts",
                variant: None,
                site: "test::route",
                bounds: None,
            });
            let named = Tensor::<1, Int>::from_ints([2, 5, 2], &burn_tensor::Device::flex());
            observing.observe(ObservedBounds::new(named, |named| Bounds {
                read: named.iter().sum::<i64>() as u64,
                write: 0,
                flops: 0,
            }));
        }

        assert_eq!(seen.events(), ["+experts:-:-", "?9", "-"]);
    }

    /// Dropping the observation stops observing the thread, so a pass that panicked cannot leave it
    /// observed by an observer nobody holds.
    #[test]
    fn dropping_the_guard_stops_observing() {
        {
            let (_seen, _observation) = installed();
            assert!(observing().is_some());
        }
        assert!(
            observing().is_none(),
            "a pass that panicked must not leave the thread observed"
        );
    }

    /// An observation inside another hands the thread back to the outer one when it ends, so the
    /// outer observer still sees the tags it opened close.
    #[test]
    fn a_nested_observation_hands_the_thread_back_to_the_outer_one() {
        let declared = |label| Declaration {
            label,
            variant: None,
            site: "test::forward",
            bounds: None,
        };
        let (outer, _observation) = installed();
        let block = observing()
            .expect("installed above")
            .enter(declared("block"));
        {
            let (inner, _observation) = installed();
            drop(
                observing()
                    .expect("installed above")
                    .enter(declared("probe")),
            );
            assert_eq!(inner.events(), ["+probe:-:-", "-"]);
        }
        drop(block);

        assert_eq!(
            outer.events(),
            ["+block:-:-", "-"],
            "the outer observer sees the components again once the inner one ends, so its open tag closes on it"
        );
        assert!(
            observing().is_some(),
            "and the thread is still observed by it"
        );
    }

    /// An observation belongs to its thread: one starting or ending elsewhere neither observes this
    /// thread nor stops its own observation.
    #[test]
    fn an_observation_on_another_thread_neither_starts_nor_stops_this_one() {
        let (mine, _observation) = installed();

        std::thread::spawn(|| {
            assert!(
                observing().is_none(),
                "another thread's observation is not this one's"
            );
            let (_theirs, _observation) = installed();
        })
        .join()
        .expect("the other thread ran");

        let observing =
            observing().expect("an observation that ended elsewhere must not stop this one");
        drop(observing.enter(Declaration {
            label: "attention",
            variant: None,
            site: "test::attend",
            bounds: None,
        }));
        assert_eq!(mine.events(), ["+attention:-:-", "-"]);
    }
}
