#[cfg(feature = "instrument")]
use alloc::boxed::Box;

use crate::Bounds;

/// What a measured run does with the components it enters.
///
/// Implemented by whoever is measuring; the seam knows nothing about reports, devices or
/// kernels. The component side of what a runtime's launch observer is to its kernels.
///
/// A callback must not install or end a `ComponentObservation` on its thread, nor call an
/// annotated method: the thread's observer slot is borrowed while it runs.
// Neither `Send` nor `Sync`: an observer lives in the slot of the thread that installed it and is
// never shared, so requiring either would force interior mutability on implementors that do not
// need it.
pub trait ComponentObserver {
    /// A component was entered, at the path formed by every tag still open.
    fn enter(&self, declared: Declaration<'_>);
    /// The most recently entered component was left.
    fn exit(&self);
    /// Whether the components entered evaluate their bounds for this observer. A bounds function
    /// may read values back, so an observer of a run it times says no, taking the floors from a
    /// pricing run of its own.
    fn prices(&self) -> bool {
        true
    }
}

/// An observer shared with whoever reads it once the run is over, installed as the observer
/// itself.
#[cfg(feature = "std")]
impl<O: ComponentObserver + ?Sized> ComponentObserver for std::sync::Arc<O> {
    fn enter(&self, declared: Declaration<'_>) {
        (**self).enter(declared);
    }
    fn exit(&self) {
        (**self).exit();
    }
    fn prices(&self) -> bool {
        (**self).prices()
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

#[cfg(feature = "instrument")]
std::thread_local! {
    /// Whether this thread is observed, kept apart from the observer so the unobserved path is
    /// one read rather than a borrow.
    ///
    /// Per thread, not process-wide: an observer belongs to the thread that installed it, and a
    /// global flag would let one thread arm the annotations of another holding no observer, or
    /// let a run ending on one thread stop the observation of a run still going on another.
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
    #[cfg(feature = "instrument")]
    return OBSERVED.with(core::cell::Cell::get).then_some(Observing);
    #[cfg(not(feature = "instrument"))]
    return None;
}

/// A scoped change to component observation on **this thread**.
///
/// Installs an observer or suspends observation until the guard drops, restoring the previous
/// observer even if the scope unwinds. Other threads' component observation is unaffected.
/// Without the `instrument` feature, the guard is inert and observers cannot be installed.
///
/// Neither `Send` nor `Sync`: it restores the slot of the thread that made it, and dropped on
/// another thread it would stop observing one that holds no observer.
#[must_use = "the previous observation is restored as soon as this guard is dropped"]
pub struct ComponentObservation {
    /// The observer this one replaced, put back on drop.
    #[cfg(feature = "instrument")]
    replaced: Option<Box<dyn ComponentObserver>>,
    _thread: core::marker::PhantomData<*const ()>,
}

#[cfg(feature = "instrument")]
impl ComponentObservation {
    /// Install `observer` for this thread.
    ///
    /// An observation started inside another replaces it until its guard drops, and the outer
    /// observer then sees the components again: each guard puts back what it replaced, so guards
    /// drop in the reverse order they were made, as scopes drop them.
    pub fn new(observer: Box<dyn ComponentObserver>) -> Self {
        Self::replace(Some(observer))
    }

    /// `None` suspends observation until the guard restores it, including on unwind.
    fn replace(observer: Option<Box<dyn ComponentObserver>>) -> Self {
        let enabled = observer.is_some();
        let replaced = OBSERVER.with(|slot| slot.replace(observer));
        OBSERVED.with(|observed| observed.set(enabled));
        Self {
            replaced,
            _thread: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "instrument")]
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
        Tag {
            _thread: core::marker::PhantomData,
        }
    }

    /// Whether the component being entered should evaluate its bounds. See
    /// [`ComponentObserver::prices`].
    #[inline]
    pub fn prices(self) -> bool {
        let mut prices = false;
        Self::with_observer(|observer| prices = observer.prices());
        prices
    }

    /// Suspend observation while a bounds expression runs, so its annotated helpers
    /// do not become entries of the model. Dropping the guard restores observation.
    #[doc(hidden)]
    #[inline]
    pub fn suspend(self) -> ComponentObservation {
        #[cfg(feature = "instrument")]
        return ComponentObservation::replace(None);
        #[cfg(not(feature = "instrument"))]
        ComponentObservation {
            _thread: core::marker::PhantomData,
        }
    }

    #[cfg_attr(not(feature = "instrument"), allow(unused_variables))]
    fn with_observer(call: impl FnOnce(&dyn ComponentObserver)) {
        #[cfg(feature = "instrument")]
        OBSERVER.with(|slot| {
            if let Some(observer) = slot.borrow().as_ref() {
                call(observer.as_ref());
            }
        });
    }
}

/// An open component tag. Closes on drop.
///
/// Neither `Send` nor `Sync`, like the guard that installs an observer: it closes the component on the
/// observer of the thread that drops it, which has to be the thread that opened it.
#[must_use = "a tag closes as soon as it is dropped, so it has to outlive the body"]
pub struct Tag {
    _thread: core::marker::PhantomData<*const ()>,
}

impl Drop for Tag {
    fn drop(&mut self) {
        Observing::with_observer(|observer| observer.exit());
    }
}

#[cfg(all(test, feature = "instrument"))]
mod tests {
    use super::*;
    use crate::observed::installed;

    /// A thread no observer was installed on is not observed, so annotated methods take the one-read
    /// path.
    #[test]
    fn nothing_is_observed_until_an_observer_is_installed() {
        assert!(observing().is_none());
    }

    /// A failed bounds calculation must not disable observation for subsequent components.
    #[test]
    fn observation_is_restored_after_bounds_panic() {
        let (seen, _observation) = installed();
        let failed = std::panic::catch_unwind(|| {
            let _suspended = observing().unwrap().suspend();
            panic!("bounds failed");
        });
        assert!(failed.is_err());

        drop(
            observing()
                .expect("observation restored")
                .enter(Declaration {
                    label: "after panic",
                    variant: None,
                    site: "test::forward",
                    bounds: None,
                }),
        );
        assert_eq!(seen.events(), ["+after panic:-:-", "-"]);
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

    /// Dropping the observation stops observing the thread, so a run that panicked cannot leave it
    /// observed by an observer nobody holds.
    #[test]
    fn dropping_the_guard_stops_observing() {
        {
            let (_seen, _observation) = installed();
            assert!(observing().is_some());
        }
        assert!(
            observing().is_none(),
            "a run that panicked must not leave the thread observed"
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
