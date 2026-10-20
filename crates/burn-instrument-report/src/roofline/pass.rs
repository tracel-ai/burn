//! The roofline applied to a pass: what each component is held to, and how those floors add
//! up the tree.

use std::collections::HashSet;
use std::time::Duration;

use burn_instrument::{Bounds, Breakdown, Component, ComponentPath, FusedElsewhere, Peaks};

use super::{Floor, Score};
use crate::reading::TimeReading;

/// The roofline of one pass: every component held against its device's peaks — its floor and
/// its score — and the floors of the ladder's two rungs below it.
pub(crate) struct Roofline<'a> {
    /// The pass's peaks, with any impossible reading dropped.
    peaks: Peaks,
    /// What the work no floor prices stands at.
    reading: TimeReading,
    /// The components whose work all ran in a fused kernel recorded elsewhere, and so have
    /// their floors set aside. See [`fused_elsewhere`](Self::fused_elsewhere).
    fused_elsewhere: FusedElsewhere<'a>,
}

impl<'a> Roofline<'a> {
    pub(crate) fn new(breakdown: &'a Breakdown, peaks: Peaks, reading: TimeReading) -> Self {
        Self {
            peaks,
            reading,
            fused_elsewhere: breakdown.fused_elsewhere(),
        }
    }

    /// The peaks the pass is scored against, with any impossible reading dropped.
    pub(crate) fn peaks(&self) -> &Peaks {
        &self.peaks
    }

    /// The components whose work all ran in a fused kernel recorded elsewhere — on a
    /// component above them, or outside every component.
    ///
    /// Such a component ran nothing, so it has no time to score and its declared
    /// floor is set aside: the work it declared is in the kernel that carried
    /// it out, which stands at what it took on the component that holds it. Counting
    /// the floor as well would price that work twice.
    pub(crate) fn fused_elsewhere(&self) -> &FusedElsewhere<'a> {
        &self.fused_elsewhere
    }

    /// This component's floor and the wall that sets it, when it declared bounds and the
    /// device was probed — whatever it measured. See [`Floor`].
    pub(crate) fn floor_of(&self, component: &Component) -> Option<Floor> {
        Floor::new(self.priced(component)?, &self.peaks)
    }

    /// Score one component against its binding wall, over its apportioned share of
    /// the clean pass.
    ///
    /// `None` where nothing under the component was timed: a percentage over
    /// zero measured time is not a percentage.
    pub(crate) fn score_of(&self, component: &Component) -> Option<Score> {
        let measured = self.reading.apportioned(component);
        (!measured.is_zero())
            .then(|| Score::new(self.priced(component)?, measured, &self.peaks))
            .flatten()
    }

    /// Whether anything under `component` is priced — whether, that is, its own
    /// floor has parts to be read against.
    ///
    /// The question [`deepest_floor`](Self::deepest_floor) descends on and the
    /// worklist ranks on, asked once so the two cannot disagree about which
    /// components are composites.
    pub(crate) fn composite(&self, component: &Component) -> bool {
        component
            .children
            .iter()
            .any(|child| child.bounds.is_some() || self.composite(child))
    }

    /// The floor of the finest-grained declarations under `component` — every
    /// component at its own roof, taken as deep as the tree prices one.
    ///
    /// Falls back to the component's own floor where nothing below declares one, and
    /// to its measured time where nothing does at all: a component nobody
    /// priced cannot be claimed back, so it counts at what it cost.
    pub(crate) fn deepest_floor(&self, component: &Component) -> Duration {
        if self.composite(component) {
            return self.parts_floor(component, Self::deepest_floor);
        }
        self.own_floor(component)
            .unwrap_or(self.reading.apportioned(component))
    }

    /// The floor of the outermost priced components under `component` — each at
    /// its own floor, which assumes perfect fusion inside it.
    ///
    /// Descends through a component that declares none, so an unpriced root
    /// — the model itself — does not hide the priced components under it and
    /// leave the fusion rung equal to the kernel one. What such a component
    /// launched itself stands at what the device spent on it, as in
    /// [`deepest_floor`](Self::deepest_floor).
    pub(crate) fn outermost_floor(&self, component: &Component) -> Duration {
        if let Some(floor) = self.own_floor(component) {
            return floor;
        }
        match self.composite(component) {
            true => self.parts_floor(component, Self::outermost_floor),
            false => self.reading.apportioned(component),
        }
    }

    /// A composite's floor read through its parts: each child at `child_floor`, and what
    /// `component` launched itself that no part prices.
    ///
    /// Whatever the children did not account for is this component's own work, and no
    /// declaration prices it — so it stands, at what the device spent on it.
    fn parts_floor(
        &self,
        component: &Component,
        child_floor: fn(&Self, &Component) -> Duration,
    ) -> Duration {
        component
            .children
            .iter()
            .map(|child| child_floor(self, child))
            .sum::<Duration>()
            + self.reading.apportion(self.unpriced_kernel_time(component))
    }

    /// This component's own floor. See [`floor_of`](Self::floor_of).
    fn own_floor(&self, component: &Component) -> Option<Duration> {
        self.floor_of(component).map(|floor| floor.time)
    }

    /// What `component` is held to, unless its work all ran elsewhere. See
    /// [`fused_elsewhere`](Self::fused_elsewhere).
    fn priced(&self, component: &Component) -> Option<Bounds> {
        match self.fused_elsewhere.contains(component) {
            true => None,
            false => Self::held(component),
        }
    }

    /// What `component` is held to: its declared bounds plus the traffic observed
    /// beyond them, with at least the arithmetic of its parts.
    ///
    /// Fusion removes traffic, never arithmetic, so a composite that prices
    /// only the traffic it must move — the whole model does, rather than
    /// restate every operator's formula — still has every multiply-add beneath
    /// it to do. Everything a component is scored against goes through here, so its
    /// floor and its percentage cannot be of two different bounds.
    fn held(component: &Component) -> Option<Bounds> {
        let own = component.bounds? + component.observed.unwrap_or_default();
        let flops = component
            .parts_bounds()
            .map_or(own.flops, |parts| own.flops.max(parts.flops));
        Some(Bounds { flops, ..own })
    }

    /// Device time in `component`'s own kernels that no declaration beneath it
    /// prices — what a rung counts at measured time because nothing else
    /// counts it.
    ///
    /// A kernel that fused components' work is recorded on their closest
    /// common parent, and where every component it names is still priced —
    /// it ran kernels of its own besides — the rung already counts that work
    /// at their floors: counting the kernel's time as well would price the
    /// same work twice and lift the rung above the pass. Where any of them is
    /// not, the kernel stands at what it took, like the rest of this
    /// component's own work. A component whose work all ran in it is not:
    /// its floor is set aside, so the kernel is what counts its work. And a
    /// kernel that also carried this component's own work stands at what it
    /// took whatever it names, since no floor below prices that part of it.
    fn unpriced_kernel_time(&self, component: &Component) -> Duration {
        let priced: HashSet<ComponentPath> = component
            .descendants()
            .into_iter()
            .filter(|(_, below)| self.priced(below).is_some())
            .map(|(path, _)| path)
            .collect();
        component
            .kernels
            .iter()
            .filter(|kernel| {
                kernel.fuses.is_empty()
                    || kernel.fuses_own
                    || !kernel.fuses.iter().all(|path| priced.contains(path))
            })
            .map(|kernel| kernel.total)
            .sum()
    }
}
