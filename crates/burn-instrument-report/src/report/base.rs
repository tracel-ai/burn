//! The table a collected pass prints.
//!
//! One page: the device and what it can do, how the pass ran, the component
//! tree with each component scored against the wall it is bound by, the ladder, and
//! a worklist ranked by what is actually recoverable. It is plain text,
//! greppable, and says what it does not know rather than leaving a column
//! blank.

use std::time::Duration;

use burn_instrument::{Bounds, CollectedPass, Component, ComponentPath, PeakFinding, Peaks};

use crate::Comparison;
use crate::roofline::{Floor, Ladder, Score};

/// A collected pass, scored against its device's peaks, and the table that
/// renders it ([`Display`](std::fmt::Display)).
///
/// A front door that knows what the pass was (a model, a batch, a phase)
/// prints that above this table; the table says only what the file does.
pub struct Report<'a> {
    pass: &'a CollectedPass,
    /// The pass's peaks, with any impossible reading dropped.
    peaks: Peaks,
    /// What the header must say about the probe set before its numbers are
    /// read, already ordered — see [`Peaks::findings`].
    findings: Vec<PeakFinding>,
    /// The components whose work all ran in a fused kernel recorded elsewhere. See
    /// [`Breakdown::fused_elsewhere`](burn_instrument::Breakdown::fused_elsewhere).
    fused_elsewhere: Vec<&'a Component>,
    /// What a component's milliseconds are, given how its kernels were timed.
    reading: TimeReading,
    /// Whether the pass splits into the device's time and the host's.
    host_split: HostSplit,
}

/// What a component's milliseconds are, given how the pass timed its kernels and whether an
/// uncollected run was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimeReading {
    /// Read back after the pass, so the kernels ran back to back: a component's time is its
    /// kernels' own device time.
    InPlace,
    /// Each read as it launched, so the kernels ran one at a time: a component's time is its
    /// share of the uncollected run, the one unperturbed number there is.
    ApportionedOnto(Duration),
    /// Each read as it launched, and no uncollected run to apportion onto: a component's time
    /// is its kernels' serialized time, and only the proportions mean anything.
    Serialized,
}

impl TimeReading {
    pub(super) fn new(pass: &CollectedPass) -> Self {
        match (pass.breakdown.timed_alone, pass.clean_wall) {
            (false, _) => Self::InPlace,
            (true, Some(wall)) => Self::ApportionedOnto(wall),
            (true, None) => Self::Serialized,
        }
    }
}

/// Whether a pass splits into the device's time and the time the device waited on the host,
/// and why not where it does not: the clean pass less what its kernels took holds only when
/// that device time is a pipelined pass's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostSplit {
    /// The time the device spent waiting on the host.
    Measured(Duration),
    /// No uncollected run was measured to split.
    NoCleanRun,
    /// A launch timed alone serializes the pass, so its total is not a pipelined pass's.
    TimedAlone,
    /// Nothing was timed on the device.
    NothingTimed,
    /// The kernels measured more than the pass took: two clocks disagreeing, not a negative
    /// host.
    ClocksDisagree,
}

impl HostSplit {
    pub(super) fn new(pass: &CollectedPass) -> Self {
        let Some(wall) = pass.clean_wall else {
            return Self::NoCleanRun;
        };
        let device = pass.breakdown.total_time();
        if pass.breakdown.timed_alone {
            Self::TimedAlone
        } else if device.is_zero() {
            Self::NothingTimed
        } else if device > wall {
            Self::ClocksDisagree
        } else {
            Self::Measured(wall - device)
        }
    }
}

impl<'a> Report<'a> {
    /// Score `pass` against the peaks its device measured, or against none
    /// where it was not probed.
    ///
    /// An impossible peak is dropped here rather than divided by — see
    /// [`Peaks::trusted`]. The findings are read off the raw peaks first, so
    /// the header can say what was rejected: a probe that half-ran is a fact
    /// about the run, and silently falling back to a coarser wall would leave a
    /// reader wondering why nothing is held against `flops`, or why every component
    /// moved at once.
    pub fn new(pass: &'a CollectedPass) -> Self {
        let peaks = pass.peaks.unwrap_or_default();
        Self {
            pass,
            findings: peaks.findings(),
            peaks: peaks.trusted(),
            fused_elsewhere: pass.breakdown.fused_elsewhere(),
            reading: TimeReading::new(pass),
            host_split: HostSplit::new(pass),
        }
    }

    /// This pass held against `saved`, component by component.
    pub fn against(self, saved: &'a CollectedPass) -> Comparison<'a> {
        Comparison::new(self, saved)
    }

    /// The pass this report scores.
    pub(crate) fn pass(&self) -> &'a CollectedPass {
        self.pass
    }

    /// The peaks the pass is scored against, with any impossible reading dropped.
    pub(crate) fn peaks(&self) -> &Peaks {
        &self.peaks
    }

    /// What the probe set's readings say about themselves, in the order a report says it.
    pub(crate) fn findings(&self) -> &[PeakFinding] {
        &self.findings
    }

    /// What a component's milliseconds are.
    pub(crate) fn reading(&self) -> TimeReading {
        self.reading
    }

    /// Whether the pass splits into the device's time and the host's.
    pub(crate) fn host_split(&self) -> HostSplit {
        self.host_split
    }

    /// What the pass took: the clean pass's wall time where it was measured,
    /// and the collected pass's device time otherwise.
    pub fn wall(&self) -> Duration {
        self.pass
            .clean_wall
            .unwrap_or_else(|| self.pass.breakdown.total_time())
    }

    /// Of the clean pass, the time the device spent waiting on the host — the
    /// launches, the op recording, everything between kernels — where the
    /// collected pass can say.
    ///
    /// It is the clean pass's wall less the device time its kernels took,
    /// which holds only when that device time is a pipelined pass's: `None`
    /// where no clean pass was measured, where any launch was timed alone,
    /// where nothing was measured on the device at all, and where the kernels
    /// add up to more than the clean pass took — two clocks disagreeing, not a
    /// negative host.
    pub fn framework(&self) -> Option<Duration> {
        match self.host_split {
            HostSplit::Measured(host) => Some(host),
            HostSplit::NoCleanRun
            | HostSplit::TimedAlone
            | HostSplit::NothingTimed
            | HostSplit::ClocksDisagree => None,
        }
    }

    /// Whether all of `component`'s work ran in a fused kernel recorded elsewhere —
    /// on a component above it, or outside every component.
    ///
    /// Such a component ran nothing, so it has no time to score and its declared
    /// floor is set aside: the work it declared is in the kernel that carried
    /// it out, which stands at what it took on the component that holds it. Counting
    /// the floor as well would price that work twice.
    pub(crate) fn is_fused_elsewhere(&self, component: &Component) -> bool {
        // By identity rather than by path: every caller holds a component borrowed
        // from this report, and none of them its path. A tree that kept its
        // collector's component indices would make both unnecessary.
        self.fused_elsewhere
            .iter()
            .any(|fused| std::ptr::eq(*fused, component))
    }

    /// The ladder for this run: what it took, what faster kernels reach, and
    /// what perfect fusion inside each component reaches past that.
    pub fn ladder(&self) -> Ladder {
        Ladder {
            measured: self.wall(),
            host: self.framework().unwrap_or_default(),
            kernels: self
                .pass
                .breakdown
                .tree
                .iter()
                .map(|component| self.deepest_floor(component))
                .sum(),
            fused: self
                .pass
                .breakdown
                .tree
                .iter()
                .map(|component| self.outermost_floor(component))
                .sum(),
        }
    }

    /// The floor of the outermost priced components under `component` — each at
    /// its own floor, which assumes perfect fusion inside it.
    ///
    /// Descends through a component that declares none, so an unpriced root
    /// — the model itself — does not hide the priced components under it and
    /// leave the fusion rung equal to the kernel one. What such a component
    /// launched itself stands at what the device spent on it, as in
    /// [`deepest_floor`](Self::deepest_floor).
    fn outermost_floor(&self, component: &Component) -> Duration {
        if let Some(floor) = self.own_floor(component) {
            return floor;
        }
        match self.composite(component) {
            true => {
                component
                    .children
                    .iter()
                    .map(|child| self.outermost_floor(child))
                    .sum::<Duration>()
                    + self.apportion(self.unpriced_kernel_time(component))
            }
            false => self.apportioned(component),
        }
    }

    /// This component's own floor, when it declared bounds and the device was
    /// probed — whatever it measured. See [`Floor`].
    fn own_floor(&self, component: &Component) -> Option<Duration> {
        self.floor_of(component).map(|floor| floor.time)
    }

    /// This component's floor and the wall that sets it.
    pub(crate) fn floor_of(&self, component: &Component) -> Option<Floor> {
        Floor::new(self.priced(component)?, &self.peaks)
    }

    /// What `component` is held to, unless its work all ran elsewhere. See
    /// [`is_fused_elsewhere`](Self::is_fused_elsewhere).
    fn priced(&self, component: &Component) -> Option<Bounds> {
        match self.is_fused_elsewhere(component) {
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
    pub(super) fn held(component: &Component) -> Option<Bounds> {
        let own = component.bounds? + component.observed.unwrap_or_default();
        let flops = component
            .parts_bounds()
            .map_or(own.flops, |parts| own.flops.max(parts.flops));
        Some(Bounds { flops, ..own })
    }

    /// The floor of the finest-grained declarations under `component` — every
    /// component at its own roof, taken as deep as the tree prices one.
    ///
    /// Falls back to the component's own floor where nothing below declares one, and
    /// to its measured time where nothing does at all: a component nobody
    /// priced cannot be claimed back, so it counts at what it cost.
    pub(crate) fn deepest_floor(&self, component: &Component) -> Duration {
        if self.composite(component) {
            let children: Duration = component
                .children
                .iter()
                .map(|child| self.deepest_floor(child))
                .sum();
            // Whatever the children did not account for is this component's
            // own work, and no declaration prices it — so it stands, at what
            // the device spent on it.
            return children + self.apportion(self.unpriced_kernel_time(component));
        }
        self.own_floor(component)
            .unwrap_or(self.apportioned(component))
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
    pub(super) fn unpriced_kernel_time(&self, component: &Component) -> Duration {
        let priced: Vec<ComponentPath> = component
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

    /// This component's milliseconds: its kernels' own device time where they
    /// were measured in place, and its share of the clean pass where any
    /// launch was timed alone.
    ///
    /// In place, nothing waited on a kernel while the pass ran, so each
    /// measurement is the kernel as it runs in a full queue and needs no
    /// correcting. The components then sum to the device's time, and the clean pass's
    /// remainder is the host's — the `framework` line.
    ///
    /// **Timed alone, the collected pass has no absolute clock.** Its per-launch device
    /// times are each measured with the kernel running *alone*, since timing a
    /// launch blocks on it, so two collections of the same work can read twice
    /// apart against a clean pass that barely moved.
    /// What the collected pass measures reliably is the **proportions**,
    /// because every kernel in it is measured the same way.
    ///
    /// So the proportions are apportioned onto the one unperturbed number
    /// there is: the median wall time of the clean pass. The result is
    /// absolute, sums to that wall, and is the model's time rather than the
    /// instrumentation's.
    ///
    /// What it assumes is that serialization inflates every kernel alike. It
    /// does not, quite — a fixed cost per launch over-weights a component that
    /// runs many small kernels against one that runs a few large ones — so a
    /// component is good to a few percent and not to the last digit.
    ///
    /// Timed alone with no clean pass measured, there is nothing to apportion
    /// onto: the components are the serialized kernels' own times, and only their
    /// proportions mean anything, which the header says.
    pub(crate) fn apportioned(&self, component: &Component) -> Duration {
        self.apportion(component.total_time())
    }

    /// One measured duration in a component's milliseconds. See
    /// [`apportioned`](Self::apportioned).
    pub(crate) fn apportion(&self, measured: Duration) -> Duration {
        let wall = match self.reading {
            TimeReading::ApportionedOnto(wall) => wall,
            TimeReading::InPlace | TimeReading::Serialized => return measured,
        };
        let total = self.pass.breakdown.total_time();
        if total.is_zero() {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(wall.as_secs_f64() * measured.as_secs_f64() / total.as_secs_f64())
    }

    /// This component's share of the clean pass's host time: the `framework`
    /// line, split by how long the device waited before each component's
    /// kernels in the collected pass. `None` where the pass has no host/device
    /// split — see [`framework`](Self::framework).
    ///
    /// Split rather than read off directly, because the collected pass's host
    /// pays a round trip per launch to time it, so its idle stretches are
    /// longer than the clean pass's by a cost every launch shares. As shares
    /// they still say which components the device waited on; as absolutes they
    /// would describe the instrumentation. Where that shared cost is most of
    /// every stretch the split falls back to counting launches, which is the
    /// honest reading of a device waiting on the same work each time.
    pub(crate) fn host(&self, component: &Component) -> Option<Duration> {
        self.host_share(component.total_wait())
    }

    /// The clean pass's host time charged to `waited` of the collected pass's.
    pub(crate) fn host_share(&self, waited: Duration) -> Option<Duration> {
        let framework = self.framework()?;
        let total = self.pass.breakdown.total_wait();
        (!total.is_zero()).then(|| {
            Duration::from_secs_f64(
                framework.as_secs_f64() * waited.as_secs_f64() / total.as_secs_f64(),
            )
        })
    }

    /// Score one component against its binding wall, over its apportioned share of
    /// the clean pass.
    ///
    /// `None` where nothing under the component was timed: a percentage over
    /// zero measured time is not a percentage.
    pub(crate) fn score_of(&self, component: &Component) -> Option<Score> {
        let measured = self.apportioned(component);
        (!measured.is_zero())
            .then(|| Score::new(self.priced(component)?, measured, &self.peaks))
            .flatten()
    }
}
