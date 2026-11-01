//! The table a measurement prints.
//!
//! One page: the device and what it can do, how it ran, the component
//! tree with each component scored against the wall it is bound by, the ladder, and
//! a worklist ranked by what is actually recoverable. It is plain text,
//! greppable, and says what it does not know rather than leaving a column
//! blank.

use std::time::Duration;

use burn_instrument::{ComponentPath, Measurement, PeakFinding, Peaks};

use crate::Comparison;
use crate::components::{ComponentReading, KernelReading, Unclaimed, UnclaimedReading, WorkShare};
use crate::reading::{HostSplit, TimeReading};
use crate::roofline::{Ladder, Roofline};
use crate::share::Share;

/// A measurement, scored against its device's peaks, and the table that
/// renders it ([`Display`](std::fmt::Display)).
///
/// A front door that knows what the run was (a model, a batch, a phase)
/// prints that above this table; the table says only what the file does.
pub struct Report<'a> {
    measurement: &'a Measurement,
    /// What the header must say about the probe set before its numbers are
    /// read, already ordered — see [`Peaks::findings`].
    findings: Vec<PeakFinding>,
    /// The run's roofline: what each component is held to.
    roofline: Roofline<'a>,
    /// What a component's milliseconds are, given how its kernels were timed.
    reading: TimeReading,
    /// Whether the run splits into the device's time and the host's.
    host_split: HostSplit,
    /// Every kernel's device time, summed once: what every share is of.
    device_time: Duration,
}

impl<'a> Report<'a> {
    /// Score `measurement` against the peaks its device measured, or against none
    /// where it was not probed.
    ///
    /// An impossible peak is dropped here rather than divided by — see
    /// [`Peaks::trusted`]. The findings are read off the raw peaks first, so
    /// the header can say what was rejected: a probe that half-ran is a fact
    /// about the run, and silently falling back to a coarser wall would leave a
    /// reader wondering why nothing is held against `flops`, or why every component
    /// moved at once.
    pub fn new(measurement: &'a Measurement) -> Self {
        let peaks: Peaks = measurement.peaks.unwrap_or_default();
        let reading = TimeReading::new(measurement);
        Self {
            measurement,
            findings: peaks.findings(),
            roofline: Roofline::new(&measurement.breakdown, peaks.trusted(), reading),
            reading,
            host_split: HostSplit::new(measurement),
            device_time: measurement.breakdown.total_time(),
        }
    }

    /// This measurement held against `saved`, component by component.
    pub fn against(self, saved: &'a Measurement) -> Comparison<'a> {
        Comparison::new(self, saved)
    }

    /// What the run took: the clean run's wall time where it was measured,
    /// and the instrumented run's device time otherwise.
    pub fn wall(&self) -> Duration {
        self.measurement
            .clean_wall
            .unwrap_or_else(|| self.measurement.breakdown.total_time())
    }

    /// The ladder for this run: what it took, what faster kernels reach, and
    /// what perfect fusion inside each component reaches past that.
    pub fn ladder(&self) -> Ladder {
        let tree = &self.measurement.breakdown.tree;
        Ladder {
            measured: self.wall(),
            host: self.host_split.host().unwrap_or_default(),
            kernels: tree
                .iter()
                .map(|component| self.roofline.deepest_floor(component))
                .sum(),
            fused: tree
                .iter()
                .map(|component| self.roofline.outermost_floor(component))
                .sum(),
        }
    }

    /// The run's components as the table reads them, roots first, in first-seen order.
    pub fn components(&self) -> Vec<ComponentReading<'a>> {
        self.measurement
            .breakdown
            .tree
            .iter()
            .map(|root| {
                ComponentReading::new(self, root, ComponentPath::new([root.label.as_str()]))
            })
            .collect()
    }

    /// The kernels no component claimed, as the table lists them after the components: the
    /// end of the run's, then the untagged; a group with no kernel is left out.
    pub fn unclaimed(&self) -> Vec<UnclaimedReading<'a>> {
        let breakdown = &self.measurement.breakdown;
        [
            (Unclaimed::EndOfRun, &breakdown.end_of_run),
            (Unclaimed::Untagged, &breakdown.untagged),
        ]
        .into_iter()
        .filter(|(_, kernels)| !kernels.is_empty())
        .map(|(unclaimed, kernels)| UnclaimedReading {
            unclaimed,
            kernels: KernelReading::list(self, kernels),
        })
        .collect()
    }

    /// `part` of the run's device time; `None` where the run timed nothing.
    pub(crate) fn share(&self, part: Duration) -> Option<WorkShare> {
        Share {
            part,
            whole: self.device_time,
        }
        .fraction()
        .map(WorkShare)
    }

    /// The measurement this report scores.
    pub(crate) fn measurement(&self) -> &'a Measurement {
        self.measurement
    }

    /// What the probe set's readings say about themselves, in the order a report says it.
    pub(crate) fn findings(&self) -> &[PeakFinding] {
        &self.findings
    }

    /// The run's roofline: what each component is held to.
    pub(crate) fn roofline(&self) -> &Roofline<'a> {
        &self.roofline
    }

    /// What a component's milliseconds are.
    pub(crate) fn reading(&self) -> TimeReading {
        self.reading
    }

    /// Whether the run splits into the device's time and the host's.
    pub(crate) fn host_split(&self) -> HostSplit {
        self.host_split
    }
}
