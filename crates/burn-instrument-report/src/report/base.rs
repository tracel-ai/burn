//! The table a collected pass prints.
//!
//! One page: the device and what it can do, how the pass ran, the component
//! tree with each component scored against the wall it is bound by, the ladder, and
//! a worklist ranked by what is actually recoverable. It is plain text,
//! greppable, and says what it does not know rather than leaving a column
//! blank.

use std::time::Duration;

use burn_instrument::{CollectedPass, PeakFinding, Peaks};

use crate::Comparison;
use crate::reading::{HostSplit, TimeReading};
use crate::roofline::{Ladder, Roofline};

/// A collected pass, scored against its device's peaks, and the table that
/// renders it ([`Display`](std::fmt::Display)).
///
/// A front door that knows what the pass was (a model, a batch, a phase)
/// prints that above this table; the table says only what the file does.
pub struct Report<'a> {
    pass: &'a CollectedPass,
    /// What the header must say about the probe set before its numbers are
    /// read, already ordered — see [`Peaks::findings`].
    findings: Vec<PeakFinding>,
    /// The pass's roofline: what each component is held to.
    roofline: Roofline<'a>,
    /// What a component's milliseconds are, given how its kernels were timed.
    reading: TimeReading,
    /// Whether the pass splits into the device's time and the host's.
    host_split: HostSplit,
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
        let peaks: Peaks = pass.peaks.unwrap_or_default();
        let reading = TimeReading::new(pass);
        Self {
            pass,
            findings: peaks.findings(),
            roofline: Roofline::new(&pass.breakdown, peaks.trusted(), reading),
            reading,
            host_split: HostSplit::new(pass),
        }
    }

    /// This pass held against `saved`, component by component.
    pub fn against(self, saved: &'a CollectedPass) -> Comparison<'a> {
        Comparison::new(self, saved)
    }

    /// What the pass took: the clean pass's wall time where it was measured,
    /// and the collected pass's device time otherwise.
    pub fn wall(&self) -> Duration {
        self.pass
            .clean_wall
            .unwrap_or_else(|| self.pass.breakdown.total_time())
    }

    /// The ladder for this run: what it took, what faster kernels reach, and
    /// what perfect fusion inside each component reaches past that.
    pub fn ladder(&self) -> Ladder {
        let tree = &self.pass.breakdown.tree;
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

    /// The pass this report scores.
    pub(crate) fn pass(&self) -> &'a CollectedPass {
        self.pass
    }

    /// What the probe set's readings say about themselves, in the order a report says it.
    pub(crate) fn findings(&self) -> &[PeakFinding] {
        &self.findings
    }

    /// The pass's roofline: what each component is held to.
    pub(crate) fn roofline(&self) -> &Roofline<'a> {
        &self.roofline
    }

    /// What a component's milliseconds are.
    pub(crate) fn reading(&self) -> TimeReading {
        self.reading
    }

    /// Whether the pass splits into the device's time and the host's.
    pub(crate) fn host_split(&self) -> HostSplit {
        self.host_split
    }
}
