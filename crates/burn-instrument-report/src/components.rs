//! A run's components as a report reads them: each one's time, its share of the run's work
//! and of the host, how much of it fused elsewhere, and what it is held to — the figures the
//! table prints, for a front door that lays them out its own way.

use std::fmt::{self, Display};
use std::time::Duration;

use burn_instrument::{Component, ComponentPath, FusedShare, Kernel};

use crate::Report;
use crate::roofline::{Floor, Score};
use crate::share::Percent;

/// One component of a run, read as the table reads it, with its children's readings and its
/// own kernels'.
#[derive(Clone, Debug)]
pub struct ComponentReading<'a> {
    /// The component as the run recorded it: its label, entries, variants and call sites.
    pub component: &'a Component,
    /// The labels from the run's root down to it: what tells it apart from a same-named
    /// component under another parent, and what the worklist and a comparison key it by.
    pub path: ComponentPath,
    /// Its time as the report reads the run's: its kernels' device time where they were
    /// timed in place, its share of the clean run where they were each timed alone — only the
    /// work it ran itself where part of it [fused elsewhere](Self::fused). `None` where all of
    /// it did: its cost is a kernel listed elsewhere, and a zero would read as no cost.
    pub time: Option<Duration>,
    /// Its kernels' device time of the whole run's; `None` where the run timed nothing, or
    /// all of its work fused elsewhere.
    pub share: Option<WorkShare>,
    /// The clean run's host time charged to it — the device waiting on the host before its
    /// kernels — where the run splits into the device's time and the host's, and its work is
    /// its own.
    pub host: Option<Duration>,
    /// How much of its work ran in a fused kernel recorded elsewhere, if any did.
    pub fused: Option<FusedShare>,
    /// Its time against the wall it is bound by, where it is priced and its time is its own.
    pub score: Option<Score>,
    /// What it costs at best, where anything under it is priced.
    pub floor: Option<Floor>,
    /// The components it entered, in first-seen order.
    pub children: Vec<ComponentReading<'a>>,
    /// The kernels recorded on it, in first-seen order.
    pub kernels: Vec<KernelReading<'a>>,
}

/// One kernel of a run, read as the table reads it.
#[derive(Clone, Debug)]
pub struct KernelReading<'a> {
    /// The kernel as the run recorded it: its full name, launches and device time.
    pub kernel: &'a Kernel,
    /// Its name as the table prints it: the short form of its type where that tells it
    /// apart from the kernels listed with it, a fuller one for all of them where it does
    /// not.
    pub name: String,
    /// Its time, read as its component's is.
    pub time: Duration,
    /// Its device time of the whole run's; `None` where the run timed nothing.
    pub share: Option<WorkShare>,
}

/// A part of the run's device time, from 0 to 1: what the table's `share` column prints.
///
/// A type of its own because a report holds a second fraction beside it, a component's
/// [`Score::achieved`] against its device's peak, and a component at 3% of the memory peak
/// read as 3% of the run is the misreading the table's header exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct WorkShare(pub f64);

/// As a percentage, to a tenth.
impl Display for WorkShare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Percent(self.0).fmt(f)
    }
}

/// Why no component claimed a group of kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unclaimed {
    /// Fused work the last components left queued, which the sync ending the run ran.
    EndOfRun,
    /// Launched under no component, or fused across two roots.
    Untagged,
}

impl Display for Unclaimed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::EndOfRun => "end of run",
            Self::Untagged => "untagged",
        })
    }
}

/// Kernels no component claimed, for one reason, read as the table reads them.
#[derive(Clone, Debug)]
pub struct UnclaimedReading<'a> {
    /// Why no component claimed them.
    pub unclaimed: Unclaimed,
    /// The kernels, in first-seen order.
    pub kernels: Vec<KernelReading<'a>>,
}

impl<'a> ComponentReading<'a> {
    /// `component`, at `path`, read by `report`.
    pub(crate) fn new(report: &Report<'a>, component: &'a Component, path: ComponentPath) -> Self {
        let roofline = report.roofline();
        let fused = roofline.fused_elsewhere().share(component);
        let own = fused != Some(FusedShare::All);
        Self {
            component,
            time: own.then(|| report.reading().apportioned(component)),
            share: report.share(component.total_time()).filter(|_| own),
            host: report
                .host_split()
                .share(component.total_wait())
                .filter(|_| own),
            fused,
            score: roofline.score_of(component),
            floor: roofline.floor_of(component),
            children: component
                .children
                .iter()
                .map(|child| Self::new(report, child, path.child(&child.label)))
                .collect(),
            kernels: KernelReading::list(report, &component.kernels),
            path,
        }
    }

    /// This reading and every one below it, depth-first, each with its depth.
    pub fn walk(&self) -> Vec<(usize, &ComponentReading<'a>)> {
        fn descend<'r, 'a>(
            reading: &'r ComponentReading<'a>,
            depth: usize,
            out: &mut Vec<(usize, &'r ComponentReading<'a>)>,
        ) {
            out.push((depth, reading));
            for child in &reading.children {
                descend(child, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        descend(self, 0, &mut out);
        out
    }
}

impl<'a> KernelReading<'a> {
    /// `kernels`, listed together, each read by `report` and named under the shortest name
    /// that still tells it apart from the others.
    ///
    /// A kernel's type name carries its whole generic instantiation and runs to
    /// several hundred characters, so the table would be unreadable at full
    /// length. `Short` — the last path segment before the generics — is what reads
    /// well, but three matmul families instantiate one `MatmulEntry` and collapsing
    /// them would print two different kernels under one name, which is worse than
    /// a long name. So `Short` is used only where it stays unambiguous *within the
    /// list*, and the fuller form serves the rest.
    pub(crate) fn list(report: &Report<'a>, kernels: &'a [Kernel]) -> Vec<Self> {
        use cubecl::config::{TypeNameFormatLevel, type_name_format};

        let short: Vec<String> = kernels
            .iter()
            .map(|kernel| type_name_format(&kernel.name, TypeNameFormatLevel::Short))
            .collect();
        let distinct = short.iter().collect::<std::collections::HashSet<_>>().len();
        let level = match distinct == kernels.len() {
            true => TypeNameFormatLevel::Short,
            false => TypeNameFormatLevel::Balanced,
        };
        kernels
            .iter()
            .map(|kernel| Self {
                kernel,
                name: type_name_format(&kernel.name, level),
                time: report.reading().apportion(kernel.total),
                share: report.share(kernel.total),
            })
            .collect()
    }
}
