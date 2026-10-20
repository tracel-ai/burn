//! Every priced component by what it could still give back.

use std::fmt::{self, Display};
use std::time::Duration;

use burn_instrument::ComponentPath;

use super::Report;
use crate::millis::Millis;

/// One component's line on the worklist: what it could give back, and which
/// lever gives it.
struct Recoverable {
    /// Where the component sits: a label alone is ambiguous, since `down` sits under
    /// every layer.
    path: ComponentPath,
    gain: Duration,
    /// The lever, named — each is different work for different people.
    lever: Lever,
    /// Where it stands against its binding wall.
    against: String,
}

/// What gives a component's time back, each a different kind of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lever {
    /// The host's work ahead of a single launch, which no launch can be taken away from.
    LessHostWork,
    /// The device waiting between launches, which fewer of them removes.
    FewerLaunches,
    /// The round trips between a composite's parts, which no faster kernel reaches.
    FusingInside,
    /// A leaf below its floor, which a faster kernel recovers.
    FasterKernel,
}

/// What the worklist prints, padded to its column.
impl Display for Lever {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Lever::LessHostWork => "less host work",
            Lever::FewerLaunches => "fewer launches",
            Lever::FusingInside => "fusing inside",
            Lever::FasterKernel => "a faster kernel",
        })
    }
}

/// Every priced component by what it could still give back, **counting each
/// millisecond once**.
///
/// A composite's own gain contains its children's, so ranking it by that would
/// list the same time again under every ancestor — and a reader adding the list
/// up would find more on the table than the pass took. So each component is ranked by
/// the lever that is only available at *its* level: a leaf by what a faster
/// kernel would buy, a composite by the round-trip between its parts that no
/// faster kernel can reach. Those are the ladder's two rungs, split per
/// component, and they are disjoint by construction.
pub(super) struct Worklist {
    /// The most recoverable, first.
    entries: Vec<Recoverable>,
    /// Whether the pass lost kernel time, so the gains are differences against timings that
    /// came in short.
    lost_time: bool,
}

impl Worklist {
    /// How many components the worklist names.
    const SHOWN: usize = 5;

    pub(super) fn new(report: &Report<'_>) -> Self {
        let roofline = report.roofline();
        let mut entries = Vec::new();
        for (path, component) in report.pass().breakdown.components() {
            // The third lever: what the device waited on the host for before this
            // component's own kernels. Its own, not its subtree's, so a
            // millisecond is listed once however deep the component sits.
            if let Some(host) = report
                .host_split()
                .share(component.own_wait)
                .filter(|host| !host.is_zero())
            {
                let launches: usize = component.kernels.iter().map(|kernel| kernel.launches).sum();
                // A wait before a single launch is the host's work ahead of it,
                // which no launch can be taken away from.
                let (lever, against) = match launches {
                    1 => (Lever::LessHostWork, "before its one launch".to_string()),
                    _ => (Lever::FewerLaunches, format!("{launches} launches")),
                };
                entries.push(Recoverable {
                    path: path.clone(),
                    gain: host,
                    lever,
                    against,
                });
            }
            if let Some(score) = roofline.score_of(component) {
                // A component at the roof has nothing to give back, so it sorts to
                // the bottom on its own — filtering it out would hide the component that
                // most deserves to be read.
                let (gain, lever) = match roofline.composite(component) {
                    true => (
                        roofline
                            .deepest_floor(component)
                            .saturating_sub(score.floor.time),
                        Lever::FusingInside,
                    ),
                    false => (
                        score.gain(report.reading().apportioned(component)),
                        Lever::FasterKernel,
                    ),
                };
                entries.push(Recoverable {
                    path,
                    gain,
                    lever,
                    against: format!(
                        "{:>3.0}% of {} peak",
                        score.achieved * 100.0,
                        score.floor.bound
                    ),
                });
            }
        }
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.gain));
        entries.truncate(Self::SHOWN);
        Self {
            entries,
            lost_time: report.pass().breakdown.untimed().next().is_some(),
        }
    }
}

/// The five components most worth reading first, or nothing where none is priced.
impl Display for Worklist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.entries.is_empty() {
            return Ok(());
        }
        // Each component by its whole path, so the column grows to the longest of them
        // and the gains stay under `recoverable`.
        let width = self
            .entries
            .iter()
            .map(|entry| entry.path.to_string().chars().count())
            .chain([22])
            .max()
            .unwrap_or(22);
        writeln!(
            f,
            "\n  {:<heading$}recoverable   by",
            "worth reading first",
            heading = width + 1
        )?;
        // Its gains are differences against those timings and its waits are
        // gaps between them, so it carries the same caveat.
        if self.lost_time {
            writeln!(
                f,
                "    (from a pass that lost kernel time — see `untimed`; re-measure first)"
            )?;
        }
        for entry in &self.entries {
            writeln!(
                f,
                "    {:<width$} {:>6} ms   {:<16} {}",
                entry.path,
                Millis(entry.gain),
                entry.lever,
                entry.against
            )?;
        }
        Ok(())
    }
}
