//! Every priced component by what it could still give back.

use std::fmt::{self, Display};
use std::time::Duration;

use burn_instrument::Component;

use super::Report;
use super::table::ms;

/// One component's line on the worklist: what it could give back, and which
/// lever gives it.
struct Recoverable {
    label: String,
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
    /// Most recoverable first.
    entries: Vec<Recoverable>,
    /// Whether the pass lost kernel time, so the gains are differences against timings that
    /// came in short.
    lost_time: bool,
}

impl Worklist {
    pub(super) fn new(report: &Report<'_>) -> Self {
        fn visit(report: &Report<'_>, component: &Component, out: &mut Vec<Recoverable>) {
            // The third lever: what the device waited on the host for before this
            // component's own kernels. Its own, not its subtree's, so a
            // millisecond is listed once however deep the component sits.
            if let Some(host) = report
                .host_share(component.own_wait)
                .filter(|host| !host.is_zero())
            {
                let launches: usize = component.kernels.iter().map(|kernel| kernel.launches).sum();
                // A wait before a single launch is the host's work ahead of it,
                // which no launch can be taken away from.
                let (lever, against) = match launches {
                    1 => (Lever::LessHostWork, "before its one launch".to_string()),
                    _ => (Lever::FewerLaunches, format!("{launches} launches")),
                };
                out.push(Recoverable {
                    label: component.label.clone(),
                    gain: host,
                    lever,
                    against,
                });
            }
            if let Some(score) = report.score_of(component) {
                // A component at the roof has nothing to give back, so it sorts to
                // the bottom on its own — filtering it out would hide the component that
                // most deserves to be read.
                let (gain, lever) = match report.composite(component) {
                    true => (
                        report.deepest_floor(component).saturating_sub(score.floor),
                        Lever::FusingInside,
                    ),
                    false => (
                        score.gain(report.apportioned(component)),
                        Lever::FasterKernel,
                    ),
                };
                out.push(Recoverable {
                    label: component.label.clone(),
                    gain,
                    lever,
                    against: format!("{:>3.0}% of {} peak", score.achieved * 100.0, score.bound),
                });
            }
            for child in &component.children {
                visit(report, child, out);
            }
        }
        let mut entries = Vec::new();
        for component in &report.pass().breakdown.tree {
            visit(report, component, &mut entries);
        }
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.gain));
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
        writeln!(f, "\n  worth reading first    recoverable   by")?;
        // Its gains are differences against those timings and its waits are
        // gaps between them, so it carries the same caveat.
        if self.lost_time {
            writeln!(
                f,
                "    (from a pass that lost kernel time — see `untimed`; re-measure first)"
            )?;
        }
        for entry in self.entries.iter().take(5) {
            writeln!(
                f,
                "    {:<22} {:>6} ms   {:<16} {}",
                entry.label,
                ms(entry.gain),
                entry.lever,
                entry.against
            )?;
        }
        Ok(())
    }
}
