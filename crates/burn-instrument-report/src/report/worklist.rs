//! Every priced component by what it could still give back.

use std::fmt::{self, Display};
use std::time::Duration;

use burn_instrument::Component;

use super::Report;

/// One component's line on the worklist: what it could give back, and which
/// lever gives it.
pub(super) struct Recoverable {
    pub(super) label: String,
    pub(super) gain: Duration,
    /// The lever, named — each is different work for different people.
    pub(super) lever: Lever,
    /// Where it stands against its binding wall.
    pub(super) against: String,
}

/// What gives a component's time back, each a different kind of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lever {
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
pub(super) fn worklist(rendered: &Report<'_>) -> Vec<Recoverable> {
    fn visit(rendered: &Report<'_>, component: &Component, out: &mut Vec<Recoverable>) {
        // The third lever: what the device waited on the host for before this
        // component's own kernels. Its own, not its subtree's, so a
        // millisecond is listed once however deep the component sits.
        if let Some(host) = rendered
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
        if let Some(score) = rendered.score_of(component) {
            // A component at the roof has nothing to give back, so it sorts to
            // the bottom on its own — filtering it out would hide the component that
            // most deserves to be read.
            let (gain, lever) = match rendered.composite(component) {
                true => (
                    rendered
                        .deepest_floor(component)
                        .saturating_sub(score.floor),
                    Lever::FusingInside,
                ),
                false => (
                    score.gain(rendered.apportioned(component)),
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
            visit(rendered, child, out);
        }
    }
    let mut out = Vec::new();
    for component in &rendered.pass().breakdown.tree {
        visit(rendered, component, &mut out);
    }
    out.sort_by_key(|component| std::cmp::Reverse(component.gain));
    out
}
