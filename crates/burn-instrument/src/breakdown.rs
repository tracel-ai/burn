//! What a collected pass leaves behind: the components it entered, the
//! kernels each launched, and what the collection itself can say about how
//! far those numbers are to be trusted.

use std::time::Duration;

use crate::row_path::RowPath;
use crate::tree::{Kernel, Node, Unplaced};

/// One collected pass, broken down by component.
///
/// What [`SharedCollector::collect`](crate::SharedCollector::collect) returns:
/// data only, so a breakdown collected from a run and one written out in a
/// test are the same kind of value.
#[derive(Clone, Debug)]
pub struct Breakdown {
    /// The components the collected pass entered.
    pub tree: Vec<Node>,
    /// Kernels the pass launched under no component at all.
    pub untagged: Vec<Kernel>,
    /// Kernels the sync ending the pass ran: fused work the last components
    /// left queued, which no op after them was there to seal. See
    /// [`SharedCollector::collect`](crate::SharedCollector::collect).
    pub end_of_pass: Vec<Kernel>,
    /// How long the device waited on the host before the untagged kernels
    /// and the end of the pass's — the idle time no row is charged with. See
    /// [`Node::waited`].
    pub waited_outside: Duration,
    /// Whether any launch was timed on the host rather than the device.
    ///
    /// wgpu budgets its timestamp query sets and falls back to a host timer
    /// when they run out, and a host timing is wall around a blocking submit —
    /// submission and sync included, so a pass with one reads several times
    /// its device time.
    pub host_timed: bool,
    /// Whether any launch was timed alone — read back as it launched, or by a
    /// host timer — so the kernels ran one at a time and their sum is not a
    /// pipelined pass's device time.
    ///
    /// Decides how the rows are read: in place, a row's milliseconds are its
    /// kernels' own and the clean pass splits into device and host; alone, the
    /// device times are only good as proportions and are apportioned onto the
    /// clean pass instead.
    pub timed_alone: bool,
    /// Kernel time that reached the collector under a different row from the
    /// launch it measured, so no row could account for it.
    ///
    /// Zero on every pass that resolves its timings inline, which is what the
    /// runtime does today. Reported rather than assumed, because
    /// [`device`](Self::device) is the denominator of every apportioned
    /// millisecond in a report.
    pub unplaced: Unplaced,
    /// Whether the collected pass ran without fusion — no fusion linked, or a
    /// device whose ops do not go through it.
    ///
    /// Each pass has its own way of being read wrong, and the breakdown says
    /// which it ran. Unfused, it is a graph production does not run: what a
    /// fused kernel would do across two components is charged to each of them.
    /// Fused, it is production's graph, and a kernel that carried out several
    /// components' work is charged to their closest common parent, or outside
    /// every component where they share none — so a component whose work all
    /// fused reads zero, and the kernel that did it is listed elsewhere.
    pub unfused: bool,
}

impl Breakdown {
    /// Every kernel of the pass: the ones under a component, then the
    /// untagged ones and the ones the end of the pass ran.
    pub fn kernels(&self) -> impl Iterator<Item = &Kernel> {
        self.tree
            .iter()
            .flat_map(Node::walk)
            .flat_map(|(_, node)| &node.kernels)
            .chain(&self.untagged)
            .chain(&self.end_of_pass)
    }

    /// The kernels that launched and measured no time at all.
    ///
    /// A launch the device ran took some time, so a zero is a timing the
    /// collection lost rather than a fast kernel — and every row it sits under,
    /// every share and the device total are short by it. Distinct from
    /// [`unplaced`](Self::unplaced), which is time that arrived with no row to
    /// go to: this is a row whose time never arrived.
    pub fn untimed(&self) -> impl Iterator<Item = &Kernel> {
        self.kernels()
            .filter(|kernel| kernel.launches > 0 && kernel.total.is_zero())
    }

    /// What the device spent on this pass: every kernel, summed.
    pub fn device(&self) -> Duration {
        self.kernels().map(|kernel| kernel.total).sum()
    }

    /// How long the device waited on the host across the collected pass,
    /// between every pair of kernels: what [`Node::waited`] is apportioned
    /// by.
    pub fn waited(&self) -> Duration {
        self.tree.iter().map(Node::waited_total).sum::<Duration>() + self.waited_outside
    }

    /// The rows whose work all ran in a fused kernel recorded elsewhere: no
    /// kernel under them, and named by a kernel on a row above them or outside
    /// every component. Such a row ran nothing, so its floor is the kernel's
    /// that carried its work, and a report sets the row's own aside.
    ///
    /// Asked of the kernels rather than inferred from a zero: a component that
    /// registered no device work — views and reshapes — also has no kernel, and
    /// its zero is its time.
    pub fn fused_elsewhere(&self) -> Vec<&Node> {
        fn visit<'a>(
            breakdown: &'a Breakdown,
            node: &'a Node,
            ancestors: &mut Vec<&'a Node>,
            out: &mut Vec<&'a Node>,
        ) {
            let named_above = ancestors.iter().enumerate().any(|(at, ancestor)| {
                // What this ancestor's kernels call `node`: the components between
                // the two, then `node`. A bare label would also match a same-named
                // row under another branch of the same ancestor, and a mixed
                // stack's dense and sparse `down` are both `down`.
                let path = RowPath::new(
                    ancestors[at + 1..]
                        .iter()
                        .chain([&node])
                        .map(|row| row.label.as_str()),
                );
                ancestor
                    .kernels
                    .iter()
                    .any(|kernel| kernel.fuses.contains(&path))
            });
            // Outside every component, a kernel names a row by its path from the
            // root.
            let path = RowPath::new(
                ancestors
                    .iter()
                    .chain([&node])
                    .map(|row| row.label.as_str()),
            );
            let named_outside = breakdown
                .untagged
                .iter()
                .chain(&breakdown.end_of_pass)
                .any(|kernel| kernel.fuses.contains(&path));
            if node.device_time().is_zero() && (named_above || named_outside) {
                out.push(node);
            }
            ancestors.push(node);
            for child in &node.children {
                visit(breakdown, child, ancestors, out);
            }
            ancestors.pop();
        }
        let mut out = Vec::new();
        for root in &self.tree {
            visit(self, root, &mut Vec::new(), &mut out);
        }
        out
    }
}
