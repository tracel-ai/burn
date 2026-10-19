//! The tree a measured pass leaves behind: which components it entered, how
//! they nested, what each declared it should cost, and how long each took.
//!
//! Nesting is a consequence of composition — a component's forward calls its
//! children's forwards — so nothing here maintains a list of what sits inside
//! what. Entering the same component twice at one path merges: a 36-layer stack
//! is one row entered 36 times, not 36 rows.
//!
//! The same merge puts two *different* jobs under one heading when two entry
//! points share a label, and the sum reads exactly like one entry point run
//! twice. So `#[compute]` passes the site — `module_path!()` plus the function —
//! beside the label, and [`Node::sites`] keeps the distinct ones so a report
//! can name a row fed from more than one. A site is a function, not a role: one
//! function entered at two shapes is one site, and a role its caller knows is
//! named by an argument instead.

use std::time::Duration;

use super::cursor::{Attribution, Placement};
use super::timing::TimingQuality;
use crate::breakdown::{Breakdown, Kernel, Node, Variant};
use crate::row_path::RowPath;
use crate::{Bounds, Declaration};

/// The components a pass entered, as rows built while it runs, beside the rows
/// for kernels no component holds.
///
/// A flat arena, because a tree with a cursor into it cannot be expressed with
/// ordinary borrows: a row is its index, which is what
/// [`Attribution::Component`] carries.
#[derive(Default)]
pub(crate) struct ComponentTree {
    rows: Vec<Row>,
    /// Indices of the components entered and not yet left, outermost first.
    open: Vec<usize>,
    /// Indices of the roots, in first-seen order — a pass may enter more than
    /// one component at top level (an embedding, then a stack, then a head).
    roots: Vec<usize>,
    /// Kernels launched with no component open at all. A real category: a
    /// cache allocation, a sampler, anything the annotations do not cover.
    untagged: KernelCounter,
    /// Kernels the sync ending the pass ran: fused work still queued when the
    /// last component closed.
    end_of_pass: KernelCounter,
    /// How long the device waited before kernels no component launched — the
    /// untagged ones and the end of the pass. See [`Node::waited`].
    waited_outside: Duration,
}

impl ComponentTree {
    /// Enter the component `declared` names under the innermost open one, and
    /// say where its launches land.
    ///
    /// Entering a label already seen at this path merges into its row, so a
    /// repeated component is one row entered many times.
    pub(crate) fn enter(&mut self, declared: Declaration<'_>) -> Attribution {
        let row = self.resolve(declared.label);
        self.rows[row].enter(declared);
        self.open.push(row);
        Attribution::Component(row)
    }

    /// Leave the innermost open component, and say where launches land now.
    pub(crate) fn exit(&mut self) -> Attribution {
        self.open.pop();
        match self.open.last() {
            Some(row) => Attribution::Component(*row),
            None => Attribution::Untagged,
        }
    }

    /// The innermost component still open, if any.
    pub(crate) fn innermost(&self) -> Option<usize> {
        self.open.last().copied()
    }

    /// Count one launch of `kernel` on the row `placement` puts it on, named
    /// for the rows below it whose work it carried out.
    ///
    /// By name and count rather than one row per launch: a 28-layer stack
    /// issues the same kernel 28 times, and 28 identical rows say less than one
    /// row saying 28.
    pub(crate) fn count_launch(&mut self, placement: &Placement, kernel: &str) {
        // The row it lands on is not among what it names: `fuses` names rows
        // below it, and the row's own work is the row's by placement.
        let own = |row: &usize| placement.attribution == Attribution::Component(*row);
        let fuses_own = placement.span.iter().any(own);
        let fuses = placement
            .span
            .iter()
            .filter(|row| !own(row))
            .map(|row| self.path_under(*row, placement.attribution))
            .collect();
        self.kernels(placement.attribution)
            .count(kernel, fuses, fuses_own);
    }

    /// Add `duration` to the row that counted `kernel`'s launch under
    /// `attribution`, or `false` when no launch of it was counted there.
    pub(crate) fn credit(
        &mut self,
        attribution: Attribution,
        kernel: &str,
        duration: Duration,
    ) -> bool {
        self.kernels(attribution).credit(kernel, duration)
    }

    /// Charge `idle` to the row whose kernel ended it. See [`Node::waited`].
    pub(crate) fn charge_idle(&mut self, attribution: Attribution, idle: Duration) {
        match attribution {
            Attribution::Component(row) => self.rows[row].waited += idle,
            Attribution::Untagged | Attribution::EndOfPass => self.waited_outside += idle,
        }
    }

    /// The closest component every one of `rows` sits under, itself included
    /// — where a kernel that carried out all of their work belongs. `None` for
    /// rows under different roots, which share no row.
    pub(crate) fn common_parent(&self, rows: &[usize]) -> Option<usize> {
        let (first, rest) = rows.split_first()?;
        let mut common: Vec<usize> = self.ancestors(*first).collect();
        for row in rest {
            let theirs: Vec<usize> = self.ancestors(*row).collect();
            common.retain(|ancestor| theirs.contains(ancestor));
        }
        // Each list runs from the row up to its root, so the first ancestor
        // they share is the closest.
        common.first().copied()
    }

    /// Add `extra` to `row` and every parent that declares a floor. A row
    /// with no floor is timed but not scored, and stays that way; no traffic
    /// leaves every row as it was, so a batch of one marks nothing.
    pub(crate) fn add_observed(&mut self, row: usize, extra: Bounds) {
        if extra == Bounds::default() {
            return;
        }
        let scored: Vec<usize> = self.ancestors(row).collect();
        for row in scored {
            let row = &mut self.rows[row];
            if row.bounds.is_some() {
                row.observed = Some(row.observed.unwrap_or_default() + extra);
            }
        }
    }

    /// The breakdown these rows make, with what the collection says about its
    /// timings.
    ///
    /// # Panics
    ///
    /// If a component is still open — a tree read mid-pass is missing the
    /// kernels its open components have yet to launch.
    pub(crate) fn breakdown(&self, timings: TimingQuality, unfused: bool) -> Breakdown {
        assert!(
            self.open.is_empty(),
            "a component is still open: read the tree after the pass, not during it"
        );
        Breakdown {
            tree: self.roots.iter().map(|root| self.node(*root)).collect(),
            untagged: self.untagged.kernels(),
            end_of_pass: self.end_of_pass.kernels(),
            waited_outside: self.waited_outside,
            host_timed: timings.host_timed,
            timed_alone: timings.timed_alone,
            unplaced: timings.unplaced,
            unfused,
        }
    }

    /// The row `label` names under the innermost open component, created on
    /// first sight.
    fn resolve(&mut self, label: &str) -> usize {
        let parent = self.innermost();
        let siblings = match parent {
            Some(parent) => &self.rows[parent].children,
            None => &self.roots,
        };
        if let Some(found) = siblings
            .iter()
            .copied()
            .find(|row| self.rows[*row].label == label)
        {
            return found;
        }
        let row = self.rows.len();
        self.rows.push(Row::new(label, parent));
        match parent {
            Some(parent) => self.rows[parent].children.push(row),
            None => self.roots.push(row),
        }
        row
    }

    /// The kernels a launch under `attribution` is counted among.
    fn kernels(&mut self, attribution: Attribution) -> &mut KernelCounter {
        match attribution {
            Attribution::Untagged => &mut self.untagged,
            Attribution::Component(row) => &mut self.rows[row].kernels,
            Attribution::EndOfPass => &mut self.end_of_pass,
        }
    }

    /// `row` and every row above it, innermost first.
    fn ancestors(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(Some(row), |row| self.rows[*row].parent)
    }

    /// `row` named the way a reader of the row a kernel sits on would say it:
    /// the components between the two, then `row` itself.
    ///
    /// A path rather than a bare label, because a label is unique only among
    /// siblings. One `transformer block` in a mixed stack holds a dense
    /// `feed forward › down` and a `sparse feed forward › down`, and a kernel
    /// that named `down` could be either — which marks the wrong row as fused
    /// and prices the ladder off whichever one was found first.
    ///
    /// Relative, since the kernel is already recorded on `under`. A kernel
    /// under no component at all is named from the root.
    fn path_under(&self, row: usize, under: Attribution) -> RowPath {
        let stop = match under {
            Attribution::Component(under) => Some(under),
            Attribution::Untagged | Attribution::EndOfPass => None,
        };
        let mut labels: Vec<&str> = self
            .ancestors(row)
            .take_while(|ancestor| Some(*ancestor) != stop)
            .map(|ancestor| self.rows[ancestor].label.as_str())
            .collect();
        labels.reverse();
        RowPath::new(labels)
    }

    /// The finished node at `row`, and everything under it.
    fn node(&self, row: usize) -> Node {
        let row = &self.rows[row];
        Node {
            label: row.label.clone(),
            variants: row.variants.clone(),
            sites: row.sites.iter().map(|site| site.to_string()).collect(),
            bounds: row.bounds,
            observed: row.observed,
            entries: row.entries,
            children: row.children.iter().map(|child| self.node(*child)).collect(),
            kernels: row.kernels.kernels(),
            waited: row.waited,
        }
    }
}

/// One row while it is being built.
struct Row {
    label: String,
    /// The component it was entered under, or `None` for a root.
    parent: Option<usize>,
    /// Variants seen under this label, in first-seen order.
    variants: Vec<Variant>,
    /// Distinct call sites that entered this label, in first-seen order.
    sites: Vec<&'static str>,
    bounds: Option<Bounds>,
    /// See [`Node::observed`].
    observed: Option<Bounds>,
    entries: usize,
    children: Vec<usize>,
    kernels: KernelCounter,
    /// See [`Node::waited`].
    waited: Duration,
}

impl Row {
    fn new(label: &str, parent: Option<usize>) -> Self {
        Self {
            label: label.to_string(),
            parent,
            variants: Vec::new(),
            sites: Vec::new(),
            // Left empty: `enter` sums each entry's bounds into it over
            // `unwrap_or_default`, so seeding a zero here says nothing.
            bounds: None,
            observed: None,
            entries: 0,
            children: Vec::new(),
            kernels: KernelCounter::default(),
            waited: Duration::ZERO,
        }
    }

    /// Merge one more entry of this row.
    fn enter(&mut self, declared: Declaration<'_>) {
        self.entries += 1;
        // The set, not the count: what a reader acts on is *which* places fed
        // this row, and a site that fed 36 of 36 entries says the same thing as
        // one that fed 1.
        if !self.sites.contains(&declared.site) {
            self.sites.push(declared.site);
        }
        // Counted per entry, not per row: a component entered 36 times that
        // routed differently on two of them is the case the histogram exists
        // for, and a set would lose exactly that.
        if let Some(variant) = declared.variant {
            match self.variants.iter_mut().find(|seen| seen.name == variant) {
                Some(seen) => seen.entries += 1,
                None => self.variants.push(Variant {
                    name: variant.to_string(),
                    entries: 1,
                }),
            }
        }
        // Summed across entries — see `Node::bounds`. A component that declared
        // them on its first entry and not on a later one keeps the sum of what
        // was declared rather than losing the row's roofline entirely.
        if let Some(bounds) = declared.bounds {
            self.bounds = Some(self.bounds.unwrap_or_default() + bounds);
        }
    }
}

/// The kernels one row counted, by name, in first-seen order.
#[derive(Default)]
struct KernelCounter(Vec<Kernel>);

impl KernelCounter {
    /// Count one launch of `name`, carrying out the work of `fuses` below the
    /// row and, where `fuses_own`, the row's own.
    fn count(&mut self, name: &str, fuses: Vec<RowPath>, fuses_own: bool) {
        let kernel = match self.0.iter().position(|kernel| kernel.name == name) {
            Some(found) => &mut self.0[found],
            None => {
                self.0.push(Kernel {
                    name: name.to_string(),
                    launches: 0,
                    total: Duration::ZERO,
                    fuses: Vec::new(),
                    fuses_own: false,
                });
                self.0.last_mut().expect("pushed right above")
            }
        };
        kernel.launches += 1;
        kernel.fuses_own |= fuses_own;
        // The union across launches: one fused kernel serves a norm and a
        // projection in every layer, and the row names both once.
        for path in fuses {
            if !kernel.fuses.contains(&path) {
                kernel.fuses.push(path);
            }
        }
    }

    /// Add `duration` to `name`, or `false` when no launch of it was counted
    /// here.
    ///
    /// By row, not by name across the tree: every projection of a block can
    /// launch the same kernel, and a search by name alone would credit the
    /// first row it finds with every other row's time.
    fn credit(&mut self, name: &str, duration: Duration) -> bool {
        match self.0.iter_mut().find(|counted| counted.name == name) {
            Some(counted) => {
                counted.total += duration;
                true
            }
            None => false,
        }
    }

    fn kernels(&self) -> Vec<Kernel> {
        self.0.clone()
    }
}
