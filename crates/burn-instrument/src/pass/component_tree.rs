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
/// ordinary borrows: a row is its [`RowId`], which is what
/// [`Attribution::Component`] carries.
#[derive(Default)]
pub(crate) struct ComponentTree {
    rows: Vec<RowRecord>,
    /// The components entered and not yet left, outermost first.
    open: Vec<RowId>,
    /// The roots, in first-seen order — a pass may enter more than one
    /// component at top level (an embedding, then a stack, then a head).
    roots: Vec<RowId>,
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

/// One row of a [`ComponentTree`], valid in that tree only: minted by
/// [`ComponentTree::enter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowId(usize);

impl ComponentTree {
    /// Enter the component `declared` names under the innermost open one, and
    /// say where its launches land.
    ///
    /// Entering a label already seen at this path merges into its row, so a
    /// repeated component is one row entered many times.
    pub(crate) fn enter(&mut self, declared: Declaration<'_>) -> Attribution {
        let row = self.resolve(declared.label);
        self.row_mut(row).enter(declared);
        self.open.push(row);
        Attribution::Component(row)
    }

    /// Leave the innermost open component, and say where launches land now.
    pub(crate) fn exit(&mut self) -> Attribution {
        self.open.pop();
        match self.innermost() {
            Some(row) => Attribution::Component(row),
            None => Attribution::Untagged,
        }
    }

    /// The innermost component still open, if any.
    pub(crate) fn innermost(&self) -> Option<RowId> {
        self.open.last().copied()
    }

    /// Count one launch of `kernel` on the row `placement` puts it on, carrying
    /// out the work of the rows it spans.
    ///
    /// By name and count rather than one row per launch: a 28-layer stack
    /// issues the same kernel 28 times, and 28 identical rows say less than one
    /// row saying 28. Called for every launch, so it allocates only for a
    /// kernel or a spanned row the row has not counted before; the rows it
    /// spans are named once, in [`breakdown`](Self::breakdown).
    pub(crate) fn count_launch(&mut self, placement: &Placement, kernel: &str) {
        self.kernels(placement.attribution).count(kernel, placement);
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
            Attribution::Component(row) => self.row_mut(row).waited += idle,
            Attribution::Untagged | Attribution::EndOfPass => self.waited_outside += idle,
        }
    }

    /// The closest component every one of `rows` sits under, itself included
    /// — where a kernel that carried out all of their work belongs. `None` for
    /// rows under different roots, which share no row.
    pub(crate) fn common_parent(&self, rows: &[RowId]) -> Option<RowId> {
        let (first, rest) = rows.split_first()?;
        let mut common: Vec<RowId> = self.ancestors(*first).collect();
        for row in rest {
            let theirs: Vec<RowId> = self.ancestors(*row).collect();
            common.retain(|ancestor| theirs.contains(ancestor));
        }
        // Each list runs from the row up to its root, so the first ancestor
        // they share is the closest.
        common.first().copied()
    }

    /// Add `extra` to `row` and every parent that declares a floor. A row
    /// with no floor is timed but not scored, and stays that way; no traffic
    /// leaves every row as it was, so a batch of one marks nothing.
    pub(crate) fn add_observed(&mut self, row: RowId, extra: Bounds) {
        if extra == Bounds::default() {
            return;
        }
        let scored: Vec<RowId> = self.ancestors(row).collect();
        for row in scored {
            let row = self.row_mut(row);
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
            untagged: self.kernels_named(&self.untagged, Attribution::Untagged),
            end_of_pass: self.kernels_named(&self.end_of_pass, Attribution::EndOfPass),
            waited_outside: self.waited_outside,
            host_timed: timings.host_timed,
            timed_alone: timings.timed_alone,
            unplaced: timings.unplaced,
            unfused,
        }
    }

    /// The row `label` names under the innermost open component, created on
    /// first sight.
    fn resolve(&mut self, label: &str) -> RowId {
        let parent = self.innermost();
        let siblings = match parent {
            Some(parent) => &self.row(parent).children,
            None => &self.roots,
        };
        if let Some(found) = siblings
            .iter()
            .copied()
            .find(|row| self.row(*row).label == label)
        {
            return found;
        }
        let row = RowId(self.rows.len());
        self.rows.push(RowRecord::new(label, parent));
        match parent {
            Some(parent) => self.row_mut(parent).children.push(row),
            None => self.roots.push(row),
        }
        row
    }

    fn row(&self, row: RowId) -> &RowRecord {
        &self.rows[row.0]
    }

    fn row_mut(&mut self, row: RowId) -> &mut RowRecord {
        &mut self.rows[row.0]
    }

    /// The kernels a launch under `attribution` is counted among.
    fn kernels(&mut self, attribution: Attribution) -> &mut KernelCounter {
        match attribution {
            Attribution::Untagged => &mut self.untagged,
            Attribution::Component(row) => &mut self.row_mut(row).kernels,
            Attribution::EndOfPass => &mut self.end_of_pass,
        }
    }

    /// `row` and every row above it, innermost first.
    fn ancestors(&self, row: RowId) -> impl Iterator<Item = RowId> + '_ {
        std::iter::successors(Some(row), |row| self.row(*row).parent)
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
    fn path_under(&self, row: RowId, under: Attribution) -> RowPath {
        let stop = match under {
            Attribution::Component(under) => Some(under),
            Attribution::Untagged | Attribution::EndOfPass => None,
        };
        let mut labels: Vec<&str> = self
            .ancestors(row)
            .take_while(|ancestor| Some(*ancestor) != stop)
            .map(|ancestor| self.row(ancestor).label.as_str())
            .collect();
        labels.reverse();
        RowPath::new(labels)
    }

    /// The kernels `counter` counted on the row at `under`, each naming the
    /// rows it spans by their path from there.
    fn kernels_named(&self, counter: &KernelCounter, under: Attribution) -> Vec<Kernel> {
        counter.kernels(|row| self.path_under(row, under))
    }

    /// The finished node at `row`, and everything under it.
    fn node(&self, id: RowId) -> Node {
        let row = self.row(id);
        Node {
            label: row.label.clone(),
            variants: row.variants.clone(),
            sites: row.sites.iter().map(|site| site.to_string()).collect(),
            bounds: row.bounds,
            observed: row.observed,
            entries: row.entries,
            children: row.children.iter().map(|child| self.node(*child)).collect(),
            kernels: self.kernels_named(&row.kernels, Attribution::Component(id)),
            waited: row.waited,
        }
    }
}

/// One row while it is being built.
struct RowRecord {
    label: String,
    /// The component it was entered under, or `None` for a root.
    parent: Option<RowId>,
    /// Variants seen under this label, in first-seen order.
    variants: Vec<Variant>,
    /// Distinct call sites that entered this label, in first-seen order.
    sites: Vec<&'static str>,
    bounds: Option<Bounds>,
    /// See [`Node::observed`].
    observed: Option<Bounds>,
    entries: usize,
    children: Vec<RowId>,
    kernels: KernelCounter,
    /// See [`Node::waited`].
    waited: Duration,
}

impl RowRecord {
    fn new(label: &str, parent: Option<RowId>) -> Self {
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
struct KernelCounter(Vec<CountedKernel>);

/// One kernel a row counted, the rows it spans kept as ids until the
/// breakdown names them.
struct CountedKernel {
    name: String,
    launches: usize,
    total: Duration,
    /// The rows below the counting row whose work it carried out, in
    /// first-seen order. See [`Kernel::fuses`].
    fuses: Vec<RowId>,
    /// See [`Kernel::fuses_own`].
    fuses_own: bool,
}

impl KernelCounter {
    /// Count one launch of `name` at `placement`, the row it is counted on.
    fn count(&mut self, name: &str, placement: &Placement) {
        let kernel = match self.0.iter().position(|kernel| kernel.name == name) {
            Some(found) => &mut self.0[found],
            None => {
                self.0.push(CountedKernel {
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
        // The union across launches: one fused kernel serves a norm and a
        // projection in every layer, and the row names both once. The row it
        // lands on is not among what it names: `fuses` names rows below it,
        // and the row's own work is the row's by placement.
        for row in &placement.span {
            if placement.attribution == Attribution::Component(*row) {
                kernel.fuses_own = true;
            } else if !kernel.fuses.contains(row) {
                kernel.fuses.push(*row);
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

    /// The kernels counted, each spanned row named by `path`.
    fn kernels(&self, path: impl Fn(RowId) -> RowPath) -> Vec<Kernel> {
        self.0
            .iter()
            .map(|kernel| Kernel {
                name: kernel.name.clone(),
                launches: kernel.launches,
                total: kernel.total,
                fuses: kernel.fuses.iter().map(|row| path(*row)).collect(),
                fuses_own: kernel.fuses_own,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_component_is_one_row_entered_many_times() {
        let mut components = ComponentTree::default();
        for _ in 0..3 {
            components.enter(declared("block"));
            components.enter(declared("attention"));
            components.exit();
            components.exit();
        }

        // Three layers are one row entered three times, not three rows: a
        // 36-layer stack would otherwise print 36 identical subtrees.
        assert_eq!(
            labels(&finished(&components).tree),
            ["blockx3", "  attentionx3"]
        );
    }

    #[test]
    fn one_row_records_every_execution_that_served_it() {
        let mut components = ComponentTree::default();
        for variant in ["tile", "tile", "tensor-ops", "tile"] {
            components.enter(Declaration {
                variant: Some(variant),
                ..declared("attention core")
            });
            components.exit();
        }

        let tree = finished(&components).tree;
        // Merging is keyed on the label, so a row that routed two ways is still
        // one row — with a histogram saying how the entries split, in the order
        // the executions were first seen.
        assert_eq!(labels(&tree), ["attention corex4"]);
        assert_eq!(
            tree[0]
                .variants
                .iter()
                .map(|variant| (variant.name.as_str(), variant.entries))
                .collect::<Vec<_>>(),
            [("tile", 3), ("tensor-ops", 1)]
        );
    }

    #[test]
    fn a_row_that_declares_no_variant_carries_none() {
        let mut components = ComponentTree::default();
        components.enter(declared("out proj"));
        components.exit();

        // Most rows have no execution to choose between, and an empty histogram
        // is what lets the table print nothing rather than a placeholder.
        assert!(finished(&components).tree[0].variants.is_empty());
    }

    #[test]
    fn siblings_keep_first_seen_order() {
        let mut components = ComponentTree::default();
        components.enter(declared("block"));
        for name in ["attn norm", "attention", "ffn norm", "feed-forward"] {
            components.enter(declared(name));
            components.exit();
        }
        components.exit();

        assert_eq!(
            labels(&finished(&components).tree),
            [
                "blockx1",
                "  attn normx1",
                "  attentionx1",
                "  ffn normx1",
                "  feed-forwardx1"
            ],
            "a breakdown reads in the order the pass ran, not alphabetically"
        );
    }

    #[test]
    fn a_pass_may_enter_several_components_at_top_level() {
        let mut components = ComponentTree::default();
        for name in ["embedding", "stack", "lm head"] {
            components.enter(declared(name));
            components.exit();
        }
        assert_eq!(
            labels(&finished(&components).tree),
            ["embeddingx1", "stackx1", "lm headx1"]
        );
    }

    /// Two entries written in different places under one label are one row,
    /// and the row says it was fed from both.
    ///
    /// The merge is what makes a stack readable and is also how a row comes to
    /// hold two different jobs, so the tree is unchanged and the sites are
    /// recorded for a report to name.
    #[test]
    fn a_row_fed_from_two_sites_records_both() {
        let mut components = ComponentTree::default();
        for site in ["my_model::rope::apply", "my_model::rope::apply_at"] {
            components.enter(Declaration {
                site,
                ..declared("rope")
            });
            components.exit();
        }

        let tree = finished(&components).tree;
        assert_eq!(labels(&tree), ["ropex2"], "one label, so still one row");
        assert_eq!(
            tree[0].sites,
            ["my_model::rope::apply", "my_model::rope::apply_at"],
            "in the order they were first seen"
        );
    }

    /// One site entered many times is one site: what a reader acts on is
    /// *which* places fed a row, and a count would repeat the entry count.
    #[test]
    fn a_site_entered_repeatedly_is_recorded_once() {
        let mut components = ComponentTree::default();
        for _ in 0..36 {
            components.enter(declared("block"));
            components.exit();
        }

        let tree = finished(&components).tree;
        assert_eq!(tree[0].entries, 36);
        assert_eq!(tree[0].sites, ["test::forward"]);
    }

    #[test]
    fn bounds_sum_across_entries() {
        let mut components = ComponentTree::default();
        for _ in 0..4 {
            components.enter(Declaration {
                bounds: Some(Bounds {
                    read: 7,
                    write: 3,
                    flops: 3,
                }),
                ..declared("attention")
            });
            components.exit();
        }
        // A component is legally entered at several shapes in one pass, so the
        // row accounts for all of them rather than asserting they agreed.
        assert_eq!(
            finished(&components).tree[0].bounds,
            Some(Bounds {
                read: 28,
                write: 12,
                flops: 12
            })
        );
    }

    #[test]
    fn a_component_without_bounds_reports_none() {
        let mut components = ComponentTree::default();
        components.enter(declared("swiglu"));
        components.exit();
        assert_eq!(
            finished(&components).tree[0].bounds,
            None,
            "no floor beats a floor no kernel can meet"
        );
    }

    #[test]
    fn a_tree_read_mid_pass_is_a_bug_not_a_partial_result() {
        let mut components = ComponentTree::default();
        components.enter(declared("block"));
        let read =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finished(&components).tree));
        assert!(
            read.is_err(),
            "an open component has kernels yet to launch, so its row is not finished"
        );
    }

    /// Observed traffic lands beside the declared floor of its row and of every
    /// parent that declares one.
    #[test]
    fn observed_traffic_reaches_its_row_and_every_row_around_it() {
        let mut components = ComponentTree::default();
        let floor = Bounds {
            read: 10,
            write: 1,
            flops: 4,
        };
        components.enter(declared("model"));
        components.enter(Declaration {
            bounds: Some(floor),
            ..declared("block")
        });
        let experts = row(components.enter(Declaration {
            bounds: Some(floor),
            ..declared("experts")
        }));
        components.exit();
        components.enter(Declaration {
            bounds: Some(floor),
            ..declared("norm")
        });
        components.exit();
        components.exit();
        components.exit();
        let extra = Bounds {
            read: 400,
            write: 0,
            flops: 0,
        };
        components.add_observed(experts, extra);

        let tree = finished(&components).tree;
        let model = &tree[0];
        let block = &model.children[0];
        let (experts, norm) = (&block.children[0], &block.children[1]);
        assert_eq!(
            experts.bounds,
            Some(floor),
            "the declared floor stays as declared"
        );
        assert_eq!(experts.observed, Some(extra));
        assert_eq!(
            block.observed,
            Some(extra),
            "its parent moved that traffic too"
        );
        assert_eq!(norm.observed, None, "a sibling did not");
        assert_eq!(
            (model.bounds, model.observed),
            (None, None),
            "a row that declares no floor stays unscored"
        );
    }

    /// Two rows under one parent can share a label, and a fused kernel names
    /// which of them it carried out.
    ///
    /// The shape a mixed stack has: one `transformer block` holding a dense
    /// `feed forward › down` and a `sparse feed forward › down`. Named by
    /// label alone the two are one string, so a kernel that fused the sparse
    /// one marks the dense one as fused as well, and the ladder prices its
    /// work against whichever floor it met first.
    #[test]
    fn a_fused_kernel_names_which_of_two_same_named_rows_it_carried() {
        let mut components = ComponentTree::default();
        let block = components.enter(declared("block"));
        components.enter(declared("feed forward"));
        let dense = row(components.enter(declared("down")));
        components.exit();
        components.exit();
        components.enter(declared("sparse feed forward"));
        let sparse = row(components.enter(declared("down")));
        components.exit();
        components.exit();
        components.exit();

        components.count_launch(
            &Placement {
                attribution: block,
                span: vec![dense, sparse],
            },
            "ElemwiseMatmul",
        );

        let tree = finished(&components).tree;
        assert_eq!(
            named(&tree[0].kernels[0].fuses),
            ["feed forward › down", "sparse feed forward › down"],
            "two rows, named apart — by label alone both are `down`, and the \
             union would record one entry for work done in two places"
        );
    }

    /// A kernel on a parent names the rows it carried out by their path from
    /// that parent, so a second `query` under another child of the block is a
    /// different row.
    #[test]
    fn a_fused_kernel_names_rows_by_their_path_from_its_own() {
        let mut components = ComponentTree::default();
        let block = components.enter(declared("block"));
        let norm = row(components.enter(declared("attn norm")));
        components.exit();
        components.enter(declared("attention"));
        let query = row(components.enter(declared("query")));
        components.exit();
        components.exit();
        components.exit();

        components.count_launch(
            &Placement {
                attribution: block,
                span: vec![norm, query],
            },
            "ElemwiseMatmul",
        );

        let tree = finished(&components).tree;
        let block = &tree[0];
        assert_eq!(
            named(&block.kernels[0].fuses),
            ["attn norm", "attention › query"]
        );
        assert!(
            !block.kernels[0].fuses_own,
            "the block registered no work of its own"
        );
        assert!(
            block.children[0].kernels.is_empty(),
            "none of it is the norm's alone"
        );
    }

    /// A kernel carrying a parent's own work and its child's names only the
    /// child: `fuses` names rows below the one it is on, so the parent naming
    /// itself would be an empty path. That it carried the parent's work too is
    /// `fuses_own`, which no name says.
    #[test]
    fn a_parent_fused_with_its_child_names_only_the_child() {
        let mut components = ComponentTree::default();
        let residual = components.enter(declared("residual"));
        let gate = row(components.enter(declared("gate")));
        components.exit();
        components.exit();

        components.count_launch(
            &Placement {
                attribution: residual,
                span: vec![gate, row(residual)],
            },
            "ElemwiseFuse",
        );

        let tree = finished(&components).tree;
        assert_eq!(named(&tree[0].kernels[0].fuses), ["gate"]);
        assert!(tree[0].kernels[0].fuses_own);
        assert!(tree[0].children[0].kernels.is_empty());
    }

    /// Outside every component, a kernel names the rows it carried out by
    /// their path from the root: their rows have no kernel of their own, and
    /// this is where their time went.
    #[test]
    fn a_kernel_outside_every_component_names_rows_from_the_root() {
        let mut components = ComponentTree::default();
        components.enter(declared("transformer"));
        let attention = row(components.enter(declared("attention")));
        components.exit();
        components.exit();
        let backward = row(components.enter(declared("backward")));
        components.exit();

        components.count_launch(
            &Placement {
                attribution: Attribution::Untagged,
                span: vec![attention, backward],
            },
            "ElemwiseFuse",
        );

        let untagged = finished(&components).untagged;
        assert_eq!(
            named(&untagged[0].fuses),
            ["transformer › attention", "backward"]
        );
    }

    /// The sync ending the pass runs fused work the last components left
    /// queued, and its kernels have a row of their own rather than being
    /// taken for work no annotation covers.
    #[test]
    fn the_end_of_the_pass_is_a_row_of_its_own() {
        let mut components = ComponentTree::default();
        components.enter(declared("lm head"));
        components.exit();
        components.count_launch(&Placement::at(Attribution::EndOfPass), "FusedMatmul");

        let breakdown = finished(&components);
        assert_eq!(breakdown.end_of_pass[0].name, "FusedMatmul");
        assert!(breakdown.untagged.is_empty());
    }

    /// The breakdown of a pass whose timings are all trusted.
    fn finished(components: &ComponentTree) -> Breakdown {
        components.breakdown(TimingQuality::default(), false)
    }

    /// The row a component was entered on.
    fn row(entered: Attribution) -> RowId {
        match entered {
            Attribution::Component(row) => row,
            Attribution::Untagged | Attribution::EndOfPass => {
                unreachable!("entering a component lands on its row")
            }
        }
    }

    /// A declaration naming only what a test is about. Everything a test does
    /// not state is what an ordinary component declares: no variant, no floor,
    /// and one site — so a row is a merge here only where a test says so.
    fn declared(label: &str) -> Declaration<'_> {
        Declaration {
            label,
            variant: None,
            site: "test::forward",
            bounds: None,
        }
    }

    /// The paths a kernel names, as a table prints them.
    fn named(fuses: &[RowPath]) -> Vec<String> {
        fuses.iter().map(ToString::to_string).collect()
    }

    fn labels(nodes: &[Node]) -> Vec<String> {
        nodes
            .iter()
            .flat_map(|node| node.walk())
            .map(|(depth, node)| format!("{}{}x{}", "  ".repeat(depth), node.label, node.entries))
            .collect()
    }
}
