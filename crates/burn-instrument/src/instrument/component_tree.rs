//! The tree a measured run leaves behind: which components it entered, how
//! they nested, what each declared it should cost, and how long each took.
//!
//! Nesting is a consequence of composition — a component's forward calls its
//! children's forwards — so nothing here maintains a list of what sits inside
//! what. Entering the same component twice at one path merges: a 36-layer stack
//! is one component entered 36 times, not 36 components.
//!
//! The same merge puts two *different* jobs under one heading when two entry
//! points share a label, and the sum reads exactly like one entry point run
//! twice. So `#[compute]` passes the site — `module_path!()` plus the function —
//! beside the label, and [`Component::sites`] keeps the distinct ones so a report
//! can name a component fed from more than one. A site is a function, not a role: one
//! function entered at two shapes is one site, and a role its caller knows is
//! named by an argument instead.

use std::time::Duration;

use super::cursor::{Attribution, Fusion, Placement};
use super::timing::TimingQuality;
use crate::breakdown::{Breakdown, Component, Kernel, Variant};
use crate::component_path::ComponentPath;
use crate::{Bounds, Declaration};

/// The components a run entered, built while it runs, beside the kernels no
/// component holds.
///
/// A flat arena, because a tree with a cursor into it cannot be expressed with
/// ordinary borrows: a component is known by its [`ComponentId`], which is what
/// [`Attribution::Component`] carries.
#[derive(Default)]
pub struct ComponentTree {
    components: Vec<ComponentRecord>,
    /// The components entered and not yet left, outermost first.
    open: Vec<ComponentId>,
    /// The roots, in first-seen order — a run may enter more than one
    /// component at top level (an embedding, then a stack, then a head).
    roots: Vec<ComponentId>,
    /// Kernels launched with no component open at all. A real category: a
    /// cache allocation, a sampler, anything the annotations do not cover.
    untagged: KernelCounter,
    /// Kernels the sync ending the run ran: fused work still queued when the
    /// last component closed.
    end_of_run: KernelCounter,
    /// How long the device waited before kernels no component launched — the
    /// untagged ones and the end of the run. See [`Component::own_wait`].
    outside_wait: Duration,
}

/// One component of a [`ComponentTree`], valid in that tree only: minted by
/// [`ComponentTree::enter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentId(usize);

impl ComponentTree {
    /// Enter the component `declared` names under the innermost open one, and
    /// say where its launches land.
    ///
    /// Entering a label already seen at this path merges into its component, so a
    /// repeated component is one entered many times.
    pub fn enter(&mut self, declared: Declaration<'_>) -> Attribution {
        let component = self.resolve(declared.label);
        self.component_mut(component).enter(declared);
        self.open.push(component);
        Attribution::Component(component)
    }

    /// Leave the innermost open component, and say where launches land now.
    pub fn exit(&mut self) -> Attribution {
        self.open.pop();
        match self.innermost() {
            Some(component) => Attribution::Component(component),
            None => Attribution::Untagged,
        }
    }

    /// The innermost component still open, if any.
    pub fn innermost(&self) -> Option<ComponentId> {
        self.open.last().copied()
    }

    /// Count one launch of `kernel` on the component `placement` puts it on, carrying
    /// out the work of the components it spans.
    ///
    /// By name and count rather than one component per launch: a 28-layer stack
    /// issues the same kernel 28 times, and 28 identical components say less than one
    /// component saying 28. Called for every launch, so it allocates only for a
    /// kernel or a spanned component the component has not counted before; the components it
    /// spans are named once, in [`breakdown`](Self::breakdown).
    pub fn count_launch(&mut self, placement: &Placement, kernel: &str) {
        self.kernels(placement.attribution).count(kernel, placement);
    }

    /// Add `duration` to the component that counted `kernel`'s launch under
    /// `attribution`, or `false` when no launch of it was counted there.
    pub fn credit(&mut self, attribution: Attribution, kernel: &str, duration: Duration) -> bool {
        self.kernels(attribution).credit(kernel, duration)
    }

    /// Charge `idle` to the component whose kernel ended it. See [`Component::own_wait`].
    pub fn charge_idle(&mut self, attribution: Attribution, idle: Duration) {
        match attribution {
            Attribution::Component(component) => self.component_mut(component).own_wait += idle,
            Attribution::Untagged | Attribution::EndOfRun => self.outside_wait += idle,
        }
    }

    /// The closest component every one of `components` sits under, itself included
    /// — where a kernel that carried out all of their work belongs. `None` for
    /// components under different roots, which share no component.
    #[cfg(feature = "fusion")]
    pub fn common_parent(&self, components: &[ComponentId]) -> Option<ComponentId> {
        let (first, rest) = components.split_first()?;
        let mut common: Vec<ComponentId> = self.ancestors(*first).collect();
        for component in rest {
            let theirs: Vec<ComponentId> = self.ancestors(*component).collect();
            common.retain(|ancestor| theirs.contains(ancestor));
        }
        // Each list runs from the component up to its root, so the first ancestor
        // they share is the closest.
        common.first().copied()
    }

    /// Add `extra` to `component` and every parent that declares a floor. A component
    /// with no floor is timed but not scored, and stays that way; no traffic
    /// leaves every component as it was, so a batch of one marks nothing.
    pub fn add_observed(&mut self, component: ComponentId, extra: Bounds) {
        if extra == Bounds::default() {
            return;
        }
        let scored: Vec<ComponentId> = self.ancestors(component).collect();
        for component in scored {
            let component = self.component_mut(component);
            if component.bounds.is_some() {
                component.observed = Some(component.observed.unwrap_or_default() + extra);
            }
        }
    }

    /// The breakdown these components make, with what the instrument says about its
    /// timings.
    ///
    /// # Panics
    ///
    /// If a component is still open — a tree read mid-run is missing the
    /// kernels its open components have yet to launch.
    pub fn breakdown(&self, timings: TimingQuality, fusion: Fusion) -> Breakdown {
        assert!(
            self.open.is_empty(),
            "a component is still open: read the tree after the run, not during it"
        );
        Breakdown {
            tree: self.roots.iter().map(|root| self.finished(*root)).collect(),
            untagged: self.kernels_named(&self.untagged, Attribution::Untagged),
            end_of_run: self.kernels_named(&self.end_of_run, Attribution::EndOfRun),
            outside_wait: self.outside_wait,
            host_timed: timings.host_timed,
            timed_alone: timings.timed_alone,
            unplaced: timings.unplaced,
            unfused: fusion == Fusion::Unfused,
        }
    }

    /// The component `label` names under the innermost open component, created on
    /// first sight.
    fn resolve(&mut self, label: &str) -> ComponentId {
        let parent = self.innermost();
        let siblings = match parent {
            Some(parent) => &self.component(parent).children,
            None => &self.roots,
        };
        if let Some(found) = siblings
            .iter()
            .copied()
            .find(|component| self.component(*component).label == label)
        {
            return found;
        }
        let component = ComponentId(self.components.len());
        self.components.push(ComponentRecord::new(label, parent));
        match parent {
            Some(parent) => self.component_mut(parent).children.push(component),
            None => self.roots.push(component),
        }
        component
    }

    fn component(&self, component: ComponentId) -> &ComponentRecord {
        &self.components[component.0]
    }

    fn component_mut(&mut self, component: ComponentId) -> &mut ComponentRecord {
        &mut self.components[component.0]
    }

    /// The kernels a launch under `attribution` is counted among.
    fn kernels(&mut self, attribution: Attribution) -> &mut KernelCounter {
        match attribution {
            Attribution::Untagged => &mut self.untagged,
            Attribution::Component(component) => &mut self.component_mut(component).kernels,
            Attribution::EndOfRun => &mut self.end_of_run,
        }
    }

    /// `component` and every component above it, innermost first.
    fn ancestors(&self, component: ComponentId) -> impl Iterator<Item = ComponentId> + '_ {
        std::iter::successors(Some(component), |component| {
            self.component(*component).parent
        })
    }

    /// `component` named the way a reader of the component a kernel sits on would say it:
    /// the components between the two, then `component` itself.
    ///
    /// A path rather than a bare label, because a label is unique only among
    /// siblings. One `transformer block` in a mixed stack holds a dense
    /// `feed forward › down` and a `sparse feed forward › down`, and a kernel
    /// that named `down` could be either — which marks the wrong component as fused
    /// and prices the ladder off whichever one was found first.
    ///
    /// Relative, since the kernel is already recorded on `under`. A kernel
    /// under no component at all is named from the root.
    fn path_under(&self, component: ComponentId, under: Attribution) -> ComponentPath {
        let stop = match under {
            Attribution::Component(under) => Some(under),
            Attribution::Untagged | Attribution::EndOfRun => None,
        };
        let mut labels: Vec<&str> = self
            .ancestors(component)
            .take_while(|ancestor| Some(*ancestor) != stop)
            .map(|ancestor| self.component(ancestor).label.as_str())
            .collect();
        labels.reverse();
        ComponentPath::new(labels)
    }

    /// The kernels `counter` counted on the component at `under`, each naming the
    /// components it spans by their path from there.
    fn kernels_named(&self, counter: &KernelCounter, under: Attribution) -> Vec<Kernel> {
        counter.kernels(|component| self.path_under(component, under))
    }

    /// The finished component at `id`, and everything under it.
    fn finished(&self, id: ComponentId) -> Component {
        let component = self.component(id);
        Component {
            label: component.label.clone(),
            variants: component.variants.clone(),
            sites: component
                .sites
                .iter()
                .map(|site| site.to_string())
                .collect(),
            bounds: component.bounds,
            observed: component.observed,
            entries: component.entries,
            children: component
                .children
                .iter()
                .map(|child| self.finished(*child))
                .collect(),
            kernels: self.kernels_named(&component.kernels, Attribution::Component(id)),
            own_wait: component.own_wait,
        }
    }
}

/// One component while it is being built.
struct ComponentRecord {
    label: String,
    /// The component it was entered under, or `None` for a root.
    parent: Option<ComponentId>,
    /// Variants seen under this label, in first-seen order.
    variants: Vec<Variant>,
    /// Distinct call sites that entered this label, in first-seen order.
    sites: Vec<&'static str>,
    bounds: Option<Bounds>,
    /// See [`Component::observed`].
    observed: Option<Bounds>,
    entries: usize,
    children: Vec<ComponentId>,
    kernels: KernelCounter,
    /// See [`Component::own_wait`].
    own_wait: Duration,
}

impl ComponentRecord {
    fn new(label: &str, parent: Option<ComponentId>) -> Self {
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
            own_wait: Duration::ZERO,
        }
    }

    /// Merge one more entry of this component.
    fn enter(&mut self, declared: Declaration<'_>) {
        self.entries += 1;
        // The set, not the count: what a reader acts on is *which* places fed
        // this component, and a site that fed 36 of 36 entries says the same thing as
        // one that fed 1.
        if !self.sites.contains(&declared.site) {
            self.sites.push(declared.site);
        }
        // Counted per entry, not per component: a component entered 36 times that
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
        // Summed across entries — see `Component::bounds`. A component that declared
        // them on its first entry and not on a later one keeps the sum of what
        // was declared rather than losing the component's roofline entirely.
        if let Some(bounds) = declared.bounds {
            self.bounds = Some(self.bounds.unwrap_or_default() + bounds);
        }
    }
}

/// The kernels one component counted, by name, in first-seen order.
#[derive(Default)]
struct KernelCounter(Vec<CountedKernel>);

/// One kernel a component counted, the components it spans kept as ids until the
/// breakdown names them.
struct CountedKernel {
    name: String,
    launches: usize,
    total: Duration,
    /// The components below the counting component whose work it carried out, in
    /// first-seen order. See [`Kernel::fuses`].
    fuses: Vec<ComponentId>,
    /// See [`Kernel::fuses_own`].
    fuses_own: bool,
}

impl KernelCounter {
    /// Count one launch of `name` at `placement`, the component it is counted on.
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
        // projection in every layer, and the component names both once. The component it
        // lands on is not among what it names: `fuses` names components below it,
        // and the component's own work is the component's by placement.
        for component in &placement.span {
            if placement.attribution == Attribution::Component(*component) {
                kernel.fuses_own = true;
            } else if !kernel.fuses.contains(component) {
                kernel.fuses.push(*component);
            }
        }
    }

    /// Add `duration` to `name`, or `false` when no launch of it was counted
    /// here.
    ///
    /// By component, not by name across the tree: every projection of a block can
    /// launch the same kernel, and a search by name alone would credit the
    /// first component it finds with every other component's time.
    fn credit(&mut self, name: &str, duration: Duration) -> bool {
        match self.0.iter_mut().find(|counted| counted.name == name) {
            Some(counted) => {
                counted.total += duration;
                true
            }
            None => false,
        }
    }

    /// The kernels counted, each spanned component named by `path`.
    fn kernels(&self, path: impl Fn(ComponentId) -> ComponentPath) -> Vec<Kernel> {
        self.0
            .iter()
            .map(|kernel| Kernel {
                name: kernel.name.clone(),
                launches: kernel.launches,
                total: kernel.total,
                fuses: kernel
                    .fuses
                    .iter()
                    .map(|component| path(*component))
                    .collect(),
                fuses_own: kernel.fuses_own,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three layers are one component entered three times, not three components: a 36-layer stack would
    /// otherwise print 36 identical subtrees.
    #[test]
    fn a_repeated_component_is_one_component_entered_many_times() {
        let mut components = ComponentTree::default();
        for _ in 0..3 {
            components.enter(declared("block"));
            components.enter(declared("attention"));
            components.exit();
            components.exit();
        }

        assert_eq!(
            labels(&finished(&components).tree),
            ["blockx3", "  attentionx3"]
        );
    }

    /// Merging is keyed on the label, so a component that routed two ways is still one component, with a
    /// histogram saying how the entries split, in the order the executions were first seen.
    #[test]
    fn one_component_records_every_execution_that_served_it() {
        let mut components = ComponentTree::default();
        for variant in ["tile", "tile", "tensor-ops", "tile"] {
            components.enter(Declaration {
                variant: Some(variant),
                ..declared("attention core")
            });
            components.exit();
        }

        let tree = finished(&components).tree;
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

    /// Most components have no execution to choose between, and an empty histogram is what lets a table
    /// print nothing rather than a placeholder.
    #[test]
    fn a_component_that_declares_no_variant_carries_none() {
        let mut components = ComponentTree::default();
        components.enter(declared("out proj"));
        components.exit();

        assert!(finished(&components).tree[0].variants.is_empty());
    }

    /// A breakdown reads in the order the run ran, not alphabetically.
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
            "a breakdown reads in the order the run ran, not alphabetically"
        );
    }

    /// A run may enter several components at top level — an embedding, a stack, a head — and each
    /// is a root.
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

    /// Two entries written in different places under one label are one component,
    /// and the component says it was fed from both.
    ///
    /// The merge is what makes a stack readable and is also how a component comes to
    /// hold two different jobs, so the tree is unchanged and the sites are
    /// recorded for a report to name.
    #[test]
    fn a_component_fed_from_two_sites_records_both() {
        let mut components = ComponentTree::default();
        for site in ["my_model::rope::apply", "my_model::rope::apply_at"] {
            components.enter(Declaration {
                site,
                ..declared("rope")
            });
            components.exit();
        }

        let tree = finished(&components).tree;
        assert_eq!(
            labels(&tree),
            ["ropex2"],
            "one label, so still one component"
        );
        assert_eq!(
            tree[0].sites,
            ["my_model::rope::apply", "my_model::rope::apply_at"],
            "in the order they were first seen"
        );
    }

    /// One site entered many times is one site: what a reader acts on is
    /// *which* places fed a component, and a count would repeat the entry count.
    /// A site is recorded once however many times it is entered: what a reader acts on is which
    /// places fed a component.
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

    /// A component is legally entered at several shapes in one run, so its component accounts for all of
    /// them rather than asserting they agreed.
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
        assert_eq!(
            finished(&components).tree[0].bounds,
            Some(Bounds {
                read: 28,
                write: 12,
                flops: 12
            })
        );
    }

    /// A component that declares no floor has none, rather than a zero no kernel can meet.
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

    /// Reading the tree with a component still open is a bug, not a partial result: the open
    /// component has kernels yet to launch.
    #[test]
    fn a_tree_read_mid_pass_is_a_bug_not_a_partial_result() {
        let mut components = ComponentTree::default();
        components.enter(declared("block"));
        let read =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finished(&components).tree));
        assert!(
            read.is_err(),
            "an open component has kernels yet to launch, so its component is not finished"
        );
    }

    /// Observed traffic lands beside the declared floor of its component and of every
    /// parent that declares one.
    #[test]
    fn observed_traffic_reaches_its_component_and_every_component_around_it() {
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
        let experts = component(components.enter(Declaration {
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
            "a component that declares no floor stays unscored"
        );
    }

    /// Two components under one parent can share a label, and a fused kernel names
    /// which of them it carried out.
    ///
    /// The shape a mixed stack has: one `transformer block` holding a dense
    /// `feed forward › down` and a `sparse feed forward › down`. Named by
    /// label alone the two are one string, so a kernel that fused the sparse
    /// one marks the dense one as fused as well, and the ladder prices its
    /// work against whichever floor it met first.
    #[test]
    fn a_fused_kernel_names_which_of_two_same_named_components_it_carried() {
        let mut components = ComponentTree::default();
        let block = components.enter(declared("block"));
        components.enter(declared("feed forward"));
        let dense = component(components.enter(declared("down")));
        components.exit();
        components.exit();
        components.enter(declared("sparse feed forward"));
        let sparse = component(components.enter(declared("down")));
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
            "two components, named apart — by label alone both are `down`, and the \
             union would record one entry for work done in two places"
        );
    }

    /// A kernel on a parent names the components it carried out by their path from
    /// that parent, so a second `query` under another child of the block is a
    /// different component.
    #[test]
    fn a_fused_kernel_names_components_by_their_path_from_its_own() {
        let mut components = ComponentTree::default();
        let block = components.enter(declared("block"));
        let norm = component(components.enter(declared("attn norm")));
        components.exit();
        components.enter(declared("attention"));
        let query = component(components.enter(declared("query")));
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
    /// child: `fuses` names components below the one it is on, so the parent naming
    /// itself would be an empty path. That it carried the parent's work too is
    /// `fuses_own`, which no name says.
    #[test]
    fn a_parent_fused_with_its_child_names_only_the_child() {
        let mut components = ComponentTree::default();
        let residual = components.enter(declared("residual"));
        let gate = component(components.enter(declared("gate")));
        components.exit();
        components.exit();

        components.count_launch(
            &Placement {
                attribution: residual,
                span: vec![gate, component(residual)],
            },
            "ElemwiseFuse",
        );

        let tree = finished(&components).tree;
        assert_eq!(named(&tree[0].kernels[0].fuses), ["gate"]);
        assert!(tree[0].kernels[0].fuses_own);
        assert!(tree[0].children[0].kernels.is_empty());
    }

    /// Outside every component, a kernel names the components it carried out by
    /// their path from the root: their components have no kernel of their own, and
    /// this is where their time went.
    #[test]
    fn a_kernel_outside_every_component_names_components_from_the_root() {
        let mut components = ComponentTree::default();
        components.enter(declared("transformer"));
        let attention = component(components.enter(declared("attention")));
        components.exit();
        components.exit();
        let backward = component(components.enter(declared("backward")));
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

    /// The sync ending the run executes fused work the last components left
    /// queued, and its kernels have a component of their own rather than being
    /// taken for work no annotation covers.
    #[test]
    fn the_end_of_the_pass_is_a_component_of_its_own() {
        let mut components = ComponentTree::default();
        components.enter(declared("lm head"));
        components.exit();
        components.count_launch(&Placement::at(Attribution::EndOfRun), "FusedMatmul");

        let breakdown = finished(&components);
        assert_eq!(breakdown.end_of_run[0].name, "FusedMatmul");
        assert!(breakdown.untagged.is_empty());
    }

    /// The breakdown of a run whose timings are all trusted.
    fn finished(components: &ComponentTree) -> Breakdown {
        components.breakdown(TimingQuality::default(), Fusion::Fused)
    }

    /// The id of the component `entered` lands on.
    fn component(entered: Attribution) -> ComponentId {
        match entered {
            Attribution::Component(component) => component,
            Attribution::Untagged | Attribution::EndOfRun => {
                unreachable!("entering a component lands on it")
            }
        }
    }

    /// A declaration naming only what a test is about. Everything a test does
    /// not state is what an ordinary component declares: no variant, no floor,
    /// and one site — so a component is a merge here only where a test says so.
    fn declared(label: &str) -> Declaration<'_> {
        Declaration {
            label,
            variant: None,
            site: "test::forward",
            bounds: None,
        }
    }

    /// The paths a kernel names, as a table prints them.
    fn named(fuses: &[ComponentPath]) -> Vec<String> {
        fuses.iter().map(ToString::to_string).collect()
    }

    fn labels(components: &[Component]) -> Vec<String> {
        components
            .iter()
            .flat_map(|component| component.walk())
            .map(|(depth, component)| {
                format!(
                    "{}{}x{}",
                    "  ".repeat(depth),
                    component.label,
                    component.entries
                )
            })
            .collect()
    }
}
