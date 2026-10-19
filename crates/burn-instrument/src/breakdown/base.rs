//! What a collected pass leaves behind: the components it entered, the
//! kernels each launched, and what the collection itself can say about how
//! far those numbers are to be trusted.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::Bounds;
use crate::component_path::ComponentPath;

/// One collected pass, broken down by component.
///
/// What [`PassCollector::collect`] returns: data only, so a breakdown
/// collected from a pass and one written out in a test are the same kind of
/// value.
///
#[cfg_attr(
    feature = "cubecl",
    doc = "[`PassCollector::collect`]: crate::PassCollector::collect"
)]
#[cfg_attr(not(feature = "cubecl"), doc = concat!("[`PassCollector::collect`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.PassCollector.html#method.collect")))]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Breakdown {
    /// The components the collected pass entered.
    pub tree: Vec<Component>,
    /// Kernels the pass launched under no component at all, and fused kernels
    /// that carried out work from outside every component or across two roots,
    /// named for the components they did carry out (see [`Kernel::fuses`]).
    pub untagged: Vec<Kernel>,
    /// Kernels the sync ending the pass ran: fused work the last components
    /// left queued, which no op after them was there to seal. See
    /// [`PassCollector::collect`].
    ///
    #[cfg_attr(
        feature = "cubecl",
        doc = "[`PassCollector::collect`]: crate::PassCollector::collect"
    )]
    #[cfg_attr(not(feature = "cubecl"), doc = concat!("[`PassCollector::collect`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.PassCollector.html#method.collect")))]
    pub end_of_pass: Vec<Kernel>,
    /// How long the device waited on the host before the untagged kernels
    /// and the end of the pass's — the idle time no component is charged with. See
    /// [`Component::own_wait`].
    // Stored as `waited_outside`, the key version 1 files carry.
    #[serde(rename = "waited_outside", with = "nanoseconds")]
    pub outside_wait: Duration,
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
    /// Decides how the components are read: in place, a component's milliseconds are its
    /// kernels' own and the clean pass splits into device and host; alone, the
    /// device times are only good as proportions and are apportioned onto the
    /// clean pass instead.
    pub timed_alone: bool,
    /// Kernel time that reached the recorder under a different component from the
    /// launch it measured, or a measurement that came back empty, so no component
    /// could account for it.
    ///
    /// Each measurement is kept beside the component that counted its launch and read
    /// back once the pass has run, so this is zero unless a runtime hands one
    /// over somewhere else. Reported rather than assumed, because
    /// [`device`](Self::total_time) is the denominator of every apportioned
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
            .flat_map(Component::walk)
            .flat_map(|(_, component)| &component.kernels)
            .chain(&self.untagged)
            .chain(&self.end_of_pass)
    }

    /// The kernels that launched and measured no time at all.
    ///
    /// A launch the device ran took some time, so a zero is a timing the
    /// collection lost rather than a fast kernel — and every component it sits under,
    /// every share and the device total are short by it. Distinct from
    /// [`unplaced`](Self::unplaced), which is time that arrived with no component to
    /// go to: this is a component whose time never arrived.
    pub fn untimed(&self) -> impl Iterator<Item = &Kernel> {
        self.kernels()
            .filter(|kernel| kernel.launches > 0 && kernel.total.is_zero())
    }

    /// What the device spent on this pass: every kernel, summed.
    pub fn total_time(&self) -> Duration {
        self.kernels().map(|kernel| kernel.total).sum()
    }

    /// How long the device waited on the host across the collected pass,
    /// between every pair of kernels: what [`Component::own_wait`] is apportioned
    /// by.
    pub fn total_wait(&self) -> Duration {
        self.tree
            .iter()
            .map(Component::total_wait)
            .sum::<Duration>()
            + self.outside_wait
    }

    /// The components whose work all ran in a fused kernel recorded elsewhere: no
    /// kernel under them, and named by a kernel on a component above them or outside
    /// every component. Such a component ran nothing, so its floor is the kernel's
    /// that carried its work, and a report sets the component's own aside.
    ///
    /// Asked of the kernels rather than inferred from a zero: a component that
    /// registered no device work — views and reshapes — also has no kernel, and
    /// its zero is its time; and a component whose own kernels lost their timings
    /// also reads zero, though it ran them.
    pub fn fused_elsewhere(&self) -> Vec<&Component> {
        fn visit<'a>(
            breakdown: &'a Breakdown,
            component: &'a Component,
            ancestors: &mut Vec<&'a Component>,
            out: &mut Vec<&'a Component>,
        ) {
            let named_above = ancestors.iter().enumerate().any(|(at, ancestor)| {
                // What this ancestor's kernels call `component`: the components between
                // the two, then `component`. A bare label would also match a same-named
                // component under another branch of the same ancestor, and a mixed
                // stack's dense and sparse `down` are both `down`.
                let path = ComponentPath::new(
                    ancestors[at + 1..]
                        .iter()
                        .chain([&component])
                        .map(|component| component.label.as_str()),
                );
                ancestor
                    .kernels
                    .iter()
                    .any(|kernel| kernel.fuses.contains(&path))
            });
            // Outside every component, a kernel names a component by its path from the
            // root.
            let path = ComponentPath::new(
                ancestors
                    .iter()
                    .chain([&component])
                    .map(|component| component.label.as_str()),
            );
            let named_outside = breakdown
                .untagged
                .iter()
                .chain(&breakdown.end_of_pass)
                .any(|kernel| kernel.fuses.contains(&path));
            let ran_nothing = component
                .walk()
                .iter()
                .all(|(_, component)| component.kernels.is_empty());
            if ran_nothing && (named_above || named_outside) {
                out.push(component);
            }
            ancestors.push(component);
            for child in &component.children {
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

/// One kernel a component launched: what it is called, how many times it ran,
/// and what the device spent on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kernel {
    /// The name the kernel gives itself, in full — shortened for display only,
    /// so two kernels that abbreviate alike are still counted apart.
    pub name: String,
    /// Launches, across every entry of the component.
    pub launches: usize,
    /// Device time across those launches.
    #[serde(with = "nanoseconds")]
    pub total: Duration,
    /// The components below the component it is recorded on whose work it carried
    /// out, in first-seen order, each by its path from that component — or from the
    /// root, for a kernel outside every component.
    ///
    /// A fused kernel is recorded on the closest common parent of the
    /// components whose work it did, and that component's own work is its by
    /// placement, so a kernel on a parent naming one child did the child's
    /// work and the parent's. Empty for a kernel that did one component's work
    /// on that component.
    pub fuses: Vec<ComponentPath>,
    /// Whether, besides the components [`fuses`](Self::fuses) names, it carried out
    /// work the component it is recorded on registered itself — a residual add fused
    /// with its child's. That work is priced by no component below, so the kernel
    /// stands at what it took even where every component it names is priced.
    pub fuses_own: bool,
}

/// Which execution served some of a component's entries, and how many.
///
/// A component is merged on its **label**, which is its identity; the variant says
/// which of several executions ran under it. So `attention core ×9` staying one
/// component while eight entries took the tile kernel and one took tensor ops is the
/// intended reading — the alternative, a component per execution, would be two components
/// whose times a reader has to add back together to get the component's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Variant {
    /// What the execution calls itself.
    pub name: String,
    /// Entries it served, across the whole pass.
    pub entries: usize,
}

/// One component as the pass measured it: what it declared, and what it took.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Component {
    /// What the component is called — its label, or its type.
    pub label: String,
    /// Which executions served its entries, in first-seen order. Empty for a
    /// component that declares no variant, which is most of them.
    pub variants: Vec<Variant>,
    /// The call sites this component was entered from, in first-seen order.
    ///
    /// One for almost every component. More than one is a **merge**: two entries
    /// written in different places landed on one label, and the component now holds
    /// the sum of both. That is what a reader wants for a stack of layers and is also
    /// how a component comes to hold two different jobs, so the merge stands and a
    /// report names the sites, which is what points at a split when there is
    /// one to make.
    ///
    /// A site is a *function*: one entered many times is one site, so this sees
    /// two entry points and not two roles.
    pub sites: Vec<String>,
    /// The sum of what every entry declared it should cost at best, or `None`
    /// when the component declares none.
    ///
    /// Summed rather than checked for agreement because a component is legally
    /// entered at several shapes in one pass: chunked prefill runs a full chunk
    /// and a padded tail, and both are real work the component has to account for.
    pub bounds: Option<Bounds>,
    /// Traffic observed beyond [`bounds`](Self::bounds) under this component,
    /// such as the experts a sparse layer's routes read past one component's. `None`
    /// where nothing under it observed any, or where it declares no floor.
    pub observed: Option<Bounds>,
    /// How many times the pass entered it.
    pub entries: usize,
    /// The components it entered, in first-seen order.
    pub children: Vec<Component>,
    /// The kernels recorded on this component, in first-seen order: its own work, and
    /// fused kernels that carried out several of its descendants' work, which
    /// land on their closest common parent (see [`Kernel::fuses`]).
    ///
    /// A component with children of its own can still launch kernels directly
    /// — the reshapes and residual adds between them — and those are exactly
    /// the work no child accounts for.
    pub kernels: Vec<Kernel>,
    /// How long the device sat idle, in the collected pass, before the kernels
    /// this component launched directly — waiting on the host to issue them.
    ///
    /// Idle time is charged to the kernel that ended it: that launch is what
    /// the device was waiting for. It is the collected pass's, whose host also
    /// records every launch, so as an absolute it includes the instrumentation;
    /// as a share of every gap it says which components the device waited on,
    /// which is what a clean pass's host time is split by.
    // Stored as `waited`, the key version 1 files carry.
    #[serde(rename = "waited", with = "nanoseconds")]
    pub own_wait: Duration,
}

impl Component {
    /// Every component below this one, depth-first, each with its path from here —
    /// the form a kernel on this component names it by in [`Kernel::fuses`].
    ///
    /// Not [`walk`](Self::walk), which yields a depth and not a path: two components
    /// at one depth under different parents can share a label, and telling those
    /// apart is the whole reason a kernel records a path.
    pub fn descendants(&self) -> Vec<(ComponentPath, &Component)> {
        let mut out = Vec::new();
        self.descend(&ComponentPath::default(), &mut out);
        out
    }

    /// The components below this one, which sits at `at`, onto `out`. See
    /// [`descendants`](Self::descendants).
    fn descend<'a>(&'a self, at: &ComponentPath, out: &mut Vec<(ComponentPath, &'a Component)>) {
        for child in &self.children {
            let path = at.child(&child.label);
            out.push((path.clone(), child));
            child.descend(&path, out);
        }
    }

    /// How long the device waited on the host before every kernel under this
    /// component, its children's included. See [`own_wait`](Self::own_wait).
    pub fn total_wait(&self) -> Duration {
        self.own_wait
            + self
                .children
                .iter()
                .map(|child| child.total_wait())
                .sum::<Duration>()
    }

    /// Device time in the kernels this component launched directly.
    pub fn own_time(&self) -> Duration {
        self.kernels.iter().map(|kernel| kernel.total).sum()
    }

    /// Device time in every kernel under this component, its children's
    /// included.
    ///
    /// **This is the only time a component has.** Nothing drains at its
    /// boundaries, so a clock around its host call would measure how fast the
    /// host registered ops, not what the device spent. The kernels are what the
    /// device actually ran, and their sum across a pass comes out within a few
    /// percent of the clean pass's wall time, which is the cross-check that says
    /// so.
    ///
    /// Zero for a component whose work all fused with another's: the kernel
    /// that did both is on their closest common parent, and counted there.
    pub fn total_time(&self) -> Duration {
        self.own_time()
            + self
                .children
                .iter()
                .map(|child| child.total_time())
                .sum::<Duration>()
    }

    /// The sum of this component's children's bounds, which is what it would
    /// cost with no fusion between them.
    ///
    /// A composite's own bounds assume perfect fusion *within itself* and so
    /// sits strictly below this. The gap is the intermediate traffic a perfect
    /// fuser would delete — the rung of the ladder that no faster kernel can
    /// recover.
    pub fn parts_bounds(&self) -> Option<Bounds> {
        let parts: Vec<Bounds> = self
            .children
            .iter()
            .filter_map(|child| child.bounds)
            .collect();
        (!parts.is_empty()).then(|| parts.into_iter().sum())
    }

    /// Every component under and including this one, depth-first, with its depth.
    pub fn walk(&self) -> Vec<(usize, &Component)> {
        fn descend<'a>(
            component: &'a Component,
            depth: usize,
            out: &mut Vec<(usize, &'a Component)>,
        ) {
            out.push((depth, component));
            for child in &component.children {
                descend(child, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        descend(self, 0, &mut out);
        out
    }
}

/// Kernel time that could not be credited to a component — a timing that reached the
/// recorder under a different attribution from the launch it measured, or a
/// window that came back with no measurement in it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unplaced {
    /// How many timings had nowhere to go.
    pub timings: usize,
    /// What they came to.
    #[serde(with = "nanoseconds")]
    pub total: Duration,
}

/// A [`Duration`] as a whole number of nanoseconds, which any reader of the file can use.
mod nanoseconds {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        // Saturating: a pass would have to run for 584 years to overflow.
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        serializer.serialize_u64(nanoseconds)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        u64::deserialize(deserializer).map(Duration::from_nanos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A composite's parts move more than the whole, and must: the difference is the intermediate a
    /// perfect fuser never lands, which is the rung of the ladder only fusion recovers.
    #[test]
    fn a_composites_parts_are_summed_for_the_fusion_rung() {
        let component = Component {
            variants: Vec::new(),
            sites: Vec::new(),
            label: "feed-forward".into(),
            bounds: Some(Bounds {
                read: 75,
                write: 25,
                flops: 60,
            }),
            observed: None,
            entries: 1,
            children: vec![
                Component {
                    variants: Vec::new(),
                    sites: Vec::new(),
                    label: "gate/up".into(),
                    bounds: Some(Bounds {
                        read: 60,
                        write: 20,
                        flops: 40,
                    }),
                    observed: None,
                    entries: 1,
                    children: Vec::new(),
                    kernels: Vec::new(),
                    own_wait: Duration::ZERO,
                },
                Component {
                    variants: Vec::new(),
                    sites: Vec::new(),
                    label: "down".into(),
                    bounds: Some(Bounds {
                        read: 45,
                        write: 15,
                        flops: 20,
                    }),
                    observed: None,
                    entries: 1,
                    children: Vec::new(),
                    kernels: Vec::new(),
                    own_wait: Duration::ZERO,
                },
            ],
            kernels: Vec::new(),
            own_wait: Duration::ZERO,
        };
        let parts = component
            .parts_bounds()
            .expect("both children declared one");
        assert_eq!(parts.bytes(), 140);
        let own = component.bounds.unwrap();
        assert!(own.moves_within(parts) && own.bytes() < parts.bytes());
    }

    /// A component is fused elsewhere only when it ran no kernel of its own: one
    /// whose kernels lost their timings reads zero too, and setting its floor
    /// aside would hide the work it did.
    #[test]
    fn a_component_fused_elsewhere_ran_no_kernel_of_its_own() {
        let kernel = |name: &str, total: u64, fuses: &[&str]| Kernel {
            name: name.to_string(),
            launches: 1,
            total: Duration::from_nanos(total),
            fuses: fuses
                .iter()
                .map(|label| ComponentPath::new([*label]))
                .collect(),
            fuses_own: false,
        };
        let leaf = |label: &str, kernels: Vec<Kernel>| Component {
            label: label.to_string(),
            variants: Vec::new(),
            sites: Vec::new(),
            bounds: None,
            observed: None,
            entries: 1,
            children: Vec::new(),
            kernels,
            own_wait: Duration::ZERO,
        };
        let block = Component {
            children: vec![
                leaf("norm", Vec::new()),
                // Ran a kernel of its own whose timing was lost.
                leaf("query", vec![kernel("Matmul", 0, &[])]),
            ],
            kernels: vec![kernel("ElemwiseMatmul", 10, &["norm", "query"])],
            ..leaf("block", Vec::new())
        };
        let breakdown = Breakdown {
            tree: vec![block],
            untagged: Vec::new(),
            end_of_pass: Vec::new(),
            outside_wait: Duration::ZERO,
            host_timed: false,
            timed_alone: false,
            unplaced: Unplaced::default(),
            unfused: false,
        };

        let fused: Vec<&str> = breakdown
            .fused_elsewhere()
            .iter()
            .map(|component| component.label.as_str())
            .collect();
        assert_eq!(fused, ["norm"]);
    }
}
