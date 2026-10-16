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

use burn_tensor::Device;
use cubecl::profile::{ProfileDuration, ProfileTicks, TimingMethod};

use crate::breakdown::Breakdown;
use crate::cursor::{Attribution, LaunchCursor};
use crate::row_path::RowPath;
use crate::{Bounds, Collector, Declaration, ObservedBounds};

/// One kernel a component launched: what it is called, how many times it ran,
/// and what the device spent on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel {
    /// The name the kernel gives itself, in full — shortened for display only,
    /// so two kernels that abbreviate alike are still counted apart.
    pub name: String,
    /// Launches, across every entry of the component.
    pub launches: usize,
    /// Device time across those launches.
    pub total: Duration,
    /// The components below the row it is recorded on whose work it carried
    /// out, in first-seen order, each by its path from that row — or from the
    /// root, for a kernel outside every component.
    ///
    /// A fused kernel is recorded on the closest common parent of the
    /// components whose work it did, and that row's own work is its by
    /// placement, so a kernel on a parent naming one child did the child's
    /// work and the parent's. Empty for a kernel that did one component's work
    /// on that component's row.
    pub fuses: Vec<RowPath>,
    /// Whether, besides the rows [`fuses`](Self::fuses) names, it carried out
    /// work the row it is recorded on registered itself — a residual add fused
    /// with its child's. That work is priced by no row below, so the kernel
    /// stands at what it took even where every row it names is priced.
    pub fuses_own: bool,
}

/// Which execution served some of a row's entries, and how many.
///
/// A row is merged on its **label**, which is its identity; the variant says
/// which of several executions ran under it. So `attention core ×9` staying one
/// row while eight entries took the tile kernel and one took tensor ops is the
/// intended reading — the alternative, a row per execution, would be two rows
/// whose times a reader has to add back together to get the component's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Variant {
    /// What the execution calls itself.
    pub name: String,
    /// Entries it served, across the whole pass.
    pub entries: usize,
}

/// One component's row: what it declared, and what it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    /// What the component is called — its label, or its type.
    pub label: String,
    /// Which executions served its entries, in first-seen order. Empty for a
    /// component that declares no variant, which is most of them.
    pub variants: Vec<Variant>,
    /// The call sites this row was entered from, in first-seen order.
    ///
    /// One for almost every row. More than one is a **merge**: two entries
    /// written in different places landed on one label, and the row now holds
    /// the sum of both. That is what a reader wants for a stack of layers and is also
    /// how a row comes to hold two different jobs, so the merge stands and a
    /// report names the sites, which is what points at a split when there is
    /// one to make.
    ///
    /// A site is a *function*: one entered many times is one site, so this sees
    /// two entry points and not two roles.
    pub sites: Vec<&'static str>,
    /// The sum of what every entry declared it should cost at best, or `None`
    /// when the component declares none.
    ///
    /// Summed rather than checked for agreement because a component is legally
    /// entered at several shapes in one pass: chunked prefill runs a full chunk
    /// and a padded tail, and both are real work the row has to account for.
    pub bounds: Option<Bounds>,
    /// Traffic observed beyond [`bounds`](Self::bounds) under this component,
    /// such as the experts a sparse layer's routes read past one row's. `None`
    /// where nothing under it observed any, or where it declares no floor.
    pub observed: Option<Bounds>,
    /// How many times the pass entered it.
    pub entries: usize,
    /// The components it entered, in first-seen order.
    pub children: Vec<Node>,
    /// The kernels launched while this component was the innermost one open,
    /// in first-seen order.
    ///
    /// A component with children of its own can still launch kernels directly
    /// — the reshapes and residual adds between them — and those are exactly
    /// the work no child accounts for.
    pub kernels: Vec<Kernel>,
    /// How long the device sat idle, in the collected pass, before the kernels
    /// this component launched directly — waiting on the host to issue them.
    ///
    /// Idle time is charged to the kernel that ended it: that launch is what
    /// the device was waiting for. It is the collected pass's, whose host pays
    /// a round trip per launch to time it, so as an absolute it describes the
    /// instrumentation; as a share of every gap it says which components the
    /// device waited on, which is what a clean pass's host time is split by.
    pub waited: Duration,
}

impl Node {
    /// Every row below this one, depth-first, each with its path from here —
    /// the form a kernel on this row names it by in [`Kernel::fuses`].
    ///
    /// Not [`walk`](Self::walk), which yields a depth and not a path: two rows
    /// at one depth under different parents can share a label, and telling those
    /// apart is the whole reason a kernel records a path.
    pub fn descendants(&self) -> Vec<(RowPath, &Node)> {
        let mut out = Vec::new();
        self.descend(&RowPath::default(), &mut out);
        out
    }

    /// The rows below this one, which sits at `at`, onto `out`. See
    /// [`descendants`](Self::descendants).
    fn descend<'a>(&'a self, at: &RowPath, out: &mut Vec<(RowPath, &'a Node)>) {
        for child in &self.children {
            let path = at.child(&child.label);
            out.push((path.clone(), child));
            child.descend(&path, out);
        }
    }

    /// How long the device waited on the host before every kernel under this
    /// component, its children's included. See [`waited`](Self::waited).
    pub fn waited_total(&self) -> Duration {
        self.waited
            + self
                .children
                .iter()
                .map(|child| child.waited_total())
                .sum::<Duration>()
    }

    /// Device time in the kernels this component launched directly.
    pub fn kernel_time(&self) -> Duration {
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
    pub fn device_time(&self) -> Duration {
        self.kernel_time()
            + self
                .children
                .iter()
                .map(|child| child.device_time())
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

    /// Every node under and including this one, depth-first, with its depth.
    pub fn walk(&self) -> Vec<(usize, &Node)> {
        fn descend<'a>(node: &'a Node, depth: usize, out: &mut Vec<(usize, &'a Node)>) {
            out.push((depth, node));
            for child in &node.children {
                descend(child, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        descend(self, 0, &mut out);
        out
    }
}

/// Builds a [`Node`] tree from the components a pass enters.
///
/// # What the timing is, and is not
///
/// A component has no clock of its own. Nothing drains the device at its
/// boundaries — a drain there would seal every fused segment at the boundary
/// and collect a graph production does not run — so its time is the device
/// time of the kernels recorded under it, each timed by the runtime as it
/// launched.
///
/// A kernel is recorded under the component whose work it carried out. Unfused,
/// that is the component that issued it, which [`LaunchCursor`] keeps true on
/// every launching thread. Fused, the fusion server says which operations each
/// block carries out and the collector knows who registered them
/// ([`block_starts`](Self::block_starts)): a kernel that did one component's
/// work is that component's however late it ran, and one that did several's —
/// a norm fused into the projection after it — is on their closest common
/// parent, named for them. Every kernel is on the one row that holds all it
/// did; outside a block, a launch falls back to the component whose operation
/// sealed the segment.
pub struct TreeCollector {
    device: Device,
    /// Before the pass, so every measurement's window can be placed after it
    /// on one clock and the gaps between them read off.
    epoch: std::time::Instant,
    /// Where each launching thread records, moved at every boundary.
    cursor: LaunchCursor,
    /// A mutex rather than a cell: components are entered on the thread driving
    /// the model and kernels arrive on the runtime's worker, so the two halves
    /// of the tree are written from different threads. The lock is held only
    /// for bookkeeping — never across a device call or a cursor move — so the
    /// launching thread waits on arithmetic at worst.
    state: std::sync::Mutex<State>,
}

/// The arena the tree is built in, and the path currently open.
#[derive(Default)]
struct State {
    nodes: Vec<Entry>,
    /// Indices of the components entered and not yet left, outermost first.
    open: Vec<usize>,
    /// Indices of the roots, in first-seen order — a pass may enter more than
    /// one component at top level (an embedding, then a stack, then a head).
    roots: Vec<usize>,
    /// Kernels launched with no component open at all. A real category: a
    /// cache allocation, a sampler, anything the annotations do not cover.
    untagged: Vec<Kernel>,
    /// Kernels the sync ending the pass ran: fused work still queued when the
    /// last component closed. See [`TreeCollector::finish`].
    end_of_pass: Vec<Kernel>,
    /// Whether any launch was timed on the host rather than the device.
    ///
    /// One is enough to taint the totals: a host timing is wall around a
    /// blocking submit, so it carries submission and sync and is several times
    /// the kernel it names.
    host_timed: bool,
    /// Whether any launch's measurement was read back as it launched, which
    /// runs the kernels one at a time. See [`Breakdown::timed_alone`].
    timed_alone: bool,
    /// Measurements handed over unread, each beside the row its launch was
    /// counted on. Read back by [`TreeCollector::finish`], once the pass has
    /// run.
    kept: Vec<Kept>,
    /// Observed traffic not yet priced, each with the component it arrived
    /// under. Priced in [`TreeCollector::finish`].
    pending: Vec<Pending>,
    /// How long the device waited before kernels no component launched — the
    /// untagged ones and the end of the pass. See [`Node::waited`].
    waited_outside: Duration,
    /// Which component each operation the fusion server registered came from,
    /// by its first output's id — what a block's operations are mapped back
    /// through. See [`TreeCollector::block_starts`].
    registered: std::collections::HashMap<u64, Attribution>,
    /// Kernel time that could not be credited to a row, and how many timings
    /// that was.
    ///
    /// A measurement is filed under the attribution of the thread it arrives
    /// on, and the runtime hands it over on the thread that issued the launch,
    /// inside the same call — so it finds the row its launch was counted on.
    /// Anything else has nowhere to go, nor does a window that comes back with
    /// no measurement in it, and [`Breakdown::device`] is the denominator of
    /// every share, so a silent drop would move every number read off it.
    /// Counted and reported, like a host timing.
    unplaced: Unplaced,
}

/// One kernel's window on the device, placed after the collector's epoch.
struct Window {
    attribution: Attribution,
    start: Duration,
    end: Duration,
}

/// A launch's measurement, not yet read back, and the row it belongs to.
struct Kept {
    attribution: Attribution,
    kernel: &'static str,
    profile: ProfileDuration,
}

/// Observed traffic not yet priced, and the component it arrived under.
struct Pending {
    row: usize,
    bounds: ObservedBounds,
}

/// Kernel time that could not be credited to a row — a timing that reached the
/// collector under a different attribution from the launch it measured, or a
/// window that came back with no measurement in it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unplaced {
    /// How many timings had nowhere to go.
    pub timings: usize,
    /// What they came to.
    pub total: Duration,
}

/// One node while it is being built: a flat arena, because a tree with a cursor
/// into it cannot be expressed with ordinary borrows.
struct Entry {
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
    /// Kernels launched directly under this component, in first-seen order.
    kernels: Vec<Kernel>,
    /// See [`Node::waited`].
    waited: Duration,
}

impl TreeCollector {
    /// A collector for work running on `device`.
    ///
    /// Points every launching thread at no component first: the cursor is a
    /// thread-local, so a worker still pointing into a previous collector's
    /// arena would record this one's first kernels at an index that means
    /// something else here.
    pub fn new(device: &Device) -> Self {
        let cursor = LaunchCursor::new(device);
        cursor.point_at(Attribution::Untagged);
        Self {
            device: device.clone(),
            epoch: std::time::Instant::now(),
            cursor,
            state: std::sync::Mutex::new(State::default()),
        }
    }

    /// End the pass: run the fused work still queued, under a row of its own,
    /// stop attributing, and read back every measurement the pass kept.
    ///
    /// A fused segment runs when an op seals it, and the pass's last one has
    /// no op after it — the output is returned unread — so its kernels would
    /// otherwise launch at whatever sync came next, after the collector had
    /// been taken down, and [`Breakdown::device`] would quietly shrink by them. They
    /// are the work of the last components, but no row can say which, so they
    /// are named for when they ran.
    ///
    /// The measurements are read here rather than as each kernel launched:
    /// reading one waits for its kernel, and waiting after every launch runs a
    /// pass one kernel at a time. After the sync every window has closed, so
    /// reading them costs nothing and each is the kernel as it ran in a queue
    /// kept full. The observed traffic is priced here too, for the same
    /// reason: reading its values back waits on the device.
    ///
    /// # Panics
    ///
    /// If the device stops answering, or the work producing observed values
    /// failed.
    pub fn finish(&self) {
        self.cursor.point_at(Attribution::EndOfPass);
        self.device
            .sync()
            .expect("the device answers at the end of the pass");
        self.cursor.point_at(Attribution::Untagged);

        // Read outside the lock: a window the backend could not close on the
        // device resolves by waiting, and a launch still being recorded would
        // wait behind it.
        let kept = std::mem::take(&mut self.locked().kept);
        let read: Vec<(Attribution, &'static str, Option<ProfileTicks>)> = kept
            .into_iter()
            .map(|kept| {
                let ticks = cubecl::future::block_on(kept.profile.resolve());
                (kept.attribution, kept.kernel, ticks)
            })
            .collect();
        let mut state = self.locked();
        // Every operation of the pass has run, so none is left to map back.
        state.registered.clear();
        let mut windows = Vec::with_capacity(read.len());
        for (attribution, kernel, ticks) in read {
            match ticks {
                Some(ticks) => {
                    state.credit(attribution, kernel, ticks.duration());
                    windows.push(Window {
                        attribution,
                        start: ticks.start_duration_since(self.epoch),
                        end: ticks.end_duration_since(self.epoch),
                    });
                }
                None => state.unplaced.timings += 1,
            }
        }
        state.charge_waits(windows);
        let pending = std::mem::take(&mut state.pending);
        drop(state);

        // Priced outside the lock: each reads its values back.
        let priced: Vec<(usize, Bounds)> = pending
            .into_iter()
            .map(|pending| (pending.row, pending.bounds.resolve()))
            .collect();
        let mut state = self.locked();
        for (row, extra) in priced {
            state.add_observed(row, extra);
        }
    }

    /// What the pass left behind, once [`finish`](Self::finish) has read it
    /// all back.
    ///
    /// # Panics
    ///
    /// If a component is still open — a tree read mid-pass is missing the
    /// kernels its open components have yet to launch.
    pub fn breakdown(&self) -> Breakdown {
        let state = self.locked();
        assert!(
            state.open.is_empty(),
            "a component is still open: read the tree after the pass, not during it"
        );
        Breakdown {
            tree: state.roots.iter().map(|root| state.build(*root)).collect(),
            untagged: state.untagged.clone(),
            end_of_pass: state.end_of_pass.clone(),
            waited_outside: state.waited_outside,
            host_timed: state.host_timed,
            // A host timer syncs at both ends of every window, so a launch it
            // timed ran alone too.
            timed_alone: state.timed_alone || state.host_timed,
            unplaced: state.unplaced,
            // See `LaunchCursor::fused`.
            unfused: !self.cursor.fused(),
        }
    }

    /// The shared state, recovering a lock a panicking pass poisoned: a tree
    /// half-built is still worth reading, and the panic itself is reported by
    /// whoever caught it.
    fn locked(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl State {
    /// `index` named the way a reader of the row the kernel sits on would say
    /// it: the components between the two, then `index` itself.
    ///
    /// A path rather than a bare label, because a label is unique only among
    /// siblings. One `transformer block` in a mixed stack holds a dense
    /// `feed forward › down` and a `sparse feed forward › down`, and a kernel
    /// that named `down` could be either — which marks the wrong row as fused
    /// and prices the ladder off whichever one was found first.
    ///
    /// Relative, since the kernel is already recorded on `under`. A kernel
    /// under no component at all is named from the root.
    fn path_under(&self, index: usize, under: Attribution) -> RowPath {
        let stop = match under {
            Attribution::Component(node) => Some(node),
            Attribution::Untagged | Attribution::EndOfPass => None,
        };
        let mut parts: Vec<&str> = Vec::new();
        let mut at = Some(index);
        while let Some(node) = at {
            if Some(node) == stop {
                break;
            }
            parts.push(self.nodes[node].label.as_str());
            at = self.nodes[node].parent;
        }
        parts.reverse();
        RowPath::new(parts)
    }

    /// The node `label` names under the currently open component, creating it
    /// on first sight so a repeated entry merges into one row.
    fn resolve(&mut self, label: &str) -> usize {
        let siblings = match self.open.last() {
            Some(parent) => &self.nodes[*parent].children,
            None => &self.roots,
        };
        if let Some(found) = siblings
            .iter()
            .copied()
            .find(|index| self.nodes[*index].label == label)
        {
            return found;
        }
        let index = self.nodes.len();
        let parent = self.open.last().copied();
        self.nodes.push(Entry {
            label: label.to_string(),
            parent,
            variants: Vec::new(),
            sites: Vec::new(),
            // Left empty: `enter` sums this entry's bounds into it, and does so
            // over `unwrap_or_default`, so seeding a zero here says nothing.
            bounds: None,
            observed: None,
            entries: 0,
            children: Vec::new(),
            kernels: Vec::new(),
            waited: Duration::ZERO,
        });
        match self.open.last() {
            Some(parent) => self.nodes[*parent].children.push(index),
            None => self.roots.push(index),
        }
        index
    }

    /// Where the innermost open component is, as a launch is attributed.
    fn innermost(&self) -> Attribution {
        match self.open.last() {
            Some(index) => Attribution::Component(*index),
            None => Attribution::Untagged,
        }
    }

    /// The closest component every one of `components` sits under, itself
    /// included — where a kernel that carried out all of their work belongs.
    fn common_parent(&self, components: &[usize]) -> Option<usize> {
        let path = |mut index: usize| {
            let mut path = vec![index];
            while let Some(parent) = self.nodes[index].parent {
                path.push(parent);
                index = parent;
            }
            path
        };
        let (first, rest) = components.split_first()?;
        let mut common = path(*first);
        for component in rest {
            let theirs = path(*component);
            common.retain(|ancestor| theirs.contains(ancestor));
        }
        // Each path runs from the component up to its root, so the first
        // ancestor they share is the closest.
        common.first().copied()
    }

    /// The kernels a launch under `attribution` is counted among.
    fn kernels(&mut self, attribution: Attribution) -> &mut Vec<Kernel> {
        match attribution {
            Attribution::Untagged => &mut self.untagged,
            Attribution::Component(index) => &mut self.nodes[index].kernels,
            Attribution::EndOfPass => &mut self.end_of_pass,
        }
    }

    /// Charge every stretch the device sat idle to the row whose kernel ended
    /// it. See [`Node::waited`].
    ///
    /// In start order rather than launch order: the fusion worker and the
    /// model thread both launch, and the device runs them as they arrive. A
    /// window that starts before the last one ended overlapped it, and the
    /// device was not idle for it at all.
    fn charge_waits(&mut self, mut windows: Vec<Window>) {
        windows.sort_by_key(|window| window.start);
        let mut busy_until: Option<Duration> = None;
        for window in windows {
            if let Some(busy) = busy_until {
                let idle = window.start.saturating_sub(busy);
                match window.attribution {
                    Attribution::Component(index) => self.nodes[index].waited += idle,
                    Attribution::Untagged | Attribution::EndOfPass => {
                        self.waited_outside += idle;
                    }
                }
            }
            busy_until = Some(busy_until.map_or(window.end, |busy| busy.max(window.end)));
        }
    }

    /// Add `duration` to the row that counted `kernel`'s launch under
    /// `attribution`.
    ///
    /// By row, not by name across the tree: every projection of a block can
    /// launch the same kernel, and a search by name alone would credit the
    /// first row it finds with every other row's time.
    fn credit(&mut self, attribution: Attribution, kernel: &str, duration: Duration) {
        let into = self.kernels(attribution);
        match into.iter_mut().find(|counted| counted.name == kernel) {
            Some(counted) => counted.total += duration,
            // No launch of it was counted there, so there is no honest row for
            // it. Say how much rather than let `Breakdown::device` quietly shrink.
            None => {
                self.unplaced.timings += 1;
                self.unplaced.total += duration;
            }
        }
    }

    /// Add `extra` to `row` and every parent that declares a floor. A row
    /// with no floor is timed but not scored, and stays that way; no traffic
    /// leaves every row as it was, so a batch of one marks nothing.
    fn add_observed(&mut self, row: usize, extra: Bounds) {
        if extra == Bounds::default() {
            return;
        }
        let mut at = Some(row);
        while let Some(index) = at {
            let node = &mut self.nodes[index];
            if node.bounds.is_some() {
                node.observed = Some(node.observed.unwrap_or_default() + extra);
            }
            at = node.parent;
        }
    }

    /// The finished node at `index`, and everything under it.
    fn build(&self, index: usize) -> Node {
        let entry = &self.nodes[index];
        Node {
            label: entry.label.clone(),
            variants: entry.variants.clone(),
            sites: entry.sites.clone(),
            bounds: entry.bounds,
            observed: entry.observed,
            entries: entry.entries,
            children: entry
                .children
                .iter()
                .map(|child| self.build(*child))
                .collect(),
            kernels: entry.kernels.clone(),
            waited: entry.waited,
        }
    }
}

impl TreeCollector {
    /// Record a kernel under the calling thread's attribution.
    ///
    /// The calling thread's, not the model thread's open component: under
    /// fusion a kernel launches on the fusion worker, dozens of ops after the
    /// model thread registered its work, and the model's stack by then says
    /// where the model has got to rather than what sealed this kernel.
    ///
    /// By name and count rather than one row per launch: a 28-layer stack
    /// issues the same kernel 28 times, and 28 identical rows say less than
    /// one row saying 28.
    fn record_kernel(&self, name: &str) {
        let attribution = LaunchCursor::current();
        let span = LaunchCursor::span();
        let mut state = self.locked();
        // The row it lands on is not among what it names: `fuses` names rows
        // below it, and the row's own work is the row's by placement.
        let own = |index: &usize| attribution == Attribution::Component(*index);
        let fuses_own = span.iter().any(own);
        let fuses: Vec<RowPath> = span
            .iter()
            .filter(|index| !own(index))
            .map(|index| state.path_under(*index, attribution))
            .collect();
        let into = state.kernels(attribution);
        let kernel = match into.iter().position(|kernel| kernel.name == name) {
            Some(found) => &mut into[found],
            None => {
                into.push(Kernel {
                    name: name.to_string(),
                    launches: 0,
                    total: Duration::ZERO,
                    fuses: Vec::new(),
                    fuses_own: false,
                });
                into.last_mut().expect("pushed right above")
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

    /// The fusion server registered an operation, keyed by its first output's
    /// id: it came from whichever component the server's thread is pointed at
    /// — the one that registered it, since the cursor moves in the same queue
    /// the operations do.
    fn registered(&self, operation: u64) {
        let attribution = LaunchCursor::current();
        self.locked().registered.insert(operation, attribution);
    }

    /// A block is about to run the operations `keys` names: record its
    /// launches under the closest component containing every one of them,
    /// named for the components they came from where there are several.
    ///
    /// This is what the sealing order cannot say. A block runs when some
    /// later operation seals its segment, and it can carry out work several
    /// components registered — so a norm fused into the projection after it is
    /// neither the norm's kernel nor the projection's, and their common parent
    /// is the one row it belongs to whole. An operation that ran unfused is a
    /// block of one, and goes back to the component that registered it however
    /// late it ran.
    ///
    /// A block with work from outside every component is recorded outside
    /// them too, and so is one whose components sit under different roots and
    /// share no row; one whose operations were never seen registering —
    /// queued before the pass — keeps the sealing order's answer.
    fn block_starts(&self, keys: impl Iterator<Item = u64>) {
        let state = self.locked();
        let mut components: Vec<usize> = Vec::new();
        let mut outside = false;
        let mut seen = false;
        for key in keys {
            match state.registered.get(&key) {
                Some(Attribution::Component(index)) => {
                    seen = true;
                    if !components.contains(index) {
                        components.push(*index);
                    }
                }
                Some(_) => {
                    seen = true;
                    outside = true;
                }
                None => {}
            }
        }
        let (attribution, span) = match (seen, outside) {
            (false, _) => (LaunchCursor::current(), Vec::new()),
            // Still named for the components whose work it carried: recorded
            // outside them, it is the only place their time went.
            (true, true) => (Attribution::Untagged, components.clone()),
            (true, false) => {
                let span = match components.len() {
                    1 => Vec::new(),
                    _ => components.clone(),
                };
                match state.common_parent(&components) {
                    Some(common) => (Attribution::Component(common), span),
                    // Components under different roots — a forward and a
                    // backward fused across the step that joins them — share
                    // no row, so the kernel is recorded outside every one of
                    // them, still named for what it did.
                    None => (Attribution::Untagged, span),
                }
            }
        };
        drop(state);
        LaunchCursor::enter_block(attribution, span);
    }

    /// The block that last started has run.
    fn block_ran(&self) {
        LaunchCursor::leave_block();
    }

    /// Keep a launch's measurement unread, beside the row that counted it.
    ///
    /// The runtime hands it over on the thread that issued the launch, inside
    /// the same call, so the calling thread's attribution is still the
    /// launch's. It is read back in [`finish`](Self::finish).
    fn keep(&self, kernel: &'static str, profile: ProfileDuration) {
        let attribution = LaunchCursor::current();
        let mut state = self.locked();
        state.host_timed |= !matches!(profile.timing_method(), TimingMethod::Device);
        state.kept.push(Kept {
            attribution,
            kernel,
            profile,
        });
    }

    /// Add what a launch took, read back as it launched, to the row that
    /// counted it.
    ///
    /// The runtime reads measurements this way while its profiling logger
    /// reads them too, and a kernel read as it launched ran alone — so the
    /// pass is marked [`timed_alone`](Self::timed_alone).
    fn record_duration(&self, kernel: &str, duration: Duration, method: TimingMethod) {
        let attribution = LaunchCursor::current();
        let mut state = self.locked();
        state.host_timed |= !matches!(method, TimingMethod::Device);
        state.timed_alone = true;
        state.credit(attribution, kernel, duration);
    }
}

impl Collector for TreeCollector {
    fn enter(&self, declared: Declaration<'_>) {
        let index = {
            let mut state = self.locked();
            let index = state.resolve(declared.label);
            let node = &mut state.nodes[index];
            node.entries += 1;
            // The set, not the count: what a reader acts on is *which* places
            // fed this row, and a site that fed 36 of 36 entries says the same
            // thing as one that fed 1.
            if !node.sites.contains(&declared.site) {
                node.sites.push(declared.site);
            }
            // Counted per entry, not per row: a component entered 36 times
            // that routed differently on two of them is the case the histogram
            // exists for, and a set would lose exactly that.
            if let Some(variant) = declared.variant {
                match node.variants.iter_mut().find(|seen| seen.name == variant) {
                    Some(seen) => seen.entries += 1,
                    None => node.variants.push(Variant {
                        name: variant.to_string(),
                        entries: 1,
                    }),
                }
            }
            // Summed across entries — see `Node::bounds`. A component that
            // declared them on its first entry and not on a later one keeps the
            // sum of what was declared rather than losing the row's roofline
            // entirely.
            if let Some(bounds) = declared.bounds {
                node.bounds = Some(node.bounds.unwrap_or_default() + bounds);
            }
            state.open.push(index);
            index
        };
        // Outside the lock: a full queue waits for the worker, and the worker
        // may be waiting on the lock to record a launch.
        self.cursor.point_at(Attribution::Component(index));
    }

    fn exit(&self) {
        let back_to = {
            let mut state = self.locked();
            state.open.pop();
            state.innermost()
        };
        self.cursor.point_at(back_to);
    }

    fn observe(&self, bounds: ObservedBounds) {
        let mut state = self.locked();
        // Outside every component there is no floor to add it to.
        if let Some(&row) = state.open.last() {
            state.pending.push(Pending { row, bounds });
        }
    }
}

/// A handle on a collector that is also installed, so a run can read the tree
/// after the pass rather than handing ownership into the thread-local slot and
/// having no way back to it.
pub struct SharedCollector(std::sync::Arc<TreeCollector>);

impl SharedCollector {
    /// A collector for `device`, and a handle on it.
    pub fn new(device: &Device) -> Self {
        Self(std::sync::Arc::new(TreeCollector::new(device)))
    }

    /// Another handle on the same collector — what gets installed.
    pub(crate) fn handle(&self) -> Self {
        Self(std::sync::Arc::clone(&self.0))
    }

    /// Run `pass` collected, and end it: every component it enters arrives
    /// from this thread and every kernel the runtime issues for it from
    /// whichever thread issued it, until the pass's last queued segment has
    /// run and its measurements are read back.
    ///
    /// Collected, the pass runs the graph it runs in production — nothing is
    /// drained at a boundary — so call it on work already synced: an operation
    /// queued before it would be watched too.
    ///
    /// Taken by value: a collector holds one pass, and its breakdown is read
    /// once that pass has ended.
    ///
    /// # Panics
    ///
    /// If another pass is being collected in the process (the runtimes'
    /// observer slots are process-wide), if the device stops answering at the
    /// end of the pass, or if the work producing an observed value failed.
    pub fn collect(self, pass: impl FnOnce()) -> Breakdown {
        // Two installations of one collector: the components arrive on this
        // thread and the kernels on the runtime's worker, so the tree is
        // written from both. Each kernel lands under the attribution of the
        // thread that issued it — see `LaunchCursor`.
        let watching = crate::observer::watch_launches(&self);
        let collecting = crate::collect(Box::new(self.handle()));
        pass();
        core::mem::drop(collecting);
        // While the kernels are still being watched: the pass's last fused
        // segment is still queued, and it runs at the next sync. Every
        // measurement the pass kept is read back there too.
        self.0.finish();
        core::mem::drop(watching);
        self.0.breakdown()
    }
}

impl Collector for SharedCollector {
    fn enter(&self, declared: Declaration<'_>) {
        self.0.enter(declared);
    }
    fn exit(&self) {
        self.0.exit();
    }
    fn observe(&self, bounds: ObservedBounds) {
        self.0.observe(bounds);
    }
}

/// What the launch observer hands over. See [`crate::observer`].
impl SharedCollector {
    /// A kernel was issued. See [`TreeCollector::record_kernel`].
    pub(crate) fn record_kernel(&self, kernel: &str) {
        self.0.record_kernel(kernel);
    }

    /// A launch's measurement, unread. See [`TreeCollector::keep`].
    pub(crate) fn keep(&self, kernel: &'static str, profile: ProfileDuration) {
        self.0.keep(kernel, profile);
    }

    /// The fusion server registered an operation. See
    /// [`TreeCollector::registered`].
    pub(crate) fn registered(&self, operation: u64) {
        self.0.registered(operation);
    }

    /// A fused or unfused block is about to run. See
    /// [`TreeCollector::block_starts`].
    pub(crate) fn block_starts(&self, operations: impl Iterator<Item = u64>) {
        self.0.block_starts(operations);
    }

    /// The block that last started has run. See [`TreeCollector::block_ran`].
    pub(crate) fn block_ran(&self) {
        self.0.block_ran();
    }

    /// A launch's duration, read as it launched. See
    /// [`TreeCollector::record_duration`].
    pub(crate) fn record_duration(&self, kernel: &str, duration: Duration, method: TimingMethod) {
        self.0.record_duration(kernel, duration, method);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector() -> TreeCollector {
        TreeCollector::new(&Device::flex())
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

    #[test]
    fn a_repeated_component_is_one_row_entered_many_times() {
        let collector = collector();
        for _ in 0..3 {
            collector.enter(declared("block"));
            collector.enter(declared("attention"));
            collector.exit();
            collector.exit();
        }

        // Three layers are one row entered three times, not three rows: a
        // 36-layer stack would otherwise print 36 identical subtrees.
        assert_eq!(
            labels(&collector.breakdown().tree),
            ["blockx3", "  attentionx3"]
        );
    }

    /// Observed traffic lands beside the declared floor of its row and of every
    /// parent that declares one.
    #[test]
    fn observed_traffic_reaches_its_row_and_every_row_around_it() {
        let collector = collector();
        let floor = Bounds {
            read: 10,
            write: 1,
            flops: 4,
        };
        collector.enter(declared("model"));
        collector.enter(Declaration {
            bounds: Some(floor),
            ..declared("block")
        });
        collector.enter(Declaration {
            bounds: Some(floor),
            ..declared("experts")
        });
        let named =
            burn_tensor::Tensor::<1, burn_tensor::Int>::from_ints([3, 1, 3, 7], &Device::flex());
        collector.observe(ObservedBounds::new(named, |named| Bounds {
            read: 100 * named.len() as u64,
            write: 0,
            flops: 0,
        }));
        collector.exit();
        collector.enter(Declaration {
            bounds: Some(floor),
            ..declared("norm")
        });
        collector.exit();
        collector.exit();
        collector.exit();
        collector.finish();

        let tree = collector.breakdown().tree;
        let model = &tree[0];
        let block = &model.children[0];
        let (experts, norm) = (&block.children[0], &block.children[1]);
        let extra = Bounds {
            read: 400,
            write: 0,
            flops: 0,
        };
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

    #[test]
    fn one_row_records_every_execution_that_served_it() {
        let collector = collector();
        for variant in ["tile", "tile", "tensor-ops", "tile"] {
            collector.enter(Declaration {
                variant: Some(variant),
                ..declared("attention core")
            });
            collector.exit();
        }

        let tree = collector.breakdown().tree;
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
        let collector = collector();
        collector.enter(declared("out proj"));
        collector.exit();

        // Most rows have no execution to choose between, and an empty histogram
        // is what lets the table print nothing rather than a placeholder.
        assert!(collector.breakdown().tree[0].variants.is_empty());
    }

    #[test]
    fn siblings_keep_first_seen_order() {
        let collector = collector();
        collector.enter(declared("block"));
        for name in ["attn norm", "attention", "ffn norm", "feed-forward"] {
            collector.enter(declared(name));
            collector.exit();
        }
        collector.exit();

        assert_eq!(
            labels(&collector.breakdown().tree),
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
        let collector = collector();
        for name in ["embedding", "stack", "lm head"] {
            collector.enter(declared(name));
            collector.exit();
        }
        assert_eq!(
            labels(&collector.breakdown().tree),
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
        let collector = collector();
        for site in ["my_model::rope::apply", "my_model::rope::apply_at"] {
            collector.enter(Declaration {
                site,
                ..declared("rope")
            });
            collector.exit();
        }

        let tree = collector.breakdown().tree;
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
        let collector = collector();
        for _ in 0..36 {
            collector.enter(declared("block"));
            collector.exit();
        }

        let tree = collector.breakdown().tree;
        assert_eq!(tree[0].entries, 36);
        assert_eq!(tree[0].sites, ["test::forward"]);
    }

    #[test]
    fn bounds_sum_across_entries() {
        let collector = collector();
        for _ in 0..4 {
            collector.enter(Declaration {
                bounds: Some(Bounds {
                    read: 7,
                    write: 3,
                    flops: 3,
                }),
                ..declared("attention")
            });
            collector.exit();
        }
        // A component is legally entered at several shapes in one pass, so the
        // row accounts for all of them rather than asserting they agreed.
        assert_eq!(
            collector.breakdown().tree[0].bounds,
            Some(Bounds {
                read: 28,
                write: 12,
                flops: 12
            })
        );
    }

    #[test]
    fn a_component_without_bounds_reports_none() {
        let collector = collector();
        collector.enter(declared("swiglu"));
        collector.exit();
        assert_eq!(
            collector.breakdown().tree[0].bounds,
            None,
            "no floor beats a floor no kernel can meet"
        );
    }

    /// A launch on the thread driving the model lands under the component that
    /// thread has open, and under none once it has closed — the unfused case,
    /// where every kernel is issued synchronously by the op that needs it.
    #[test]
    fn a_launch_lands_under_the_component_its_thread_has_open() {
        let collector = collector();
        collector.enter(declared("block"));
        collector.enter(declared("attention"));
        collector.record_kernel("Matmul");
        collector.exit();
        collector.record_kernel("Add");
        collector.exit();
        collector.record_kernel("Sample");

        let tree = collector.breakdown().tree;
        let names = |node: &Node| -> Vec<String> {
            node.kernels
                .iter()
                .map(|kernel| kernel.name.clone())
                .collect()
        };
        assert_eq!(names(&tree[0].children[0]), ["Matmul"]);
        assert_eq!(names(&tree[0]), ["Add"], "the block's own, after its child");
        assert_eq!(
            collector
                .locked()
                .untagged
                .iter()
                .map(|kernel| kernel.name.as_str())
                .collect::<Vec<_>>(),
            ["Sample"]
        );
    }

    /// A launch is attributed by the thread that issues it, never by reading
    /// the model thread's stack across threads. A thread the cursor never
    /// reached records nothing under a component, however deep the model
    /// thread is — reading across would credit a fused kernel to wherever the
    /// model had got to by the time it launched.
    #[test]
    fn a_launch_on_a_thread_the_cursor_never_reached_is_untagged() {
        let collector = std::sync::Arc::new(collector());
        collector.enter(declared("attention"));

        let elsewhere = std::sync::Arc::clone(&collector);
        std::thread::spawn(move || elsewhere.record_kernel("Stray"))
            .join()
            .expect("the other thread ran");
        collector.exit();

        assert!(collector.breakdown().tree[0].kernels.is_empty());
        assert_eq!(collector.locked().untagged[0].name, "Stray");
    }

    /// A device measurement of `micros`, as the runtime hands one over: unread.
    fn measured(micros: u64) -> ProfileDuration {
        let start = std::time::Instant::now();
        let end = start + Duration::from_micros(micros);
        ProfileDuration::new_device_time(async move {
            cubecl::profile::ProfileTicks::from_start_end(start, end)
        })
    }

    /// A measurement is kept beside the row its launch was counted on and
    /// read back when the pass ends — so a kernel two rows share by name keeps
    /// each row's time apart, and nothing waited on a kernel while the pass
    /// ran.
    #[test]
    fn a_kept_measurement_lands_on_the_row_that_counted_its_launch() {
        let collector = collector();
        for (label, micros) in [("query", 13), ("key", 5)] {
            collector.enter(declared(label));
            collector.record_kernel("MatmulEntry");
            collector.keep("MatmulEntry", measured(micros));
            collector.exit();
        }
        assert_eq!(
            collector.breakdown().tree[0].kernel_time(),
            Duration::ZERO,
            "nothing is read back while the pass runs"
        );

        collector.finish();
        let tree = collector.breakdown().tree;
        assert_eq!(tree[0].kernel_time(), Duration::from_micros(13));
        assert_eq!(tree[1].kernel_time(), Duration::from_micros(5));
        assert_eq!(collector.locked().unplaced, Unplaced::default());
        assert!(
            !collector.breakdown().timed_alone,
            "every kernel ran in a full queue"
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
        let collector = collector();
        collector.enter(declared("block"));
        collector.enter(declared("feed forward"));
        collector.enter(declared("down"));
        collector.registered(1);
        collector.exit();
        collector.exit();
        collector.enter(declared("sparse feed forward"));
        collector.enter(declared("down"));
        collector.registered(2);

        // One kernel carrying out both `down` rows, so it lands on the block
        // that holds them and names each.
        collector.block_starts([1, 2].into_iter());
        collector.record_kernel("ElemwiseMatmul");
        collector.block_ran();

        collector.exit();
        collector.exit();
        collector.exit();

        let tree = collector.breakdown().tree;
        assert_eq!(
            named(&tree[0].kernels[0].fuses),
            ["feed forward › down", "sparse feed forward › down"],
            "two rows, named apart — by label alone both are `down`, and the \
             union would record one entry for work done in two places"
        );
    }

    /// A block that carried out several components' work lands on their
    /// closest common parent, named for them — the one row that holds all of
    /// it — and the thread goes back to what it was pointed at once it has run.
    #[test]
    fn a_fused_block_lands_on_the_closest_parent_of_its_components() {
        let collector = collector();
        collector.enter(declared("block"));
        collector.enter(declared("attn norm"));
        collector.registered(1);
        collector.exit();
        collector.enter(declared("attention"));
        collector.enter(declared("query"));
        collector.registered(2);

        // Sealed by the query's own op, so the cursor points at the query.
        collector.block_starts([1, 2].into_iter());
        collector.record_kernel("ElemwiseMatmul");
        collector.block_ran();
        collector.record_kernel("Matmul");

        collector.exit();
        collector.exit();
        collector.exit();

        let tree = collector.breakdown().tree;
        let block = &tree[0];
        assert_eq!(block.kernels.len(), 1, "the fused kernel is the block's");
        assert_eq!(block.kernels[0].name, "ElemwiseMatmul");
        assert_eq!(
            named(&block.kernels[0].fuses),
            ["attn norm", "attention › query"],
            "each named the way the block names it, so a second `query`              under another child of the block is a different row"
        );
        assert!(
            !block.kernels[0].fuses_own,
            "the block registered no work of its own"
        );
        assert!(
            block.children[0].kernels.is_empty(),
            "none of it is the norm's alone"
        );
        let query = &block.children[1].children[0];
        assert_eq!(
            query
                .kernels
                .iter()
                .map(|kernel| kernel.name.as_str())
                .collect::<Vec<_>>(),
            ["Matmul"],
            "and the cursor was back on the query once the block had run"
        );
    }

    /// A block with work from outside every component is recorded outside
    /// them, and still names the components it carried: their rows have no
    /// kernel of their own, and this is where their time went.
    #[test]
    fn a_block_with_outside_work_is_recorded_outside_and_named() {
        let collector = collector();
        collector.registered(1);
        collector.enter(declared("inner"));
        collector.registered(2);

        collector.block_starts([1, 2].into_iter());
        collector.record_kernel("ElemwiseFuse");
        collector.block_ran();
        collector.exit();

        let state = collector.locked();
        assert_eq!(state.untagged[0].name, "ElemwiseFuse");
        assert_eq!(named(&state.untagged[0].fuses), ["inner"]);
    }

    /// A block carrying a parent's own work and its child's lands on the
    /// parent, and names only the child: `fuses` names rows below the one it
    /// is on, so the parent naming itself would be an empty path. That it
    /// carried the parent's work too is `fuses_own`, which no name says.
    #[test]
    fn a_parent_fused_with_its_child_names_only_the_child() {
        let collector = collector();
        collector.enter(declared("residual"));
        collector.enter(declared("gate"));
        collector.registered(1);
        collector.exit();
        collector.registered(2);

        collector.block_starts([1, 2].into_iter());
        collector.record_kernel("ElemwiseFuse");
        collector.block_ran();
        collector.exit();

        let tree = collector.breakdown().tree;
        assert_eq!(tree[0].kernels[0].name, "ElemwiseFuse");
        assert_eq!(named(&tree[0].kernels[0].fuses), ["gate"]);
        assert!(tree[0].kernels[0].fuses_own);
        assert!(tree[0].children[0].kernels.is_empty());
    }

    /// Components under different roots share no row — a forward and a
    /// backward, fused across the step that joins them — so a kernel that did
    /// both is recorded outside every component, named for them. It runs on
    /// the fusion worker, where a panic would take the device down with it.
    #[test]
    fn a_block_across_two_roots_is_recorded_outside_them() {
        let collector = collector();
        collector.enter(declared("transformer"));
        collector.registered(1);
        collector.exit();
        collector.enter(declared("backward"));
        collector.registered(2);

        collector.block_starts([1, 2].into_iter());
        collector.record_kernel("ElemwiseFuse");
        collector.block_ran();
        collector.exit();

        let state = collector.locked();
        assert_eq!(state.untagged[0].name, "ElemwiseFuse");
        assert_eq!(named(&state.untagged[0].fuses), ["transformer", "backward"]);
    }

    /// An operation that ran unfused is a block of one, and goes back to the
    /// component that registered it however late something sealed it — the
    /// case sealing order alone got wrong.
    #[test]
    fn an_unfused_operation_goes_back_to_its_component() {
        let collector = collector();
        collector.enter(declared("final norm"));
        collector.registered(7);
        collector.exit();
        collector.enter(declared("lm head"));

        collector.block_starts([7].into_iter());
        collector.record_kernel("CubeRmsNormRoutine");
        collector.block_ran();
        collector.exit();

        let tree = collector.breakdown().tree;
        assert_eq!(tree[0].kernels[0].name, "CubeRmsNormRoutine");
        assert!(tree[0].kernels[0].fuses.is_empty(), "one component's work");
        assert!(tree[1].kernels.is_empty());
    }

    /// The device's idle stretches are charged to the kernel that ended each:
    /// that launch is what it was waiting for. A kernel that starts before
    /// the last one finished overlapped it, and the device waited for nothing.
    #[test]
    fn idle_time_is_charged_to_the_kernel_that_ended_it() {
        let collector = collector();
        for label in ["norm", "matmul"] {
            collector.enter(declared(label));
            collector.exit();
        }
        let micros = Duration::from_micros;
        collector.locked().charge_waits(vec![
            // Out of order on purpose: the device runs them by start time.
            Window {
                attribution: Attribution::Component(1),
                start: micros(15),
                end: micros(20),
            },
            Window {
                attribution: Attribution::Component(0),
                start: micros(0),
                end: micros(10),
            },
            Window {
                attribution: Attribution::Untagged,
                start: micros(18),
                end: micros(30),
            },
            Window {
                attribution: Attribution::Component(0),
                start: micros(40),
                end: micros(50),
            },
        ]);

        let tree = collector.breakdown().tree;
        assert_eq!(
            tree[1].waited,
            micros(5),
            "idle from 10 to 15, before the matmul"
        );
        assert_eq!(
            tree[0].waited,
            micros(10),
            "idle from 30 to 40, before the norm"
        );
        assert_eq!(
            collector.locked().waited_outside,
            Duration::ZERO,
            "the untagged kernel overlapped the matmul"
        );
    }

    /// A duration read as its kernel launched is still credited to its row,
    /// and says the pass ran that kernel alone — which is what stops the
    /// report from subtracting a serialized sum from a pipelined wall.
    #[test]
    fn a_duration_read_at_launch_marks_the_pass_timed_alone() {
        let collector = collector();
        collector.enter(declared("query"));
        collector.record_kernel("MatmulEntry");
        collector.record_duration(
            "MatmulEntry",
            Duration::from_micros(13),
            TimingMethod::Device,
        );
        collector.exit();

        assert_eq!(
            collector.breakdown().tree[0].kernel_time(),
            Duration::from_micros(13)
        );
        assert!(collector.breakdown().timed_alone);
    }

    /// The sync ending the pass runs fused work the last components left
    /// queued, and its kernels have a row of their own rather than being
    /// taken for work no annotation covers.
    #[test]
    fn the_end_of_the_pass_is_a_row_of_its_own() {
        let collector = collector();
        collector.enter(declared("lm head"));
        collector.exit();
        collector.cursor.point_at(Attribution::EndOfPass);
        collector.record_kernel("FusedMatmul");
        collector.cursor.point_at(Attribution::Untagged);

        let state = collector.locked();
        assert_eq!(state.end_of_pass[0].name, "FusedMatmul");
        assert!(state.untagged.is_empty());
    }

    /// Finishing leaves the calling thread attributing nothing, so a launch
    /// after the pass is not counted as part of it.
    #[test]
    fn finishing_stops_attributing() {
        let collector = collector();
        collector.finish();
        assert_eq!(LaunchCursor::current(), Attribution::Untagged);
    }

    #[test]
    fn a_composites_parts_are_summed_for_the_fusion_rung() {
        let node = Node {
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
                Node {
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
                    waited: Duration::ZERO,
                },
                Node {
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
                    waited: Duration::ZERO,
                },
            ],
            kernels: Vec::new(),
            waited: Duration::ZERO,
        };
        // The parts move more than the whole, and must: the difference is the
        // intermediate a perfect fuser never lands.
        let parts = node.parts_bounds().expect("both children declared one");
        assert_eq!(parts.bytes(), 140);
        let own = node.bounds.unwrap();
        assert!(own.moves_within(parts) && own.bytes() < parts.bytes());
    }

    #[test]
    fn a_tree_read_mid_pass_is_a_bug_not_a_partial_result() {
        let collector = collector();
        collector.enter(declared("block"));
        let read =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| collector.breakdown().tree));
        assert!(
            read.is_err(),
            "an open component has kernels yet to launch, so its row is not finished"
        );
    }
}
