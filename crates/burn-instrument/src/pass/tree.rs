//! Where every event of a collected pass is recorded: the components it
//! enters, from the thread driving the model, and the kernels, blocks and
//! measurements the runtimes report, from whichever thread issues them.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use burn_tensor::Device;
use cubecl::profile::{ProfileDuration, TimingMethod};

use super::cursor::{Attribution, LaunchCursor};
use super::origins::{BlockWork, OperationOrigins};
use super::row_tree::RowTree;
use super::timing::{Kept, Measured, Timeline, Timings};
use crate::breakdown::Breakdown;
use crate::{Bounds, Collector, Declaration, ObservedBounds};

/// Records a pass's events into its rows.
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
pub(crate) struct TreeCollector {
    device: Device,
    /// Before the pass, so every measurement's window can be placed after it
    /// on one clock and the gaps between them read off.
    epoch: Instant,
    /// Where each launching thread records, moved at every boundary.
    cursor: LaunchCursor,
    /// A mutex rather than a cell: components are entered on the thread driving
    /// the model and kernels arrive on the runtime's worker, so the two halves
    /// of the tree are written from different threads. The lock is held only
    /// for bookkeeping — never across a device call or a cursor move — so the
    /// launching thread waits on arithmetic at worst.
    state: Mutex<State>,
}

/// What the pass has recorded so far.
#[derive(Default)]
struct State {
    rows: RowTree,
    timings: Timings,
    /// What a block's operations are mapped back through. See
    /// [`TreeCollector::block_starts`].
    origins: OperationOrigins,
    /// Observed traffic not yet priced. See [`TreeCollector::price_observed`].
    unpriced: Vec<Unpriced>,
}

/// Observed traffic not yet priced, and the row it arrived under.
struct Unpriced {
    row: usize,
    bounds: ObservedBounds,
}

impl TreeCollector {
    /// A collector for work running on `device`.
    ///
    /// Points every launching thread at no component first: the cursor is a
    /// thread-local, so a worker still pointing into a previous collector's
    /// arena would record this one's first kernels at an index that means
    /// something else here.
    pub(crate) fn new(device: &Device) -> Self {
        let cursor = LaunchCursor::new(device);
        cursor.point_at(Attribution::Untagged);
        Self {
            device: device.clone(),
            epoch: Instant::now(),
            cursor,
            state: Mutex::new(State::default()),
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
    /// kept full. The observed traffic is priced afterwards, by
    /// [`price_observed`](Self::price_observed).
    ///
    /// # Panics
    ///
    /// If the device stops answering.
    pub(crate) fn finish(&self) {
        self.cursor.point_at(Attribution::EndOfPass);
        self.device
            .sync()
            .expect("the device answers at the end of the pass");
        self.cursor.point_at(Attribution::Untagged);

        // Read outside the lock: a window the backend could not close on the
        // device resolves by waiting, and a launch still being recorded would
        // wait behind it.
        let kept = self.locked().timings.take_kept();
        let measured: Vec<Measured> = kept.into_iter().map(Kept::resolve).collect();
        let mut state = self.locked();
        state.origins.clear();
        state.settle(measured, self.epoch);
    }

    /// Price the traffic the pass observed, adding it to the rows it arrived
    /// under.
    ///
    /// Once the launches are no longer watched: reading values back can launch
    /// kernels of its own, such as making a sliced tensor contiguous, and
    /// watched they would be recorded as work the pass did.
    ///
    /// # Panics
    ///
    /// If the work producing an observed value failed.
    pub(crate) fn price_observed(&self) {
        let unpriced = std::mem::take(&mut self.locked().unpriced);
        // Priced outside the lock: each reads its values back.
        let priced: Vec<(usize, Bounds)> = unpriced
            .into_iter()
            .map(|unpriced| (unpriced.row, unpriced.bounds.resolve()))
            .collect();
        let mut state = self.locked();
        for (row, extra) in priced {
            state.rows.add_observed(row, extra);
        }
    }

    /// What the pass left behind, once [`finish`](Self::finish) has read it
    /// all back.
    ///
    /// # Panics
    ///
    /// If a component is still open — a tree read mid-pass is missing the
    /// kernels its open components have yet to launch.
    pub(crate) fn breakdown(&self) -> Breakdown {
        let state = self.locked();
        // See `LaunchCursor::fused`.
        let unfused = !self.cursor.fused();
        state.rows.breakdown(state.timings.quality(), unfused)
    }

    /// Record a kernel where the calling thread's launches land.
    ///
    /// The calling thread's, not the model thread's open component: under
    /// fusion a kernel launches on the fusion worker, dozens of ops after the
    /// model thread registered its work, and the model's stack by then says
    /// where the model has got to rather than what sealed this kernel.
    pub(crate) fn record_kernel(&self, name: &str) {
        let placement = LaunchCursor::placement();
        self.locked().rows.count_launch(&placement, name);
    }

    /// The fusion server registered an operation, keyed by its first output's
    /// id: it came from whichever component the server's thread is pointed at
    /// — the one that registered it, since the cursor moves in the same queue
    /// the operations do.
    pub(crate) fn registered(&self, operation: u64) {
        let origin = LaunchCursor::current();
        self.locked().origins.record(operation, origin);
    }

    /// A block is about to run the operations `keys` names: land its launches
    /// where [`BlockWork::placement`] says.
    ///
    /// A block with work from outside every component is recorded outside
    /// them too, and so is one whose components sit under different roots and
    /// share no row; one whose operations were never seen registering —
    /// queued before the pass — keeps the sealing order's answer.
    pub(crate) fn block_starts(&self, keys: impl Iterator<Item = u64>) {
        let placement = {
            let state = self.locked();
            BlockWork::new(&state.origins, keys).placement(&state.rows)
        };
        LaunchCursor::enter_block(placement.unwrap_or_else(LaunchCursor::placement));
    }

    /// The block that last started has run.
    pub(crate) fn block_ran(&self) {
        LaunchCursor::leave_block();
    }

    /// Keep a launch's measurement unread, beside the row that counted it.
    ///
    /// The runtime hands it over on the thread that issued the launch, inside
    /// the same call, so the calling thread's attribution is still the
    /// launch's. It is read back in [`finish`](Self::finish).
    pub(crate) fn keep(&self, kernel: &'static str, profile: ProfileDuration) {
        let attribution = LaunchCursor::current();
        self.locked().timings.keep(attribution, kernel, profile);
    }

    /// Add what a launch took, read back as it launched, to the row that
    /// counted it.
    ///
    /// The runtime reads measurements this way while its profiling logger
    /// reads them too, and a kernel read as it launched ran alone — so the
    /// pass is marked [`timed_alone`](Breakdown::timed_alone).
    pub(crate) fn record_duration(&self, kernel: &str, duration: Duration, method: TimingMethod) {
        let attribution = LaunchCursor::current();
        let mut state = self.locked();
        state.timings.read_at_launch(method);
        state.credit(attribution, kernel, duration);
    }

    /// The shared state, recovering a lock a panicking pass poisoned: a tree
    /// half-built is still worth reading, and the panic itself is reported by
    /// whoever caught it.
    fn locked(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Collector for TreeCollector {
    fn enter(&self, declared: Declaration<'_>) {
        let entered = self.locked().rows.enter(declared);
        // Outside the lock: a full queue waits for the worker, and the worker
        // may be waiting on the lock to record a launch.
        self.cursor.point_at(entered);
    }

    fn exit(&self) {
        let back_to = self.locked().rows.exit();
        self.cursor.point_at(back_to);
    }

    fn observe(&self, bounds: ObservedBounds) {
        let mut state = self.locked();
        // Outside every component there is no floor to add it to.
        if let Some(row) = state.rows.innermost() {
            state.unpriced.push(Unpriced { row, bounds });
        }
    }
}

impl State {
    /// File every measurement read back under the row that counted its
    /// launch, and charge the device's idle stretches between them.
    fn settle(&mut self, measured: Vec<Measured>, epoch: Instant) {
        let mut timeline = Timeline::new(epoch);
        for measured in measured {
            match measured.ticks {
                Some(ticks) => {
                    self.credit(measured.attribution, measured.kernel, ticks.duration());
                    timeline.push(measured.attribution, &ticks);
                }
                None => self.timings.unplace(Duration::ZERO),
            }
        }
        for (attribution, idle) in timeline.idle() {
            self.rows.charge_idle(attribution, idle);
        }
    }

    /// Add `duration` to the row that counted `kernel`'s launch under
    /// `attribution`.
    fn credit(&mut self, attribution: Attribution, kernel: &str, duration: Duration) {
        if !self.rows.credit(attribution, kernel, duration) {
            // No launch of it was counted there, so there is no honest row
            // for it. Say how much rather than let `Breakdown::device` quietly
            // shrink.
            self.timings.unplace(duration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breakdown::{Node, Unplaced};
    use crate::row_path::RowPath;

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
        collector.price_observed();

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
                .breakdown()
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
        assert_eq!(collector.breakdown().untagged[0].name, "Stray");
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
        assert_eq!(collector.breakdown().unplaced, Unplaced::default());
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

        let untagged = collector.breakdown().untagged;
        assert_eq!(untagged[0].name, "ElemwiseFuse");
        assert_eq!(named(&untagged[0].fuses), ["inner"]);
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

        let untagged = collector.breakdown().untagged;
        assert_eq!(untagged[0].name, "ElemwiseFuse");
        assert_eq!(named(&untagged[0].fuses), ["transformer", "backward"]);
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

        let breakdown = collector.breakdown();
        assert_eq!(breakdown.end_of_pass[0].name, "FusedMatmul");
        assert!(breakdown.untagged.is_empty());
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
