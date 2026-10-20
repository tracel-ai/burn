//! Where every event of an instrumented run is recorded: the components it
//! enters, from the thread driving the model, and the kernels, blocks and
//! measurements the runtimes report, from whichever thread issues them.
//!
//! The runtime calls a [`LaunchObserver`] synchronously, on the thread that
//! issued the launch, before it is submitted. The recorder is the observer,
//! and records each launch under that thread's own attribution — moved in
//! step with the ops the thread runs, and inside a fusion block set to the
//! components whose work the block carries out, with nothing drained to make
//! it so → [`LaunchCursor`].
//!
//! # Measurements are read once the run is over
//!
//! Each launch's measurement arrives unread and the recorder keeps it, reading
//! them all back after the run's last sync. Reading one waits for its kernel,
//! so reading each as it arrived would execute the run one kernel at a time — the
//! kernels would each be timed alone, and their sum would say nothing about a
//! run that keeps the device fed.
//!
//! # The fusion server says whose work a block is
//!
//! The recorder watches the fusion server too: every operation it registers,
//! and every block of them it runs. It maps each operation to the component
//! that registered it and, while a block runs, records its launches where
//! [`BlockOrigin`] places them. Which operations count is
//! [`OperationId::new`]'s to say.

use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[cfg(feature = "fusion")]
use burn_fusion::observer::{BlockOperations, FusionObserver};
#[cfg(feature = "fusion")]
use burn_ir::OperationIr;
use burn_tensor::Device;
use cubecl::logging::{LaunchObserver, TimingRequest};
use cubecl::profile::{ProfileDuration, TimingMethod};

use super::component_tree::{ComponentId, ComponentTree};
use super::cursor::{Attribution, LaunchCursor};
#[cfg(feature = "fusion")]
use super::origins::{BlockOrigin, OperationId, OperationOrigins};
use super::timing::{Kept, Measured, Timeline, Timings};
use super::watching::Watching;
use crate::breakdown::Breakdown;
use crate::{Bounds, ComponentObserver, Declaration, ObservedBounds};

/// Records a run's events into its component tree.
///
/// # What the timing is, and is not
///
/// A component has no clock of its own. Nothing drains the device at its
/// boundaries — a drain there would seal every fused segment at the boundary
/// and record a graph production does not run — so its time is the device
/// time of the kernels recorded under it, each timed by the runtime as it
/// launched.
///
/// A kernel is recorded under the component whose work it carried out. Unfused,
/// that is the component that issued it, which [`LaunchCursor`] keeps true on
/// every launching thread. Fused, the fusion server says which operations each
/// block carries out and the recorder knows who registered them: a kernel that
/// did one component's work is that component's however late it ran, and one
/// that did several's — a norm fused into the projection after it — is on
/// their closest common parent, named for them. Every kernel is on the one component
/// that holds all it did; outside a block, a launch falls back to the
/// component whose operation sealed the segment.
pub struct RunRecorder {
    device: Device,
    /// Before the run, so every measurement's window can be placed after it
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

/// What the run has recorded so far.
#[derive(Default)]
struct State {
    tree: ComponentTree,
    timings: Timings,
    /// What a block's operations are mapped back through.
    #[cfg(feature = "fusion")]
    origins: OperationOrigins,
    /// Observed traffic not yet priced. See [`RunRecorder::price_observed`].
    unpriced: Vec<Unpriced>,
}

/// Observed traffic not yet priced, and the component it arrived under.
struct Unpriced {
    component: ComponentId,
    bounds: ObservedBounds,
}

impl RunRecorder {
    /// A recorder for work running on `device`.
    ///
    /// Points every launching thread at no component first: the cursor is a
    /// thread-local, so a worker still pointing into a previous recorder's
    /// arena would record this one's first kernels at an index that means
    /// something else here.
    pub fn new(device: &Device) -> Self {
        let cursor = LaunchCursor::new(device);
        cursor.point_at(Attribution::Untagged);
        Self {
            device: device.clone(),
            epoch: Instant::now(),
            cursor,
            state: Mutex::new(State::default()),
        }
    }

    /// End the run and say what it left behind.
    ///
    /// In this order, which is what makes the breakdown the run's alone: the
    /// work still queued runs while the launches are watched, so it is
    /// counted; the observers come out, with `watching`; and only then are
    /// observed values read back, since reading them launches kernels of its
    /// own that no component did.
    ///
    /// # Panics
    ///
    /// If the device stops answering, if the work producing an observed value
    /// failed, or if a component is still open.
    pub fn finish(&self, watching: Watching) -> Breakdown {
        self.end_run();
        drop(watching);
        self.price_observed();
        self.breakdown()
    }

    /// Run the fused work still queued, under a component of its own, stop
    /// attributing, and read back every measurement the run kept.
    ///
    /// A fused segment runs when an op seals it, and the run's last one has
    /// no op after it — the output is returned unread — so its kernels would
    /// otherwise launch at whatever sync came next, after the recorder had
    /// been taken down, and [`Breakdown::total_time`] would quietly shrink by them. They
    /// are the work of the last components, but no component can say which, so they
    /// are named for when they ran.
    ///
    /// The measurements are read here rather than as each kernel launched:
    /// reading one waits for its kernel, and waiting after every launch
    /// executes a run one kernel at a time. After the sync every window has closed, so
    /// reading them costs nothing and each is the kernel as it ran in a queue
    /// kept full. The observed traffic is priced afterwards, by
    /// [`price_observed`](Self::price_observed).
    ///
    /// # Panics
    ///
    /// If the device stops answering.
    fn end_run(&self) {
        self.cursor.point_at(Attribution::EndOfRun);
        self.device
            .sync()
            .expect("the device answers at the end of the run");
        self.cursor.point_at(Attribution::Untagged);

        // Read outside the lock: a window the backend could not close on the
        // device resolves by waiting, and a launch still being recorded would
        // wait behind it.
        let kept = self.locked().timings.take_kept();
        let measured: Vec<Measured> = kept.into_iter().map(Kept::resolve).collect();
        let mut state = self.locked();
        #[cfg(feature = "fusion")]
        state.origins.clear();
        state.settle(measured, self.epoch);
    }

    /// Price the traffic the run observed, adding it to the components it arrived
    /// under.
    ///
    /// Once the launches are no longer watched: reading values back can launch
    /// kernels of its own, such as making a sliced tensor contiguous, and
    /// watched they would be recorded as work the run did.
    ///
    /// # Panics
    ///
    /// If the work producing an observed value failed.
    fn price_observed(&self) {
        let unpriced = std::mem::take(&mut self.locked().unpriced);
        // Priced outside the lock: each reads its values back.
        let priced: Vec<(ComponentId, Bounds)> = unpriced
            .into_iter()
            .map(|unpriced| (unpriced.component, unpriced.bounds.resolve()))
            .collect();
        let mut state = self.locked();
        for (component, extra) in priced {
            state.tree.add_observed(component, extra);
        }
    }

    /// What the run left behind.
    ///
    /// # Panics
    ///
    /// If a component is still open — a tree read mid-run is missing the
    /// kernels its open components have yet to launch.
    fn breakdown(&self) -> Breakdown {
        let state = self.locked();
        state
            .tree
            .breakdown(state.timings.quality(), self.cursor.fusion())
    }

    /// The shared state, recovering a lock a panicking run poisoned: a tree
    /// half-built is still worth reading, and the panic itself is reported by
    /// whoever caught it.
    fn locked(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl ComponentObserver for RunRecorder {
    fn enter(&self, declared: Declaration<'_>) {
        let entered = self.locked().tree.enter(declared);
        // Outside the lock: a full queue waits for the worker, and the worker
        // may be waiting on the lock to record a launch.
        self.cursor.point_at(entered);
    }

    fn exit(&self) {
        let back_to = self.locked().tree.exit();
        self.cursor.point_at(back_to);
    }

    fn observe(&self, bounds: ObservedBounds) {
        let mut state = self.locked();
        // Outside every component there is no floor to add it to.
        if let Some(component) = state.tree.innermost() {
            state.unpriced.push(Unpriced { component, bounds });
        }
    }
}

/// Every launch is recorded where the thread that issued it lands its
/// launches, never where the model thread has got to: under fusion a kernel
/// launches on the fusion worker, dozens of ops after the model thread
/// registered its work.
impl LaunchObserver for RunRecorder {
    fn launched(&self, kernel: &'static str) {
        LaunchCursor::with_placement(|placement| {
            self.locked().tree.count_launch(placement, kernel);
        });
    }

    /// Timed, and handed over unread — a breakdown whose kernel components carried
    /// no time would say which kernels ran and nothing about which of them
    /// mattered, and reading each measurement at its launch waits for that
    /// kernel, which executes the run one kernel at a time.
    fn timing(&self) -> TimingRequest {
        TimingRequest::Deferred
    }

    /// Kept beside the component that counted the launch, and read back in
    /// [`finish`](RunRecorder::finish). The runtime hands it over on the
    /// thread that issued the launch, inside the same call, so the calling
    /// thread's attribution is still the launch's.
    fn profiled(&self, kernel: &'static str, profile: ProfileDuration) {
        let attribution = LaunchCursor::current();
        self.locked().timings.keep(attribution, kernel, profile);
    }

    /// Only while the runtime's profiling logger reads measurements too, which
    /// reads each at its launch and hands the observer the duration. A kernel
    /// read as it launched ran alone, so the run is marked
    /// [`timed_alone`](Breakdown::timed_alone).
    fn timed(&self, kernel: &'static str, duration: Duration, method: TimingMethod) {
        let attribution = LaunchCursor::current();
        let mut state = self.locked();
        state.timings.read_at_launch(method);
        state.credit(attribution, kernel, duration);
    }
}

#[cfg(feature = "fusion")]
impl FusionObserver for RunRecorder {
    /// Recorded under whichever component the server's thread is pointed at —
    /// the one that registered it, since the cursor moves in the same queue
    /// the operations do.
    fn registered(&self, operation: &OperationIr) {
        if let Some(operation) = OperationId::new(operation) {
            let origin = LaunchCursor::current();
            self.locked().origins.record(operation, origin);
        }
    }

    /// The block's launches land where [`BlockOrigin::placement`] says, or,
    /// for operations never seen registering, where the sealing order put the
    /// thread.
    fn block_starts(&self, operations: BlockOperations<'_>) {
        let placement = {
            let state = self.locked();
            BlockOrigin::new(
                &state.origins,
                operations.iter().filter_map(OperationId::new),
            )
            .placement(&state.tree)
        };
        LaunchCursor::enter_block(placement);
    }

    fn block_ran(&self) {
        LaunchCursor::leave_block();
    }
}

impl State {
    /// File every measurement read back under the component that counted its
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
            self.tree.charge_idle(attribution, idle);
        }
    }

    /// Add `duration` to the component that counted `kernel`'s launch under
    /// `attribution`.
    fn credit(&mut self, attribution: Attribution, kernel: &str, duration: Duration) {
        if !self.tree.credit(attribution, kernel, duration) {
            // No launch of it was counted there, so there is no honest component
            // for it. Say how much rather than let `Breakdown::total_time` quietly
            // shrink.
            self.timings.unplace(duration);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breakdown::{Component, Unplaced};

    /// A launch on the thread driving the model lands under the component that
    /// thread has open, and under none once it has closed — the unfused case,
    /// where every kernel is issued synchronously by the op that needs it.
    #[test]
    fn a_launch_lands_under_the_component_its_thread_has_open() {
        let recorder = recorder();
        recorder.enter(declared("block"));
        recorder.enter(declared("attention"));
        recorder.launched("Matmul");
        recorder.exit();
        recorder.launched("Add");
        recorder.exit();
        recorder.launched("Sample");

        let breakdown = recorder.breakdown();
        let names = |kernels: &[crate::Kernel]| -> Vec<String> {
            kernels.iter().map(|kernel| kernel.name.clone()).collect()
        };
        let block: &Component = &breakdown.tree[0];
        assert_eq!(names(&block.children[0].kernels), ["Matmul"]);
        assert_eq!(
            names(&block.kernels),
            ["Add"],
            "the block's own, after its child"
        );
        assert_eq!(names(&breakdown.untagged), ["Sample"]);
    }

    /// A launch is attributed by the thread that issues it, never by reading
    /// the model thread's stack across threads. A thread the cursor never
    /// reached records nothing under a component, however deep the model
    /// thread is — reading across would credit a fused kernel to wherever the
    /// model had got to by the time it launched.
    #[test]
    fn a_launch_on_a_thread_the_cursor_never_reached_is_untagged() {
        let recorder = std::sync::Arc::new(recorder());
        recorder.enter(declared("attention"));

        let elsewhere = std::sync::Arc::clone(&recorder);
        std::thread::spawn(move || elsewhere.launched("Stray"))
            .join()
            .expect("the other thread ran");
        recorder.exit();

        assert!(recorder.breakdown().tree[0].kernels.is_empty());
        assert_eq!(recorder.breakdown().untagged[0].name, "Stray");
    }

    /// A measurement is kept beside the component its launch was counted on and
    /// read back when the run ends — so a kernel two components share by name keeps
    /// each component's time apart, and nothing waited on a kernel while the run
    /// ran.
    #[test]
    fn a_kept_measurement_lands_on_the_component_that_counted_its_launch() {
        let recorder = recorder();
        for (label, micros) in [("query", 13), ("key", 5)] {
            recorder.enter(declared(label));
            recorder.launched("MatmulEntry");
            recorder.profiled("MatmulEntry", measured(micros));
            recorder.exit();
        }
        assert_eq!(
            recorder.breakdown().tree[0].own_time(),
            Duration::ZERO,
            "nothing is read back while the run is going"
        );

        recorder.end_run();
        let breakdown = recorder.breakdown();
        assert_eq!(breakdown.tree[0].own_time(), Duration::from_micros(13));
        assert_eq!(breakdown.tree[1].own_time(), Duration::from_micros(5));
        assert_eq!(breakdown.unplaced, Unplaced::default());
        assert!(!breakdown.timed_alone, "every kernel ran in a full queue");
    }

    /// A duration read as its kernel launched is still credited to its component,
    /// and says the run ran that kernel alone — which is what stops the
    /// report from subtracting a serialized sum from a pipelined wall.
    #[test]
    fn a_duration_read_at_launch_marks_the_pass_timed_alone() {
        let recorder = recorder();
        recorder.enter(declared("query"));
        recorder.launched("MatmulEntry");
        recorder.timed(
            "MatmulEntry",
            Duration::from_micros(13),
            TimingMethod::Device,
        );
        recorder.exit();

        let breakdown = recorder.breakdown();
        assert_eq!(breakdown.tree[0].own_time(), Duration::from_micros(13));
        assert!(breakdown.timed_alone);
    }

    /// Observed traffic is filed under the component open when it arrived,
    /// and priced only when asked: pricing reads values back.
    #[test]
    fn observed_traffic_is_priced_under_the_component_it_arrived_in() {
        let recorder = recorder();
        recorder.enter(Declaration {
            bounds: Some(Bounds::default()),
            ..declared("experts")
        });
        let routes =
            burn_tensor::Tensor::<1, burn_tensor::Int>::from_ints([3, 1, 3, 7], &Device::flex());
        recorder.observe(ObservedBounds::new(routes, |routes| Bounds {
            read: routes.len() as u64,
            write: 0,
            flops: 0,
        }));
        recorder.exit();
        assert_eq!(
            recorder.breakdown().tree[0].observed,
            None,
            "not priced yet"
        );

        recorder.price_observed();
        assert_eq!(
            recorder.breakdown().tree[0].observed,
            Some(Bounds {
                read: 4,
                write: 0,
                flops: 0
            })
        );
    }

    /// Finishing leaves the calling thread attributing nothing, so a launch
    /// after the run is not counted as part of it.
    #[test]
    fn finishing_stops_attributing() {
        let recorder = recorder();
        recorder.end_run();
        assert_eq!(LaunchCursor::current(), Attribution::Untagged);
    }

    fn recorder() -> RunRecorder {
        RunRecorder::new(&Device::flex())
    }

    /// A device measurement of `micros`, as the runtime hands one over: unread.
    fn measured(micros: u64) -> ProfileDuration {
        let start = Instant::now();
        let end = start + Duration::from_micros(micros);
        ProfileDuration::new_device_time(async move {
            cubecl::profile::ProfileTicks::from_start_end(start, end)
        })
    }

    fn declared(label: &str) -> Declaration<'_> {
        Declaration {
            label,
            variant: None,
            site: "test::forward",
            bounds: None,
        }
    }
}
