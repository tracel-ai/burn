//! Where every event of an instrumented run is recorded: the components it
//! enters, from the thread driving the model, and the kernels, blocks and
//! timings the runtimes report, from whichever thread issues them.
//!
//! The runtime calls an [`ExecutionObserver`] synchronously, on the thread that
//! issued the launch, before it is submitted. The recorder is the observer,
//! and records each launch under that thread's own attribution — moved in
//! step with the ops the thread runs, and inside a fusion block set to the
//! components whose work the block carries out, with nothing drained to make
//! it so → [`LaunchCursor`].
//!
//! # Timings are read once the run is over
//!
//! Each launch's timing arrives unread and the recorder keeps it, reading
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
use cubecl::logging::{
    Execution, ExecutionKind, ExecutionObserver, ExecutionPlace, Measurement, Timed, TimingRequest,
};
use cubecl::profile::{ProfileDuration, TimingMethod};

use super::collective_ledger::CollectiveLedger;
use super::component_tree::ComponentTree;
use super::cursor::{Attribution, LaunchCursor};
use super::members::MemberList;
#[cfg(feature = "fusion")]
use super::origins::{BlockOrigin, OperationId, OperationOrigins};
use super::pricing::Floors;
use super::timing::{ReadTiming, Timeline, Timings, UnreadTiming};
use super::watching::Watching;
use crate::breakdown::{Breakdown, CollectiveTime, PerMember};
use crate::{ComponentObserver, Declaration};

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
    /// Before the run, so every timing's window can be placed after it
    /// on one clock and the gaps between them read off.
    epoch: Instant,
    /// Where each launching thread records, moved at every boundary.
    cursor: LaunchCursor,
    /// The members of a sharded compute device, which place each execution by its device.
    members: MemberList,
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
    collectives: CollectiveLedger,
    /// What a block's operations are mapped back through.
    #[cfg(feature = "fusion")]
    origins: OperationOrigins,
    /// The floors of a pricing run, taken by the entries in the order they arrive.
    floors: Floors,
    /// What each member did, on a sharded compute device: see [`Breakdown::per_member`].
    per_member: Option<PerMember>,
}

impl RunRecorder {
    /// A recorder for work running on `device`, its entries scored against
    /// `floors`.
    ///
    /// Points every launching thread at no component first: the cursor is a
    /// thread-local, so a worker still pointing into a previous recorder's
    /// arena would record this one's first kernels at an index that means
    /// something else here.
    pub fn new(device: &Device, floors: Floors) -> Self {
        Self::with_members(device, MemberList::of(device), floors)
    }

    fn with_members(device: &Device, members: MemberList, floors: Floors) -> Self {
        let cursor = LaunchCursor::new(device);
        cursor.point_at(Attribution::Untagged);
        Self {
            device: device.clone(),
            epoch: Instant::now(),
            cursor,
            members,
            state: Mutex::new(State {
                floors,
                ..State::default()
            }),
        }
    }

    /// End the run and say what it left behind.
    ///
    /// The work still queued runs while the launches are watched, so it is
    /// counted, and the observers come out with `watching`.
    ///
    /// # Panics
    ///
    /// If the device stops answering, or if a component is still open.
    pub fn finish(&self, watching: Watching) -> Breakdown {
        self.end_run();
        drop(watching);
        self.breakdown()
    }

    /// Run the fused work still queued, under a component of its own, stop
    /// attributing, and read back every timing the run kept.
    ///
    /// A fused segment runs when an op seals it, and the run's last one has
    /// no op after it — the output is returned unread — so its kernels would
    /// otherwise launch at whatever sync came next, after the recorder had
    /// been taken down, and [`Breakdown::total_time`] would quietly shrink by them. They
    /// are the work of the last components, but no component can say which, so they
    /// are named for when they ran.
    ///
    /// The timings are read here rather than as each kernel launched:
    /// reading one waits for its kernel, and waiting after every launch
    /// executes a run one kernel at a time. After the sync every window has closed, so
    /// reading them costs nothing and each is the kernel as it ran in a queue
    /// kept full.
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
        let (unread, ledger) = {
            let mut state = self.locked();
            (
                state.timings.take_unread(),
                std::mem::take(&mut state.collectives),
            )
        };
        let read: Vec<ReadTiming> = unread.into_iter().map(UnreadTiming::read).collect();
        let collectives = ledger.settle(&self.members);
        let mut state = self.locked();
        #[cfg(feature = "fusion")]
        state.origins.clear();
        state.settle(read, self.epoch, &self.members, collectives);
    }

    /// What the run left behind.
    ///
    /// # Panics
    ///
    /// If a component is still open — a tree read mid-run is missing the
    /// kernels its open components have yet to launch.
    fn breakdown(&self) -> Breakdown {
        let state = self.locked();
        state.tree.breakdown(
            state.timings.quality(),
            self.cursor.fusion(),
            state.per_member.clone(),
        )
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
        let entered = {
            let mut state = self.locked();
            let floor = state.floors.next();
            state.tree.enter(Declaration {
                bounds: declared.bounds.or(floor),
                ..declared
            })
        };
        // Outside the lock: a full queue waits for the worker, and the worker
        // may be waiting on the lock to record a launch.
        self.cursor.point_at(entered);
    }

    fn exit(&self) {
        let back_to = self.locked().tree.exit();
        self.cursor.point_at(back_to);
    }

    /// The floors came from the pricing run.
    fn prices(&self) -> bool {
        false
    }
}

/// Every launch is recorded where the thread that issued it lands its
/// launches, never where the model thread has got to: under fusion a kernel
/// launches on the fusion worker, dozens of ops after the model thread
/// registered its work.
impl ExecutionObserver for RunRecorder {
    fn issued(&self, execution: &Execution) {
        match execution.kind {
            ExecutionKind::Kernel { name, .. } => self.count_launch(name),
            _ => self.locked().collectives.issued(execution, &self.members),
        }
    }

    /// Timed, and handed over unread — a breakdown whose kernel components carried
    /// no time would say which kernels ran and nothing about which of them
    /// mattered, and reading each timing at its launch waits for that
    /// kernel, which executes the run one kernel at a time. Communications too
    /// over several members, whose parts and waits say what each collective cost
    /// each member; a device of one runs none.
    fn timing(&self) -> TimingRequest {
        match self.members.several() {
            true => TimingRequest::Deferred(Timed::KernelsAndCommunications),
            false => TimingRequest::Deferred(Timed::Kernels),
        }
    }

    fn measured(&self, execution: &Execution, measurement: Measurement) {
        match (&execution.kind, measurement) {
            (ExecutionKind::Kernel { place, name }, Measurement::Unread(profile)) => {
                self.keep_unread(name, place, profile)
            }
            (ExecutionKind::Kernel { name, .. }, Measurement::Read { duration, method }) => {
                self.credit_read(name, duration, method)
            }
            (_, Measurement::Unread(profile)) => {
                self.locked().collectives.measured(execution, profile)
            }
            (_, Measurement::Read { .. }) => {}
        }
    }
}

impl RunRecorder {
    fn count_launch(&self, kernel: &'static str) {
        LaunchCursor::with_placement(|placement| {
            self.locked().tree.count_launch(placement, kernel);
        });
    }

    /// Kept beside the component that counted the launch and the device at
    /// `place`, and read back in [`finish`](RunRecorder::finish). The runtime
    /// hands it over on the thread that issued the launch, inside the same call,
    /// so the calling thread's attribution is still the launch's.
    fn keep_unread(&self, kernel: &'static str, place: &ExecutionPlace, profile: ProfileDuration) {
        let attribution = LaunchCursor::current();
        let ran_on = self.members.ran_on(place);
        self.locked()
            .timings
            .keep(attribution, ran_on, kernel, profile);
    }

    /// Only while the runtime's profiling logger reads timings too, which
    /// reads each at its launch and hands the observer the duration. A kernel
    /// read as it launched ran alone, so the run is marked
    /// [`timed_alone`](Breakdown::timed_alone).
    fn credit_read(&self, kernel: &'static str, duration: Duration, method: TimingMethod) {
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
    /// File every timing read back under the component that counted its
    /// launch, charge each device's idle stretches between them, and, over several `members`,
    /// keep what each did and what the `collectives` cost each.
    fn settle(
        &mut self,
        read: Vec<ReadTiming>,
        epoch: Instant,
        members: &MemberList,
        collectives: Vec<CollectiveTime>,
    ) {
        let mut timeline = Timeline::new(epoch, members.clone());
        for timed in read {
            match timed.ticks {
                Some(ticks) => {
                    self.credit(timed.attribution, timed.kernel, ticks.duration());
                    timeline.push(timed.attribution, timed.ran_on, &ticks);
                }
                None => self.timings.unplace(Duration::ZERO),
            }
        }
        let settled = timeline.settle();
        for (attribution, idle) in settled.idle {
            self.tree.charge_idle(attribution, idle);
        }
        self.per_member = settled.per_member.map(|per_member| PerMember {
            collectives,
            ..per_member
        });
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
    use burn_tensor::StreamId;
    use cubecl::logging::{Communication, ExecutionId, StreamRole, WaitTarget};
    use cubecl::server::{Collective, TransportProvider};
    use cubecl_common::device::{DeviceId, ServiceId};

    use super::super::test_observation::minted;
    use super::*;
    use crate::breakdown::{CollectiveMemberTime, CollectivePath, Component, MemberTime, Unplaced};

    /// A launch on the thread driving the model lands under the component that
    /// thread has open, and under none once it has closed — the unfused case,
    /// where every kernel is issued synchronously by the op that needs it.
    #[test]
    fn a_launch_lands_under_the_component_its_thread_has_open() {
        let recorder = recorder();
        recorder.enter(declared("block"));
        recorder.enter(declared("attention"));
        recorder.count_launch("Matmul");
        recorder.exit();
        recorder.count_launch("Add");
        recorder.exit();
        recorder.count_launch("Sample");

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
        std::thread::spawn(move || elsewhere.count_launch("Stray"))
            .join()
            .expect("the other thread ran");
        recorder.exit();

        assert!(recorder.breakdown().tree[0].kernels.is_empty());
        assert_eq!(recorder.breakdown().untagged[0].name, "Stray");
    }

    /// A timing is kept beside the component its launch was counted on and
    /// read back when the run ends — so a kernel two components share by name keeps
    /// each component's time apart, and nothing waited on a kernel while the run
    /// ran.
    #[test]
    fn a_kept_measurement_lands_on_the_component_that_counted_its_launch() {
        let recorder = recorder();
        for (label, micros) in [("query", 13), ("key", 5)] {
            recorder.enter(declared(label));
            recorder.count_launch("MatmulEntry");
            recorder.keep_unread("MatmulEntry", &place(0), unread_timing(micros));
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
    fn a_duration_read_at_launch_marks_the_run_timed_alone() {
        let recorder = recorder();
        recorder.enter(declared("query"));
        recorder.count_launch("MatmulEntry");
        recorder.credit_read(
            "MatmulEntry",
            Duration::from_micros(13),
            TimingMethod::Device,
        );
        recorder.exit();

        let breakdown = recorder.breakdown();
        assert_eq!(breakdown.tree[0].own_time(), Duration::from_micros(13));
        assert!(breakdown.timed_alone);
    }

    /// Finishing leaves the calling thread attributing nothing, so a launch
    /// after the run is not counted as part of it.
    #[test]
    fn finishing_stops_attributing() {
        let recorder = recorder();
        recorder.end_run();
        assert_eq!(LaunchCursor::current(), Attribution::Untagged);
    }

    /// A collective settles into each member's part time and exposed time, summed over the
    /// collectives of its kind. A wait whose part had ended when its stream reached it reads
    /// zero, and a wait for every queued collective is charged to the member's last part.
    #[test]
    fn a_collective_settles_into_each_members_part_and_exposed_time() {
        let [
            first,
            part_0,
            part_1,
            wait_0,
            wait_1,
            second,
            part_0b,
            part_1b,
            queued_1,
        ] = minted();
        let recorder = sharded();
        for collective in [first, second] {
            recorder.issue(collective, None, all_reduce(4096, native()));
        }
        recorder.share(part_0, first, part(0), 30);
        recorder.share(part_1, first, part(1), 20);
        recorder.share(wait_0, first, wait(0, WaitTarget::Part(part_0)), 10);
        recorder.share(wait_1, first, wait(1, WaitTarget::Part(part_1)), 0);
        recorder.share(part_0b, second, part(0), 5);
        recorder.share(part_1b, second, part(1), 7);
        recorder.share(queued_1, first, wait(1, WaitTarget::QueuedCollectives), 4);

        recorder.end_run();
        let collectives = recorder.collectives();
        assert_eq!(
            collectives.len(),
            1,
            "two all-reduces of one kind are one row"
        );
        assert_eq!(collectives[0].count, 2);
        assert_eq!(
            collectives[0].members,
            [member_time(35, 10), member_time(27, 4),]
        );
    }

    /// A skipped collective is counted though nothing of it ran, apart from the native ones, and
    /// each size is a kind of its own.
    #[test]
    fn a_skipped_collective_is_counted_untimed() {
        let [native_small, native_large, skipped] = minted();
        let recorder = sharded();
        recorder.issue(native_small, None, all_reduce(64, native()));
        recorder.issue(native_large, None, all_reduce(4096, native()));
        recorder.issue(
            skipped,
            None,
            all_reduce(64, cubecl::logging::CollectivePath::Skipped),
        );

        recorder.end_run();
        let rows: Vec<(u64, CollectivePath, usize)> = recorder
            .collectives()
            .into_iter()
            .map(|row| {
                assert_eq!(row.members, [CollectiveMemberTime::default(); 2]);
                (row.bytes, row.path, row.count)
            })
            .collect();
        let nccl = CollectivePath::Native {
            transport: "Nccl".to_string(),
        };
        assert_eq!(
            rows,
            [
                (64, nccl.clone(), 1),
                (4096, nccl, 1),
                (64, CollectivePath::Skipped, 1),
            ]
        );
    }

    /// A kernel is on the member whose device its place names, whichever thread launched it,
    /// and one on a device that is no member's is on none.
    #[test]
    fn a_kernel_is_placed_on_the_member_its_device_is() {
        let recorder = sharded();
        let kernel = |device| ExecutionKind::Kernel {
            place: ExecutionPlace {
                service: service(device),
                stream: StreamRole::Compute(StreamId { value: 0 }),
            },
            name: "Matmul",
        };
        let [on_second, elsewhere] = minted();
        recorder.issue(on_second, None, kernel(MEMBERS[1]));
        recorder.measure(on_second, kernel(MEMBERS[1]), 9);
        recorder.issue(elsewhere, None, kernel(DeviceId::new(0, 9)));
        recorder.measure(elsewhere, kernel(DeviceId::new(0, 9)), 4);

        recorder.end_run();
        let per_member = recorder.breakdown().per_member.expect("two members");
        assert_eq!(per_member.members[0], MemberTime::default());
        assert_eq!(per_member.members[1].on_device, Duration::from_micros(9));
        assert_eq!(per_member.off_members, Duration::from_micros(4));
    }

    /// Communications are timed only where there are several members to time apart.
    #[test]
    fn communications_are_timed_over_several_members() {
        assert_eq!(
            sharded().timing(),
            TimingRequest::Deferred(Timed::KernelsAndCommunications)
        );
        assert_eq!(recorder().timing(), TimingRequest::Deferred(Timed::Kernels));
    }

    /// The devices of the two members of [`sharded`].
    const MEMBERS: [DeviceId; 2] = [
        DeviceId {
            type_id: 0,
            index_id: 1,
        },
        DeviceId {
            type_id: 0,
            index_id: 2,
        },
    ];

    fn recorder() -> RunRecorder {
        RunRecorder::new(&Device::flex(), Floors::default())
    }

    /// A recorder for a sharded compute device of the two [`MEMBERS`].
    fn sharded() -> RunRecorder {
        RunRecorder::with_members(
            &Device::flex(),
            MemberList::over(MEMBERS.map(service).to_vec()),
            Floors::default(),
        )
    }

    impl RunRecorder {
        fn issue(&self, id: ExecutionId, parent: Option<ExecutionId>, kind: ExecutionKind) {
            self.issued(&Execution { id, parent, kind });
        }

        fn measure(&self, id: ExecutionId, kind: ExecutionKind, micros: u64) {
            let execution = Execution {
                id,
                parent: None,
                kind,
            };
            self.measured(&execution, Measurement::Unread(unread_timing(micros)));
        }

        /// Issue a part or wait of `collective`, and hand its window of `micros` over.
        fn share(
            &self,
            id: ExecutionId,
            collective: ExecutionId,
            kind: ExecutionKind,
            micros: u64,
        ) {
            self.issue(id, Some(collective), kind.clone());
            self.measure(id, kind, micros);
        }

        fn collectives(&self) -> Vec<crate::CollectiveTime> {
            self.breakdown()
                .per_member
                .expect("two members")
                .collectives
        }
    }

    fn all_reduce(bytes: u64, path: cubecl::logging::CollectivePath) -> ExecutionKind {
        ExecutionKind::Communication(Communication::Collective {
            operation: Collective::AllReduce,
            members: MEMBERS.map(service).to_vec(),
            bytes,
            path,
        })
    }

    fn native() -> cubecl::logging::CollectivePath {
        cubecl::logging::CollectivePath::Native(TransportProvider::Nccl)
    }

    fn part(member: usize) -> ExecutionKind {
        ExecutionKind::Part {
            place: ExecutionPlace {
                service: service(MEMBERS[member]),
                stream: StreamRole::Collective,
            },
        }
    }

    fn wait(member: usize, awaits: WaitTarget) -> ExecutionKind {
        ExecutionKind::Wait {
            place: ExecutionPlace {
                service: service(MEMBERS[member]),
                stream: StreamRole::Compute(StreamId { value: 0 }),
            },
            awaits,
        }
    }

    fn member_time(part: u64, exposed: u64) -> CollectiveMemberTime {
        CollectiveMemberTime {
            part: Duration::from_micros(part),
            exposed: Duration::from_micros(exposed),
        }
    }

    fn service(device: DeviceId) -> ServiceId {
        ServiceId::of::<()>(device)
    }

    fn place(index_id: u16) -> ExecutionPlace {
        ExecutionPlace {
            service: service(DeviceId::new(0, index_id)),
            stream: StreamRole::Compute(StreamId { value: 0 }),
        }
    }

    /// A device timing of `micros`, as the runtime hands one over: unread.
    fn unread_timing(micros: u64) -> ProfileDuration {
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
