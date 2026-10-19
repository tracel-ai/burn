//! Which row a kernel launch is recorded under, on whichever thread issues it.
//!
//! A launch belongs to the component that was open when its work was
//! registered — a point in the op stream, not a point in time. The two part as
//! soon as a launch is issued anywhere but the model's own thread, and under
//! fusion it always is: burn-fusion hands every op to a per-device worker, and
//! a fused kernel launches there when the op that seals its segment is
//! processed, dozens of ops after the model thread moved on. Reading the model
//! thread's stack at that moment credits the kernel to wherever the model has
//! got to since.
//!
//! So each thread keeps its own [`Attribution`], moved in step with the ops it
//! runs:
//!
//! * The model thread's is set directly on every enter and exit. An unfused
//!   launch is issued there, synchronously, under exactly that component.
//! * The fusion worker's is set by a task sent into **the queue the ops travel
//!   through**. A device's worker runs every service of its stage from one
//!   FIFO, so the task is processed after every op registered before the
//!   boundary and before every op registered after it. An op the worker
//!   processes is then known to come from the component that registered it,
//!   and a kernel launched outside any block lands under the component whose
//!   op sealed its segment — with nothing draining the stream to make it so.
//!
//! The task touches no fusion state, so the graph a collected pass runs is the
//! graph a clean pass runs.
//!
//! # Inside a block, the block decides
//!
//! Sealing order says which component a segment ran *during*, not whose work
//! it was. Where the fusion server says which operations a block carries out,
//! the recorder knows which components registered them, and for as long as
//! the block runs the worker's attribution is theirs instead —
//! [`enter_block`](LaunchCursor::enter_block) — with the components it spans
//! kept beside it, so a kernel that fused several components' work can be
//! named for them.

use std::cell::RefCell;
use std::sync::Arc;

use burn_tensor::Device;

use super::component_tree::RowId;
use cubecl_common::device::{DeviceId, DeviceService, DeviceServiceStage, ServerUtilitiesHandle};
use cubecl_common::device_handle::DeviceHandle;

/// The row a launch is recorded under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Attribution {
    /// No component is open, so a launch here is untagged.
    Untagged,
    /// The component at this index of the recorder's component tree.
    Component(RowId),
    /// The sync that ends the pass, which runs the fused work still queued
    /// when the last component closed.
    EndOfPass,
}

/// Where a launch lands: the row it is recorded on, and the components whose
/// work it carries out where that is more than one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Placement {
    pub(crate) attribution: Attribution,
    /// Empty outside a block, and for a block of one component's work.
    pub(crate) span: Vec<RowId>,
}

impl Placement {
    /// Under `attribution`, carrying out no one else's work.
    pub(crate) const fn at(attribution: Attribution) -> Self {
        Self {
            attribution,
            span: Vec::new(),
        }
    }
}

/// Which graph a collected pass ran. See [`Breakdown::unfused`](crate::Breakdown::unfused).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fusion {
    /// Its ops went through a fusion worker: the graph production runs.
    Fused,
    /// Each op ran as it was registered.
    Unfused,
}

thread_local! {
    /// Where a launch issued on this thread lands.
    static PLACEMENT: RefCell<Placement> =
        const { RefCell::new(Placement::at(Attribution::Untagged)) };
    /// What each block this thread is inside replaced, innermost last, so
    /// leaving one restores exactly what entering it overrode.
    static OUTER: RefCell<Vec<Placement>> = const { RefCell::new(Vec::new()) };
}

/// Moves every launching thread's [`Attribution`] in step with the ops it runs.
pub(crate) struct LaunchCursor {
    /// The fusion worker of the profiled device, where its ops go through one.
    worker: Option<DeviceHandle<WorkerCursor>>,
}

impl LaunchCursor {
    /// A cursor for work running on `device`, moving its fusion worker's
    /// attribution too where its ops go through one
    /// (`DispatchDevice::fusion_worker`).
    pub(crate) fn new(device: &Device) -> Self {
        let worker = device.as_dispatch().fusion_worker().map(|worker| {
            // Opening the handle waits on the worker while holding cubecl's channel
            // registry, which a block still running there can need too: open it idle.
            device
                .sync()
                .expect("the device answers before the worker is reached");
            DeviceHandle::new(worker)
        });
        Self { worker }
    }

    /// Whether the profiled device's ops go through a fusion worker — whether,
    /// that is, the pass being collected is the fused graph.
    pub(crate) fn fusion(&self) -> Fusion {
        match self.worker {
            Some(_) => Fusion::Fused,
            None => Fusion::Unfused,
        }
    }

    /// Point every launching thread at `attribution`: this one now, and the
    /// fusion worker once it has processed every op registered before this
    /// call.
    ///
    /// Never blocks on the worker, so a boundary costs one queued task rather
    /// than a drain — and never call it holding a lock the worker takes, since
    /// a full queue waits for the worker to make room.
    pub(crate) fn point_at(&self, attribution: Attribution) {
        Self::point_thread_at(attribution);
        if let Some(worker) = &self.worker {
            worker.submit(move |_: &mut WorkerCursor| Self::point_thread_at(attribution));
        }
    }

    /// What a launch issued on the calling thread is recorded under.
    pub(crate) fn current() -> Attribution {
        PLACEMENT.with(|placement| placement.borrow().attribution)
    }

    /// Read where a launch issued on the calling thread lands. Lent rather
    /// than copied: every launch reads it.
    pub(crate) fn with_placement<R>(read: impl FnOnce(&Placement) -> R) -> R {
        PLACEMENT.with(|placement| read(&placement.borrow()))
    }

    /// For as long as a block runs on the calling thread, land its launches
    /// at `placement`, or, for `None`, where they land now.
    ///
    /// The calling thread's alone: a block runs on the thread that entered
    /// it, and nothing else in the process launches its kernels.
    pub(crate) fn enter_block(placement: Option<Placement>) {
        let replaced = PLACEMENT.with(|current| match placement {
            Some(placement) => current.replace(placement),
            None => current.borrow().clone(),
        });
        OUTER.with(|outer| outer.borrow_mut().push(replaced));
    }

    /// The block the calling thread last entered has run: put back what it
    /// replaced.
    pub(crate) fn leave_block() {
        if let Some(replaced) = OUTER.with(|outer| outer.borrow_mut().pop()) {
            PLACEMENT.with(|current| *current.borrow_mut() = replaced);
        }
    }

    fn point_thread_at(attribution: Attribution) {
        PLACEMENT.with(|placement| placement.borrow_mut().attribution = attribution);
    }
}

/// The service a cursor task runs as, on the worker the fusion server runs on.
///
/// Stateless: what a task moves is the worker's thread-local, which is what a
/// launch issued there reads.
struct WorkerCursor;

impl DeviceService for WorkerCursor {
    fn init(_device: DeviceId) -> Self {
        WorkerCursor
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }

    /// The fusion server's stage. A device runs each stage on one worker with
    /// one queue, and sharing the fusion server's is the whole mechanism.
    fn stage() -> DeviceServiceStage {
        DeviceServiceStage::Upstream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A service on the same stage standing in for the fusion server: each of
    /// its tasks reads the cursor the way a launch issued there does.
    struct Ops(Vec<Attribution>);

    impl DeviceService for Ops {
        fn init(_device: DeviceId) -> Self {
            Ops(Vec::new())
        }

        fn utilities(&self) -> ServerUtilitiesHandle {
            Arc::new(())
        }

        fn stage() -> DeviceServiceStage {
            DeviceServiceStage::Upstream
        }
    }

    /// The property the attribution rests on: a boundary sent from the model
    /// thread is processed between the ops registered either side of it, by
    /// another service on the worker, without anything being drained.
    ///
    /// On a device id no runtime hands out, so the worker is this test's alone
    /// and its cursor starts where every thread's does.
    #[test]
    fn a_boundary_lands_between_the_ops_registered_either_side_of_it() {
        let device = DeviceId {
            type_id: u16::MAX,
            index_id: u16::MAX,
        };
        let cursor = LaunchCursor {
            worker: Some(DeviceHandle::new(device)),
        };
        let ops = DeviceHandle::<Ops>::new(device);
        let [block] = rows();
        let op = || ops.submit(|ops: &mut Ops| ops.0.push(LaunchCursor::current()));

        op();
        cursor.point_at(block);
        op();
        op();
        cursor.point_at(Attribution::EndOfPass);
        op();

        let seen = ops
            .submit_blocking(|ops: &mut Ops| ops.0.clone())
            .expect("the worker answers");
        assert_eq!(
            seen,
            [Attribution::Untagged, block, block, Attribution::EndOfPass,],
            "each op reads the boundary most recently registered before it"
        );
    }

    /// A block overrides the thread's attribution for as long as it runs and
    /// gives back exactly what it found — nested blocks included, since a
    /// fused kernel can run an unfused fallback inside itself.
    #[test]
    fn a_block_overrides_the_thread_until_it_has_run() {
        let cursor = LaunchCursor { worker: None };
        let [outer, fused, first, second, fallback] = rows();
        cursor.point_at(outer);

        let fused = Placement {
            attribution: fused,
            span: vec![row(first), row(second)],
        };
        LaunchCursor::enter_block(Some(fused.clone()));
        assert_eq!(placement(), fused);

        let fallback = Placement::at(fallback);
        LaunchCursor::enter_block(Some(fallback.clone()));
        assert_eq!(placement(), fallback);
        LaunchCursor::leave_block();

        assert_eq!(placement(), fused);
        LaunchCursor::leave_block();

        assert_eq!(placement(), Placement::at(outer));
    }

    /// Without a worker the calling thread is the only one launching, and its
    /// cursor moves at once.
    #[test]
    fn without_a_worker_the_calling_thread_moves_at_once() {
        let cursor = LaunchCursor { worker: None };
        let [norm] = rows();
        cursor.point_at(norm);
        assert_eq!(LaunchCursor::current(), norm);
        cursor.point_at(Attribution::Untagged);
        assert_eq!(LaunchCursor::current(), Attribution::Untagged);
    }

    /// `N` distinct rows, each where entering it lands a launch.
    fn rows<const N: usize>() -> [Attribution; N] {
        let mut tree = super::super::component_tree::ComponentTree::default();
        let labels = ["a", "b", "c", "d", "e", "f"];
        std::array::from_fn(|at| {
            let row = tree.enter(crate::Declaration {
                label: labels[at],
                variant: None,
                site: "test::forward",
                bounds: None,
            });
            tree.exit();
            row
        })
    }

    fn row(attribution: Attribution) -> RowId {
        match attribution {
            Attribution::Component(row) => row,
            Attribution::Untagged | Attribution::EndOfPass => unreachable!("a row"),
        }
    }

    fn placement() -> Placement {
        LaunchCursor::with_placement(Placement::clone)
    }
}
