use burn_backend::{ExecutionError, StreamId};
use burn_ir::HandleContainer;
use burn_std::allocation_label::AllocationLabelStack;
use burn_std::arena::ReservedMemory;
use std::{cell::RefCell, sync::Arc};

use crate::{FusionRuntime, stream::Operation};

const MAX_ITEM_COUNT: usize = 256;
const MAX_ITEM_SIZE: usize = 512;

type Data = burn_std::arena::Bytes<MAX_ITEM_SIZE>;
type Arena = burn_std::arena::Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE>;

std::thread_local! {
    static ARENA: RefCell<Arena> = const {RefCell::new(Arena::new())};
}

/// An [operation](Operation) that isn't fused.
///
/// This can be executed with [Self::execute].
pub struct UnfusedOp<R: FusionRuntime> {
    kind: UnfusedOpKind<R>,
    stream_id: StreamId,
    /// The allocation labels in effect where the operation was registered,
    /// applied where it runs: later, on the device's thread, under whatever
    /// work happened to flush the queue.
    labels: AllocationLabelStack,
}

impl<R: FusionRuntime> UnfusedOp<R> {
    /// Creates a new unfused [operation](Operation) that will execute on the given [StreamId].
    pub fn new<O: Operation<R> + 'static>(op: O, stream_id: StreamId) -> Self {
        let labels = AllocationLabelStack::current();
        let arena_item = match Arena::accept::<O>() {
            true => ARENA.with_borrow_mut(|arena| arena.reserve()),
            false => None,
        };

        let reserved = match arena_item {
            Some(val) => val,
            None => {
                return UnfusedOp {
                    kind: UnfusedOpKind::Alloc(Arc::new(op)),
                    stream_id,
                    labels,
                };
            }
        };
        let reserved = reserved.init(op);

        UnfusedOp {
            kind: UnfusedOpKind::Arena(UnfusedOpInArena {
                reserved,
                ptr_execute: shim_execute::<R, O>,
                ptr_skipped: shim_skipped::<R, O>,
            }),
            stream_id,
            labels,
        }
    }

    /// Executes the [operation](Operation) and modifies the given handles.
    pub fn execute(
        &self,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        self.labels.apply(|| {
            self.stream_id.executes(|| match &self.kind {
                UnfusedOpKind::Arena(o) => o.execute(handles),
                UnfusedOpKind::Alloc(o) => o.execute(handles),
            })
        })
    }

    /// The allocation labels in effect where the operation was registered.
    pub fn labels(&self) -> &AllocationLabelStack {
        &self.labels
    }

    /// Runs the [operation](Operation)'s [`skipped`](Operation::skipped) in place of its
    /// execution.
    pub fn skipped(&self) {
        self.stream_id.executes(|| match &self.kind {
            UnfusedOpKind::Arena(o) => o.skipped(),
            UnfusedOpKind::Alloc(o) => o.skipped(),
        })
    }
}

#[derive(Debug)]
struct UnfusedOpInArena<R: FusionRuntime> {
    /// The data pointer.
    reserved: ReservedMemory<MAX_ITEM_SIZE>,
    /// The execute function pointer.
    ptr_execute: fn(
        *const Data,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> Result<(), ExecutionError>,
    /// The skipped function pointer.
    ptr_skipped: fn(*const Data),
}

impl<R: FusionRuntime> Clone for UnfusedOpInArena<R> {
    fn clone(&self) -> Self {
        Self {
            reserved: self.reserved.clone(),
            ptr_execute: self.ptr_execute,
            ptr_skipped: self.ptr_skipped,
        }
    }
}

impl<R: FusionRuntime> UnfusedOpInArena<R> {
    fn execute(
        &self,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        (self.ptr_execute)(self.reserved.as_ref(), handles)
    }

    fn skipped(&self) {
        (self.ptr_skipped)(self.reserved.as_ref())
    }
}

fn shim_execute<R: FusionRuntime, O: Operation<R>>(
    ptr_data: *const Data,
    handles: &mut HandleContainer<R::FusionHandle>,
) -> Result<(), ExecutionError> {
    let operation: &O = unsafe { (ptr_data as *const O).as_ref().unwrap() };
    operation.execute(handles)
}

fn shim_skipped<R: FusionRuntime, O: Operation<R>>(ptr_data: *const Data) {
    let operation: &O = unsafe { (ptr_data as *const O).as_ref().unwrap() };
    operation.skipped()
}

impl<R: FusionRuntime> Clone for UnfusedOp<R> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            stream_id: self.stream_id,
            labels: self.labels.clone(),
        }
    }
}

enum UnfusedOpKind<R: FusionRuntime> {
    Arena(UnfusedOpInArena<R>),
    Alloc(Arc<dyn Operation<R>>),
}

impl<R: FusionRuntime> Clone for UnfusedOpKind<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Arena(o) => Self::Arena(o.clone()),
            Self::Alloc(o) => Self::Alloc(o.clone()),
        }
    }
}
