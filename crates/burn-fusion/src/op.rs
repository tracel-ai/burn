use burn_backend::StreamId;
use burn_ir::HandleContainer;
use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use crate::{FusionRuntime, stream::Operation};

const MAX_ITEM_COUNT: usize = 256;
const MAX_ITEM_SIZE: usize = 512;

type Item = crate::arena::Item<MAX_ITEM_SIZE>;
type Data = crate::arena::Bytes<MAX_ITEM_SIZE>;
type Arena = crate::arena::Arena<MAX_ITEM_COUNT, MAX_ITEM_SIZE>;

std::thread_local! {
    static ARENA: RefCell<Arena> = const {RefCell::new(Arena::new())};
}

struct UnfusedOpInArena<R: FusionRuntime> {
    /// The data pointer.
    ptr_data: *const Data,
    /// The ref count pointer.
    ptr_count: *const AtomicU32,
    /// The execute function pointer.
    ptr_execute: unsafe fn(*const Data, handles: &mut HandleContainer<R::FusionHandle>),
    /// The drop function pointer.
    ptr_drop: unsafe fn(*const Data),
}

impl<R: FusionRuntime> Clone for UnfusedOpInArena<R> {
    fn clone(&self) -> Self {
        unsafe {
            self.ptr_count
                .as_ref()
                .unwrap()
                .fetch_add(1, Ordering::Relaxed);
        };

        Self {
            ptr_data: self.ptr_data,
            ptr_count: self.ptr_count,
            ptr_execute: self.ptr_execute,
            ptr_drop: self.ptr_drop,
        }
    }
}

impl<R: FusionRuntime> Drop for UnfusedOpInArena<R> {
    fn drop(&mut self) {
        let count_prev = unsafe {
            self.ptr_count
                .as_ref()
                .unwrap()
                .fetch_sub(1, Ordering::Relaxed)
        };

        // The count is now 1
        if count_prev == 2 {
            unsafe {
                (self.ptr_drop)(self.ptr_data);

                self.ptr_count.as_ref().unwrap().store(0, Ordering::SeqCst);
            };
        }
    }
}

impl<R: FusionRuntime> UnfusedOpInArena<R> {
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        unsafe { (self.ptr_execute)(self.ptr_data, handles) };
    }
}

unsafe fn shim_execute<R: FusionRuntime, O: Operation<R>>(
    ptr_data: *const Data,
    handles: &mut HandleContainer<R::FusionHandle>,
) {
    let operation: &O = unsafe { &*(ptr_data as *const O) };
    operation.execute(handles);
}

unsafe fn shim_drop<R: FusionRuntime, O: Operation<R>>(ptr_item: *const Data) {
    let operation_ptr = ptr_item as *mut O;
    unsafe {
        core::ptr::drop_in_place(operation_ptr);
    }
}

/// An [operation](Operation) that isn't fused.
///
/// This can be executed with [Self::execute].
pub struct UnfusedOp<R: FusionRuntime> {
    kind: UnfusedOpKind<R>,
    stream_id: StreamId,
}

impl<R: FusionRuntime> UnfusedOp<R> {
    /// Creates a new unfused [operation](Operation) that will execute on the given [StreamId].
    pub fn new<O: Operation<R> + 'static>(op: O, stream_id: StreamId) -> Self {
        let arena_item = match Arena::accept::<O>() {
            true => ARENA.with_borrow_mut(|arena| arena.reserve()),
            false => None,
        };

        let ptr_item = match arena_item {
            Some(ptr) => ptr,
            None => {
                return UnfusedOp {
                    kind: UnfusedOpKind::Alloc(Arc::new(op)),
                    stream_id,
                };
            }
        };

        let item: &mut Item = unsafe { ptr_item.as_mut().unwrap() };

        let ptr_data = core::ptr::from_ref(&item.bytes);
        let ptr_count = core::ptr::from_ref(&item.count);

        #[allow(invalid_reference_casting)]
        unsafe {
            core::ptr::write(ptr_data as *mut O, op);
        };

        let ptr_execute = shim_execute::<R, O>;
        let ptr_drop = shim_drop::<R, O>;

        UnfusedOp {
            kind: UnfusedOpKind::Arena(UnfusedOpInArena {
                ptr_data,
                ptr_count,
                ptr_execute,
                ptr_drop,
            }),
            stream_id,
        }
    }

    /// Executes the [operation](Operation) and modifies the given handles.
    pub fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        let old = unsafe { StreamId::swap(self.stream_id) };
        match &self.kind {
            UnfusedOpKind::Arena(o) => o.execute(handles),
            UnfusedOpKind::Alloc(o) => o.execute(handles),
        }
        unsafe { StreamId::swap(old) };
    }
}

impl<R: FusionRuntime> Clone for UnfusedOp<R> {
    fn clone(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            stream_id: self.stream_id,
        }
    }
}

enum UnfusedOpKind<R: FusionRuntime> {
    Arena(UnfusedOpInArena<R>),
    Alloc(Arc<dyn Operation<R>>),
}

unsafe impl<R: FusionRuntime> Send for UnfusedOp<R> {}
unsafe impl<R: FusionRuntime> Sync for UnfusedOp<R> {}

impl<R: FusionRuntime> Clone for UnfusedOpKind<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Arena(o) => Self::Arena(o.clone()),
            Self::Alloc(o) => Self::Alloc(o.clone()),
        }
    }
}
