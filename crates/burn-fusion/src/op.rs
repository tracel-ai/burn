use burn_backend::StreamId;
use burn_ir::HandleContainer;
use std::{
    cell::RefCell,
    sync::{Arc, atomic::Ordering},
};

use crate::{FusionRuntime, arena::Arena, stream::Operation};

const MAX_ITEM_COUNT: usize = 256;
const MAX_ITEM_SIZE: usize = 2048;

type Storage = crate::arena::Item<MAX_ITEM_SIZE>;

std::thread_local! {
    static ARENA: RefCell<Arena::<MAX_ITEM_COUNT, MAX_ITEM_SIZE>> = const {RefCell::new(Arena::new())};
}

struct UnfusedOpInArena<R: FusionRuntime> {
    /// The index in the arena.
    index: usize,
    /// The data pointer.
    ptr_data: *mut Storage,
    /// The execute function pointer.
    ptr_execute: unsafe fn(*const Storage, handles: &mut HandleContainer<R::FusionHandle>),
    /// The drop function pointer.
    ptr_drop: unsafe fn(*const Storage),
}

impl<R: FusionRuntime> Clone for UnfusedOpInArena<R> {
    fn clone(&self) -> Self {
        unsafe {
            self.ptr_data
                .as_ref()
                .unwrap()
                .count
                .fetch_add(1, Ordering::Relaxed);
        };

        Self {
            index: self.index,
            ptr_data: self.ptr_data,
            ptr_execute: self.ptr_execute,
            ptr_drop: self.ptr_drop,
        }
    }
}

impl<R: FusionRuntime> Drop for UnfusedOpInArena<R> {
    fn drop(&mut self) {
        let count = unsafe {
            self.ptr_data
                .as_ref()
                .unwrap()
                .count
                .fetch_sub(1, Ordering::Relaxed)
        };

        if count == 1 {
            unsafe {
                (self.ptr_drop)(self.ptr_data);

                self.ptr_data
                    .as_ref()
                    .unwrap()
                    .count
                    .store(0, Ordering::Relaxed);
            };
        }
    }
}

impl<R: FusionRuntime> UnfusedOpInArena<R> {
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        unsafe { (self.ptr_execute)(self.ptr_data, handles) }
    }
}

unsafe fn shim_execute<R: FusionRuntime, O: Operation<R>>(
    ptr_data: *const Storage,
    handles: &mut HandleContainer<R::FusionHandle>,
) {
    let operation: &O = unsafe { &*(ptr_data as *const O) };
    operation.execute(handles);
}

unsafe fn shim_drop<R: FusionRuntime, O: Operation<R>>(ptr_data: *const Storage) {
    let operation_ptr = ptr_data as *mut O;
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
        let arena_item = match size_of::<O>() <= MAX_ITEM_SIZE {
            true => ARENA.with_borrow_mut(|arena| arena.reserve()),
            false => None,
        };

        let (index, ptr_data) = match arena_item {
            Some(val) => val,
            None => {
                return UnfusedOp {
                    kind: UnfusedOpKind::Alloc(Arc::new(op)),
                    stream_id: stream_id,
                };
            }
        };

        unsafe {
            core::ptr::write(ptr_data as *mut O, op);
        };
        let ptr_execute = shim_execute::<R, O>;
        let ptr_drop = shim_drop::<R, O>;

        UnfusedOp {
            kind: UnfusedOpKind::Arena(UnfusedOpInArena {
                index,
                ptr_data,
                ptr_execute,
                ptr_drop,
            }),
            stream_id: stream_id,
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
            stream_id: self.stream_id.clone(),
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
