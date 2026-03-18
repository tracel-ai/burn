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

const MAX_SIZE: usize = 2048;
const MAX_ITEM: usize = 256;

type Bytes = [u128; MAX_SIZE / 16];

std::thread_local! {
    static ARENA: RefCell<Arena> = const {RefCell::new(Arena::new(MAX_ITEM))};
}

struct Data {
    bytes: Bytes,
    count: AtomicU32,
}

struct Arena {
    buffer: Vec<Data>,
    cursor: usize,
    size: usize,
    init: bool,
}

impl Arena {
    pub const fn new(size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            cursor: 0,
            size,
            init: false,
        }
    }

    pub fn reserve_data(&mut self) -> Option<(usize, *mut Data)> {
        if !self.init {
            for _ in 0..self.size {
                self.buffer.push(Data {
                    bytes: [0; MAX_SIZE / 16],
                    count: AtomicU32::new(0),
                });
            }
            self.init = true;
        }

        for i in 0..self.size {
            let i = (i + self.cursor) % self.size;
            let data = &mut self.buffer[i];

            if data.count.load(Ordering::Relaxed) == 0 {
                self.cursor = i;
                // We start with a ref count of 2, so a ref count of 1 mean the drop must be
                // executed and a ref count of 0 means free for reuse.
                data.count.store(2, Ordering::Relaxed);

                return Some((i, &mut self.buffer[i]));
            }
        }

        None
    }

    fn reserve<R: FusionRuntime, O: Operation<R> + 'static>(
        &mut self,
        op: O,
        stream_id: StreamId,
    ) -> UnfusedOp<R> {
        let data = match size_of::<O>() <= MAX_SIZE {
            true => self.reserve_data(),
            false => None,
        };

        let (index, ptr_data) = match data {
            Some(val) => val,
            None => {
                return UnfusedOp {
                    kind: UnfusedOpKind::Alloc(Arc::new(op)),
                    stream_id,
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
            stream_id,
        }
    }
}

struct UnfusedOpInArena<R: FusionRuntime> {
    /// The index in the arena.
    index: usize,
    /// The data pointer.
    ptr_data: *mut Data,
    /// The execute function pointer.
    ptr_execute: unsafe fn(*const Data, handles: &mut HandleContainer<R::FusionHandle>),
    /// The drop function pointer.
    ptr_drop: unsafe fn(*const Data),
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
    ptr_data: *const Data,
    handles: &mut HandleContainer<R::FusionHandle>,
) {
    let operation: &O = unsafe { &*(ptr_data as *const O) };
    operation.execute(handles);
}

unsafe fn shim_drop<R: FusionRuntime, O: Operation<R>>(ptr_data: *const Data) {
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
        ARENA.with_borrow_mut(|arena| arena.reserve(op, stream_id))
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
