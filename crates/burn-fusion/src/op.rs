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

const MAX_SIZE: usize = 4096;
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
    ) -> OperationCall<R> {
        let data = match size_of::<O>() <= MAX_SIZE {
            true => self.reserve_data(),
            false => None,
        };

        let (index, ptr_data) = match data {
            Some(val) => val,
            None => {
                return OperationCall {
                    inner: OperationCallInner::Fallback(Arc::new(op)),
                    stream_id,
                };
            }
        };

        unsafe {
            core::ptr::write(ptr_data as *mut O, op);
        };
        let ptr_execute = shim_execute::<R, O>;
        let ptr_drop = shim_drop::<R, O>;

        OperationCall {
            inner: OperationCallInner::Managed(ManagedOperation {
                index,
                ptr_data,
                ptr_execute,
                ptr_drop,
            }),
            stream_id,
        }
    }
}

struct ManagedOperation<R: FusionRuntime> {
    index: usize,
    ptr_data: *mut Data,
    ptr_execute: unsafe fn(*const Data, handles: &mut HandleContainer<R::FusionHandle>),
    ptr_drop: unsafe fn(*const Data),
}

impl<R: FusionRuntime> Clone for ManagedOperation<R> {
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

impl<R: FusionRuntime> Drop for ManagedOperation<R> {
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

impl<R: FusionRuntime> ManagedOperation<R> {
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

enum OperationCallInner<R: FusionRuntime> {
    Managed(ManagedOperation<R>),
    Fallback(Arc<dyn Operation<R>>),
}

pub struct OperationCall<R: FusionRuntime> {
    inner: OperationCallInner<R>,
    stream_id: StreamId,
}

impl<R: FusionRuntime> Clone for OperationCall<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            stream_id: self.stream_id.clone(),
        }
    }
}

impl<R: FusionRuntime> OperationCall<R> {
    pub fn new<O: Operation<R> + 'static>(op: O, stream_id: StreamId) -> Self {
        ARENA.with_borrow_mut(|arena| arena.reserve(op, stream_id))
    }

    pub fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        let old = unsafe { StreamId::swap(self.stream_id) };
        match &self.inner {
            OperationCallInner::Managed(o) => o.execute(handles),
            OperationCallInner::Fallback(o) => o.execute(handles),
        }
        unsafe { StreamId::swap(old) };
    }
}

unsafe impl<R: FusionRuntime> Send for OperationCall<R> {}
unsafe impl<R: FusionRuntime> Sync for OperationCall<R> {}

impl<R: FusionRuntime> Clone for OperationCallInner<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Managed(o) => Self::Managed(o.clone()),
            Self::Fallback(o) => Self::Fallback(o.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_lazy_initialization() {
        let mut arena = Arena::new(10);
        assert_eq!(
            arena.buffer.len(),
            0,
            "Buffer should be empty before first reservation"
        );

        arena.reserve_data();

        assert_eq!(
            arena.buffer.len(),
            10,
            "Buffer should be initialized to size"
        );
        assert!(arena.init);
    }

    #[test]
    fn test_sequential_allocation_moves_cursor() {
        let mut arena = Arena::new(3);

        // First allocation
        let (idx1, _) = arena.reserve_data().expect("Should allocate");
        assert_eq!(idx1, 0);
        assert_eq!(arena.cursor, 0);

        // Second allocation
        let (idx2, _) = arena.reserve_data().expect("Should allocate");
        assert_eq!(idx2, 1);
        assert_eq!(arena.cursor, 1);
    }

    #[test]
    fn test_reuse_of_freed_data() {
        let mut arena = Arena::new(2);

        // Fill the arena
        let (_idx0, data0) = arena.reserve_data().unwrap();
        let (_idx1, _data1) = arena.reserve_data().unwrap();

        // Arena is now full (counts are 2)
        assert!(arena.reserve_data().is_none(), "Should be full");

        // Manually "free" index 0 by setting count to 0 (simulating ManagedOperation drop)
        unsafe { data0.as_ref().unwrap().count.store(0, Ordering::Relaxed) };

        // Should now be able to reserve again, and it should pick up index 0
        let (new_idx, _) = arena.reserve_data().expect("Should reuse index 0");
        assert_eq!(new_idx, 0);
    }

    #[test]
    fn test_circular_cursor_search() {
        let mut arena = Arena::new(3);

        // Fill 0, 1, 2
        let (_, _d0) = arena.reserve_data().unwrap();
        let (_, d1) = arena.reserve_data().unwrap();
        let (_, _d2) = arena.reserve_data().unwrap();

        // Free index 1 (the middle)
        unsafe { d1.as_ref().unwrap().count.store(0, Ordering::Relaxed) };

        // Currently cursor is at 2. The search starts at (cursor + i) % size.
        // It should wrap around and find index 1.
        let (reused_idx, _) = arena
            .reserve_data()
            .expect("Should find the hole at index 1");
        assert_eq!(reused_idx, 1);
        assert_eq!(arena.cursor, 1);
    }

    #[test]
    fn test_full_arena_returns_none() {
        let size = 5;
        let mut arena = Arena::new(size);

        for _ in 0..size {
            assert!(arena.reserve_data().is_some());
        }

        // Next one should fail
        assert!(arena.reserve_data().is_none());
    }
}
