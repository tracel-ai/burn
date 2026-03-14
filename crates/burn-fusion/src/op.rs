use burn_ir::HandleContainer;
use std::{cell::RefCell, sync::Arc, thread::ThreadId};

use crate::{FusionRuntime, stream::Operation};

const MAX_SIZE: usize = 4096;
const MAX_ITEM: usize = 256;
type Bytes = [u128; MAX_SIZE / 16];

struct Data {
    bytes: Bytes,
    count: u32,
    is_free: bool,
}

std::thread_local! {
    static ARENA: RefCell<Vec<Data>> = const {RefCell::new(Vec::new())};
    static THREAD_ID: RefCell<Option<ThreadId>> = const {RefCell::new(None)};
}

struct ManagedOperation<R: FusionRuntime> {
    index: usize,
    ptr_data: *mut Data,
    ptr_execute: unsafe fn(*const Data, handles: &mut HandleContainer<R::FusionHandle>),
    ptr_drop: unsafe fn(*const Data),
    thread_id: ThreadId,
}

impl<R: FusionRuntime> Clone for ManagedOperation<R> {
    fn clone(&self) -> Self {
        unsafe {
            self.ptr_data.as_mut().unwrap().count += 1;
        };

        Self {
            index: self.index,
            ptr_data: self.ptr_data,
            ptr_execute: self.ptr_execute,
            ptr_drop: self.ptr_drop,
            thread_id: self.thread_id,
        }
    }
}

impl<R: FusionRuntime> Drop for ManagedOperation<R> {
    fn drop(&mut self) {
        let count = unsafe {
            self.ptr_data.as_mut().unwrap().count -= 1;
            self.ptr_data.as_ref().unwrap().count
        };

        if count == 0 {
            assert_eq!(self.thread_id, thread_id());

            unsafe { (self.ptr_drop)(self.ptr_data) };

            ARENA.with_borrow_mut(|arena| {
                arena[self.index].is_free = true;
            });
        }
    }
}

impl<R: FusionRuntime> ManagedOperation<R> {
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        unsafe { (self.ptr_execute)(self.ptr_data, handles) }
    }
}

fn thread_id() -> ThreadId {
    THREAD_ID.with_borrow_mut(|t| match t {
        Some(t) => t.clone(),
        None => {
            *t = Some(std::thread::current().id());
            t.as_ref().unwrap().clone()
        }
    })
}

fn reserve<R: FusionRuntime, O: Operation<R>>(op: O) -> ManagedOperation<R> {
    let mut init = false;
    let thread_id = THREAD_ID.with_borrow_mut(|t| match t {
        Some(t) => t.clone(),
        None => {
            *t = Some(std::thread::current().id());
            init = true;
            t.as_ref().unwrap().clone()
        }
    });

    ARENA.with_borrow_mut(|arena| {
        if init {
            arena.reserve(MAX_ITEM);
        }

        let mut index = -1;

        for (i, data) in arena.iter_mut().enumerate() {
            if data.is_free {
                data.is_free = false;
                index = i as i32;
                break;
            }
        }

        let index = if index >= 0 {
            index as usize
        } else {
            let index = arena.len();
            if index >= MAX_ITEM {
                panic!("Too many items");
            }
            arena.push(Data {
                bytes: [0; MAX_SIZE / 16],
                is_free: false,
                count: 1,
            });
            index
        };

        let ptr_data = unsafe { arena.as_mut_ptr().add(index) };
        unsafe {
            core::ptr::write(ptr_data as *mut O, op);
        };
        let ptr_execute = shim_execute::<R, O>;
        let ptr_drop = shim_drop::<R, O>;

        ManagedOperation {
            index,
            ptr_data,
            ptr_execute,
            ptr_drop,
            thread_id,
        }
    })
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
}

impl<R: FusionRuntime> Clone for OperationCall<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<R: FusionRuntime> OperationCall<R> {
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>) {
        match &self.inner {
            OperationCallInner::Managed(o) => o.execute(handles),
            OperationCallInner::Fallback(o) => o.execute(handles),
        }
    }
}

impl<R: FusionRuntime> Clone for OperationCallInner<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Managed(o) => Self::Managed(o.clone()),
            Self::Fallback(o) => Self::Fallback(o.clone()),
        }
    }
}
