//! The read-only inputs of the operation a thread is building, each clone's reference held until
//! the operation is submitted: released earlier, a clone dropped on another thread meanwhile would
//! be the last and free the tensor ahead of the operation that reads it.

use burn_backend::{DType, Device, DeviceId, Shape};
use burn_ir::{TensorId, TensorIr, TensorStatus};
use core::cell::RefCell;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    vec::Vec,
};

use crate::{Client, FusionRuntime, FusionTensor, stream::StreamId};

thread_local! {
    static HELD: RefCell<Vec<HeldInput>> = const { RefCell::new(Vec::new()) };
}

/// One clone of an operation's read-only input, holding its reference to the tensor.
pub(crate) struct HeldInput {
    id: TensorId,
    dtype: DType,
    stream: StreamId,
    device: DeviceId,
    count: Arc<AtomicU32>,
    /// Frees the tensor on its device without queuing anything, for a clone released last.
    free: fn(DeviceId, StreamId, TensorIr),
}

/// Hold `input` until this thread's next submission of an operation.
pub(crate) fn hold(input: HeldInput) {
    HELD.with(|held| held.borrow_mut().push(input));
}

/// How many clones of `id` this thread holds.
pub(crate) fn holds(id: TensorId) -> u32 {
    HELD.with(|held| held.borrow().iter().filter(|input| input.id == id).count() as u32)
}

/// Release every input this thread holds, once the operation reading them is submitted.
///
/// Submissions take their places in one order, so a drop that sees a released reference is
/// ordered after the operation.
pub(crate) fn release() {
    // One at a time, so no borrow is held while a release reaches the client.
    while let Some(input) = HELD.with(|held| held.borrow_mut().pop()) {
        input.release();
    }
}

impl HeldInput {
    /// `tensor`'s reference to the `count` its clones share.
    pub(crate) fn of<R: FusionRuntime>(tensor: &FusionTensor<R>, count: Arc<AtomicU32>) -> Self {
        Self {
            id: tensor.id,
            dtype: tensor.dtype,
            stream: tensor.stream,
            device: tensor.client.device().to_id(),
            count,
            free: free_on::<R>,
        }
    }

    fn release(self) {
        // `AcqRel`: the drop that sees this reference gone must see the submission before it.
        if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // A free reads the id and the status alone.
            let tensor = TensorIr {
                id: self.id,
                shape: Shape::from(Vec::<usize>::new()),
                status: TensorStatus::ReadWrite,
                dtype: self.dtype,
            };
            (self.free)(self.device, self.stream, tensor);
        }
    }
}

fn free_on<R: FusionRuntime>(device: DeviceId, stream: StreamId, tensor: TensorIr) {
    Client::<R>::load(&R::FusionDevice::from_id(device)).foreign_drop(stream, tensor);
}
