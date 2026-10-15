//! The read-only inputs of the operation a thread is building, each clone's reference held until
//! the operation is submitted: released earlier, a clone dropped on another thread meanwhile would
//! be the last and free the tensor ahead of the operation that reads it.

use burn_backend::{DType, Device, DeviceId, Shape};
use burn_ir::{TensorId, TensorIr, TensorStatus};
use core::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use crate::{Client, FusionRuntime, FusionTensor, stream::StreamId};

thread_local! {
    static THIS_THREAD: RefCell<HeldInputs> = const { RefCell::new(HeldInputs(Vec::new())) };
}

/// The inputs a thread holds for the operation it is building.
pub struct HeldInputs(Vec<HeldInput>);

/// One clone of an operation's read-only input, holding its reference to the tensor.
pub struct HeldInput {
    id: TensorId,
    dtype: DType,
    stream: StreamId,
    device: DeviceId,
    count: Arc<AtomicU32>,
    /// Frees the tensor on its device without queuing anything, for a clone released last.
    free: fn(Self),
}

impl HeldInputs {
    /// Hold `input` on this thread until its next submission of an operation.
    pub fn hold(input: HeldInput) {
        THIS_THREAD.with(|held| held.borrow_mut().0.push(input));
    }

    /// How many clones of `id` this thread holds.
    pub fn count(id: TensorId) -> u32 {
        THIS_THREAD.with(|held| {
            held.borrow()
                .0
                .iter()
                .filter(|input| input.id == id)
                .count() as u32
        })
    }

    /// Release every input this thread holds, once the operation reading them is submitted.
    ///
    /// Submissions take their places in one order, so a drop that sees a released reference is
    /// ordered after the operation.
    pub fn release() {
        // One at a time, so no borrow is held while a release reaches the client.
        while let Some(input) = THIS_THREAD.with(|held| held.borrow_mut().0.pop()) {
            input.release();
        }
    }
}

impl HeldInput {
    /// `tensor`'s reference to the `count` its clones share.
    pub fn new<R: FusionRuntime>(tensor: &FusionTensor<R>, count: Arc<AtomicU32>) -> Self {
        Self {
            id: tensor.id,
            dtype: tensor.dtype,
            stream: tensor.stream,
            device: tensor.client.device().to_id(),
            count,
            free: Self::free_on::<R>,
        }
    }

    fn release(self) {
        // `AcqRel`: the drop that sees this reference gone must see the submission before it.
        if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            (self.free)(self);
        }
    }

    fn free_on<R: FusionRuntime>(self) {
        // A free reads the id and the status alone.
        let tensor = TensorIr {
            id: self.id,
            shape: Shape::from(Vec::<usize>::new()),
            status: TensorStatus::ReadWrite,
            dtype: self.dtype,
        };
        Client::<R>::load(&R::FusionDevice::from_id(self.device)).foreign_drop(self.stream, tensor);
    }
}
