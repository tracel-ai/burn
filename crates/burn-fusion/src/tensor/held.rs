//! The read-only inputs of the operation a thread is building, each clone's reference held until
//! the operation is submitted: released earlier, a clone dropped on another thread meanwhile would
//! be the last and free the tensor ahead of the operation that reads it.

use burn_backend::{DType, Device, DeviceId, Shape};
use burn_ir::{TensorId, TensorIr, TensorStatus};
use core::cell::RefCell;
use hashbrown::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use crate::{Client, FusionRuntime, FusionTensor, stream::StreamId};

thread_local! {
    static HELD: RefCell<HashMap<TensorId, HeldInput>> = RefCell::new(HashMap::new());
}

/// Hold `input` on this thread until its next submission of an operation.
pub fn hold(input: HeldInput) {
    HELD.with(|held| {
        held.borrow_mut()
            .entry(input.id)
            .and_modify(|held| held.references += 1)
            .or_insert(input);
    });
}

/// How many clones of `id` this thread holds.
pub fn count(id: TensorId) -> u32 {
    HELD.with(|held| held.borrow().get(&id).map_or(0, |input| input.references))
}

/// Release every input this thread holds, once the operation reading them is submitted.
///
/// Submissions take their places in one order, so a drop that sees a released reference is ordered
/// after the operation.
pub fn release() {
    // Taken out, so no borrow is held while a release reaches the client.
    let inputs = HELD.with(|held| core::mem::take(&mut *held.borrow_mut()));
    for input in inputs.into_values() {
        input.release();
    }
}

/// The clones of an operation's read-only input that the operation holds, with their references
/// to the tensor.
pub struct HeldInput {
    id: TensorId,
    dtype: DType,
    stream: StreamId,
    device: DeviceId,
    count: Arc<AtomicU32>,
    references: u32,
    /// Frees the tensor on its device without queuing anything, for clones released last.
    free: fn(Self),
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
            references: 1,
            free: Self::free_on::<R>,
        }
    }

    fn release(self) {
        // `AcqRel`: the drop that sees these references gone must see the submission before it.
        if self.count.fetch_sub(self.references, Ordering::AcqRel) == self.references {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_of_one_tensor_are_released_together() {
        let id = TensorId::new(0);
        let clones = Arc::new(AtomicU32::new(3));
        hold(input(id, &clones));
        hold(input(id, &clones));
        assert_eq!(count(id), 2);

        release();
        assert_eq!(clones.load(Ordering::Acquire), 1);
        assert_eq!(count(id), 0);
    }

    fn input(id: TensorId, clones: &Arc<AtomicU32>) -> HeldInput {
        HeldInput {
            id,
            dtype: DType::F32,
            stream: StreamId::current(),
            device: DeviceId {
                type_id: 0,
                index_id: 0,
            },
            count: clones.clone(),
            references: 1,
            free: |_| unreachable!("the test keeps a clone of its own"),
        }
    }
}
