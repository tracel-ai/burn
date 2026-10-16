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

/// Held inputs past which the clones held of one tensor are counted per id rather than found by a
/// scan: an operation reading many shared tensors, a concatenation, would scan once per input.
const SCANNED: usize = 16;

thread_local! {
    static HELD: RefCell<Held> = const {
        RefCell::new(Held {
            inputs: Vec::new(),
            counts: None,
        })
    };
}

/// Hold `input` on this thread until its next submission of an operation.
pub fn hold(input: HeldInput) {
    HELD.with(|held| held.borrow_mut().push(input));
}

/// How many clones of `id` this thread holds.
pub fn count(id: TensorId) -> u32 {
    HELD.with(|held| held.borrow().count(id))
}

/// Release every input this thread holds, once the operation reading them is submitted.
///
/// Submissions take their places in one order, so a drop that sees a released reference is ordered
/// after the operation.
pub fn release() {
    // One at a time, so no borrow is held while a release reaches the client.
    while let Some(input) = HELD.with(|held| held.borrow_mut().pop()) {
        input.release();
    }
}

/// The inputs a thread holds for the operation it is building.
struct Held {
    inputs: Vec<HeldInput>,
    /// The clones held of each tensor, once there are more than [`SCANNED`] inputs.
    counts: Option<HashMap<TensorId, u32>>,
}

impl Held {
    fn push(&mut self, input: HeldInput) {
        match &mut self.counts {
            Some(counts) => *counts.entry(input.id).or_default() += 1,
            None if self.inputs.len() == SCANNED => {
                let mut counts = HashMap::new();
                for held in self.inputs.iter().chain([&input]) {
                    *counts.entry(held.id).or_default() += 1;
                }
                self.counts = Some(counts);
            }
            None => {}
        }
        self.inputs.push(input);
    }

    fn count(&self, id: TensorId) -> u32 {
        match &self.counts {
            Some(counts) => counts.get(&id).copied().unwrap_or(0),
            None => self.inputs.iter().filter(|input| input.id == id).count() as u32,
        }
    }

    fn pop(&mut self) -> Option<HeldInput> {
        let input = self.inputs.pop()?;
        if self.inputs.is_empty() {
            self.counts = None;
        } else if let Some(count) = self
            .counts
            .as_mut()
            .and_then(|counts| counts.get_mut(&input.id))
        {
            *count -= 1;
        }
        Some(input)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_past_the_scanned_inputs_agree_with_a_scan() {
        let mut held = Held {
            inputs: Vec::new(),
            counts: None,
        };
        let agrees = |held: &Held| {
            (0..6).map(TensorId::new).all(|id| {
                held.count(id) == held.inputs.iter().filter(|input| input.id == id).count() as u32
            })
        };

        for n in 0..SCANNED as u64 * 2 {
            held.push(input(TensorId::new(n % 5)));
            assert!(agrees(&held), "after {} holds", n + 1);
        }
        while held.pop().is_some() {
            assert!(agrees(&held), "with {} held", held.inputs.len());
        }
        assert!(held.counts.is_none());
    }

    /// An input that is never its tensor's last reference, so dropping it frees nothing.
    fn input(id: TensorId) -> HeldInput {
        HeldInput {
            id,
            dtype: DType::F32,
            stream: StreamId::current(),
            device: DeviceId {
                type_id: 0,
                index_id: 0,
            },
            count: Arc::new(AtomicU32::new(2)),
            free: |_| unreachable!("a test input is never the last reference"),
        }
    }
}
