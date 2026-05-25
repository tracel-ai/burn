use burn_backend::{Backend, TypedDevice};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> TypedDevice<Self> for Autodiff<B, C> {
    fn complex_device(_tensor: &B::ComplexTensorPrimitive) -> B::Device {
        panic!("Autodiff does not yet support complex tensors")
    }
}