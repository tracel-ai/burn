use burn_backend::{Backend, TypedDevice, UnimplementedTensorPrimitive};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> TypedDevice<Self> for Autodiff<B, C> {
    fn complex_device(_tensor: &UnimplementedTensorPrimitive<B::ComplexTensorPrimitive>) -> B::Device {
        panic!("Autodiff does not yet support complex tensors")
    }
}