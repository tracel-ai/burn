use burn_backend::{Backend, TypedDevice};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> TypedDevice<Self> for Autodiff<B, C> {
    fn complex_device(tensor: &B::ComplexTensorPrimitive) -> B::Device {
        B::complex_device(tensor)
    }
}