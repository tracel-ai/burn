use burn_backend::{Backend, TypedDevice};

use crate::{Fusion, FusionBackend};

impl<B: Backend + FusionBackend> TypedDevice<Fusion<B>> for Fusion<B> {
    fn complex_device(tensor: &B::ComplexTensorPrimitive) -> B::Device {
        B::complex_device(tensor)
    }
}