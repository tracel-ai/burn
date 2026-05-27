use burn_backend::{Backend, TypedDevice, UnimplementedTensorPrimitive};

use crate::{Fusion, FusionBackend, FusionTensor};

impl<B: Backend + FusionBackend> TypedDevice<Fusion<B>> for Fusion<B> {
    fn complex_device(
        _tensor: &UnimplementedTensorPrimitive<FusionTensor<<B as FusionBackend>::FusionRuntime>>,
    ) -> B::Device {
        panic!("Fusion does not yet support interleaved complex tensors")
    }
}
