use burn_backend::TypedDevice;

use crate::{BackendRouter, RunnerChannel};

impl<R: RunnerChannel> TypedDevice<Self> for BackendRouter<R> {
    fn complex_device(
        _tensor: &<Self as burn_backend::BackendTypes>::ComplexTensorPrimitive,
    ) -> <Self as burn_backend::BackendTypes>::Device {
        panic!("Router backend does not yet support interleaved complex tensors")
    }
}
