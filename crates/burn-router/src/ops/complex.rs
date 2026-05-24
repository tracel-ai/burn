use burn_backend::{TypedDevice, UnimplementedTensorPrimitive};
use burn_std::Complex;

use crate::{BackendRouter, RunnerChannel};

impl<R: RunnerChannel> TypedDevice<Self> for BackendRouter<R> {
    fn complex_device(tensor: &UnimplementedTensorPrimitive<Complex<R::FloatElem>>) -> <Self as burn_backend::BackendTypes>::Device {
        panic!("Router backend does not yet support interleaved complex tensors")
    }
}