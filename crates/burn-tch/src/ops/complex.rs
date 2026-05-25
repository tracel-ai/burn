use burn_backend::{TypedDevice, element::Complex};

use crate::{FloatTchElement, LibTorch, LibTorchDevice};

impl<E: FloatTchElement> TypedDevice<Self> for LibTorch<E> {
    fn complex_device(_tensor: &burn_backend::UnimplementedTensorPrimitive<Complex<E>>) -> LibTorchDevice {
        panic!("Tch backend does not yet support interleaved complex tensors")
    }
}