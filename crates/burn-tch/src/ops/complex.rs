use burn_backend::{ComplexTensor, TypedDevice};

use crate::{LibTorch, LibTorchDevice};

impl TypedDevice<Self> for LibTorch {
    fn complex_device(_tensor: &ComplexTensor<Self>) -> LibTorchDevice {
        panic!("Tch backend does not yet support interleaved complex tensors")
    }
}
