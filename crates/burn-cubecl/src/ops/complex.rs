use burn_backend::{ComplexTensor, TypedDevice};

use crate::{CubeBackend, CubeRuntime};

impl<R> TypedDevice<Self> for CubeBackend<R>
where
    R: CubeRuntime,
{
    fn complex_device(_tensor: &ComplexTensor<Self>) -> R::CubeDevice {
        panic!("Cube backend does not yet support interleaved complex tensors")
    }
}
