use burn_backend::{DeviceOps, TypedDevice, UnimplementedTensorPrimitive};
use burn_std::Complex;
use cubecl::server::ComputeServer;

use crate::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

impl<R, F, I, BT> TypedDevice<CubeBackend<R, F, I, BT>> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn complex_device(_tensor: &UnimplementedTensorPrimitive<Complex<F>>) -> R::CubeDevice {
        panic!("Cube backend does not yet support interleaved complex tensors")
    }   
}