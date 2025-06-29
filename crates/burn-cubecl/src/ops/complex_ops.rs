use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement, tensor::CubeTensor,
};
use burn_tensor::{Device, Distribution, Shape, TensorData, ops::ComplexTensorOps};
use cubecl::server::ComputeServer;

impl<R, F, I, BT> ComplexTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    R::Server: ComputeServer,
    R::Device: burn_tensor::backend::DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn complex_from_data(_data: TensorData, _device: &Device<Self>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &Device<Self>,
    ) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_full(
        _shape: Shape,
        _fill_value: burn_tensor::Complex32,
        _device: &Device<Self>,
    ) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_shape(_tensor: &CubeTensor<R>) -> Shape {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_to_data(_tensor: &CubeTensor<R>) -> TensorData {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_device(_tensor: &CubeTensor<R>) -> Device<Self> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_to_device(_tensor: CubeTensor<R>, _device: &Device<Self>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_into_data(_tensor: CubeTensor<R>) -> TensorData {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_reshape(_tensor: CubeTensor<R>, _shape: Shape) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_transpose(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_add(_lhs: CubeTensor<R>, _rhs: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_sub(_lhs: CubeTensor<R>, _rhs: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_mul(_lhs: CubeTensor<R>, _rhs: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_div(_lhs: CubeTensor<R>, _rhs: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_neg(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_conj(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_real(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_imag(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_abs(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_arg(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_from_parts(_real: CubeTensor<R>, _imag: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_from_polar(_magnitude: CubeTensor<R>, _phase: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_exp(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_log(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_powc(_lhs: CubeTensor<R>, _rhs: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_sqrt(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_sin(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_cos(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }

    fn complex_tan(_tensor: CubeTensor<R>) -> CubeTensor<R> {
        unimplemented!("Complex tensor operations are not yet implemented for CubeCL backend")
    }
}
