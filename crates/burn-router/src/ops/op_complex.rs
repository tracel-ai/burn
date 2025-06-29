use crate::{BackendRouter, RouterTensor, RunnerChannel};
use burn_tensor::{Device, Distribution, Shape, TensorData, ops::ComplexTensorOps};

impl<R: RunnerChannel> ComplexTensorOps<Self> for BackendRouter<R> {
    fn complex_from_data(_data: TensorData, _device: &Device<Self>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &Device<Self>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_full(
        _shape: Shape,
        _fill_value: R::ComplexElem,
        _device: &Device<Self>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_shape(_tensor: &RouterTensor<R::Client>) -> Shape {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_to_data(_tensor: &RouterTensor<R::Client>) -> TensorData {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_device(_tensor: &RouterTensor<R::Client>) -> Device<Self> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_to_device(
        _tensor: RouterTensor<R::Client>,
        _device: &Device<Self>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_into_data(_tensor: RouterTensor<R::Client>) -> TensorData {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_reshape(_tensor: RouterTensor<R::Client>, _shape: Shape) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_transpose(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_add(
        _lhs: RouterTensor<R::Client>,
        _rhs: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_sub(
        _lhs: RouterTensor<R::Client>,
        _rhs: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_mul(
        _lhs: RouterTensor<R::Client>,
        _rhs: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_div(
        _lhs: RouterTensor<R::Client>,
        _rhs: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_neg(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_conj(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_real(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_imag(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_abs(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_arg(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_from_parts(
        _real: RouterTensor<R::Client>,
        _imag: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_from_polar(
        _magnitude: RouterTensor<R::Client>,
        _phase: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_exp(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_log(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_powc(
        _lhs: RouterTensor<R::Client>,
        _rhs: RouterTensor<R::Client>,
    ) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_sqrt(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_sin(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_cos(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }

    fn complex_tan(_tensor: RouterTensor<R::Client>) -> RouterTensor<R::Client> {
        unimplemented!("Complex tensor operations are not yet implemented for Router backend")
    }
}
