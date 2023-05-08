use libm::sqrt;

use crate::{backend::Backend, ElementConversion};
use core::f64::consts::SQRT_2;

pub trait ActivationOps<B: Backend> {
    fn gelu<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D> {
        let x = B::div_scalar(tensor.clone(), SQRT_2.elem());
        let x = B::erf(x);
        let x = B::add_scalar(x, 1i32.elem());
        let x = B::mul(tensor, x);
        let x = B::div_scalar(x, 2i32.elem());

        x
    }

    fn gelu_backward<const D: usize>(
        input: B::TensorPrimitive<D>,
        output: B::TensorPrimitive<D>,
        grad: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D> {
        let tmp = B::div(output.clone(), input.clone());

        let tmp2 = B::sub_scalar(tmp, 1.elem());
        let tmp2 = B::powf(tmp2, 2.);
        let tmp2 = B::add_scalar(tmp2, 1.elem());

        let constant = sqrt(2.0 / 3.14159) * 0.004475 * 3.;
        let tmp3 = B::mul_scalar(B::powf(input, 2.elem()), constant.elem());

        let df = B::mul_scalar(B::mul(tmp2, tmp3), 0.5.elem());

        B::mul(df, grad)
    }
}
