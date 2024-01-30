use crate::tensor::ops::tensor::FloatTensorOps;
use crate::{backend::Backend, ElementConversion};
use core::f64::consts::SQRT_2;

use super::FloatTensor;

/// Activation function operations.
///
/// This trait let backend implementations override activation functions for better performance.
pub trait ActivationOps<B: Backend> {
    /// Applies the ReLU activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn relu<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D> {
        let mask = B::float_lower_equal_elem(tensor.clone(), 0.elem());

        B::float_mask_fill(tensor, mask, 0.elem())
    }

    /// Applies the ReLU activation function backward.
    ///
    /// # Arguments
    ///
    /// * `output` - The output tensor.
    ///
    /// # Returns
    ///
    /// The gradient.
    fn relu_backward<const D: usize>(
        output: FloatTensor<B, D>,
        grad: FloatTensor<B, D>,
    ) -> FloatTensor<B, D> {
        let mask = B::float_lower_equal_elem(output, 0.elem());

        B::float_mask_fill(grad, mask, 0.elem())
    }

    /// Applies the Gelu activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn gelu<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D> {
        let x = B::float_div_scalar(tensor.clone(), SQRT_2.elem());
        let x = B::float_erf(x);
        let x = B::float_add_scalar(x, 1i32.elem());
        let x = B::float_mul(tensor, x);

        B::float_div_scalar(x, 2i32.elem())
    }

    /// Applies the Gelu activation function backward.
    ///
    /// # Arguments
    ///
    /// * `x` - The tensor.
    /// * `grad` - The gradient.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn gelu_backward<const D: usize>(
        x: FloatTensor<B, D>,
        grad: FloatTensor<B, D>,
    ) -> FloatTensor<B, D> {
        // Derivative of the approximate gelu implementation based on tanh.

        let constant_1 = 0.0356774;
        let constant_2 = 0.797885;
        let constant_3 = 0.0535161;
        let constant_4 = 0.398942;

        let x3 = B::float_powf_scalar(x.clone(), 3.0);

        let c1 = B::float_mul_scalar(x3.clone(), constant_1.elem());
        let c2 = B::float_mul_scalar(x.clone(), constant_2.elem());
        let c3 = B::float_mul_scalar(x3, constant_3.elem());
        let c4 = B::float_mul_scalar(x, constant_4.elem());

        let inner1 = B::float_add(c1, c2);
        let inner2 = B::float_add(c3, c4);

        let tanh = B::float_tanh(inner1);

        let sech = B::float_powf_scalar(tanh.clone(), 2.0);
        let sech = B::float_neg(sech);
        let sech = B::float_add_scalar(sech, 1.elem());

        let y1 = B::float_mul_scalar(tanh, 0.5.elem());
        let y2 = B::float_mul(inner2, sech);
        let y2 = B::float_add_scalar(y2, 0.5.elem());
        let y = B::float_add(y1, y2);

        B::float_mul(y, grad)
    }

    /// Applies the Sigmoid activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn sigmoid<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D> {
        let tensor_full = B::float_to_full_precision(&tensor);
        let tensor_tmp = B::FullPrecisionBackend::float_exp(B::FullPrecisionBackend::float_neg(
            B::FullPrecisionBackend::float_log(B::FullPrecisionBackend::float_add_scalar(
                B::FullPrecisionBackend::float_exp(B::FullPrecisionBackend::float_neg(tensor_full)),
                1.0.elem(),
            )),
        ));

        B::float_from_full_precision(tensor_tmp)
    }

    /// Applies the Sigmoid activation function backward.
    ///
    /// # Arguments
    ///
    /// * `output` - The output tensor of the sigmoid function.
    /// * `grad` - The gradient.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn sigmoid_backward<const D: usize>(
        output: FloatTensor<B, D>,
        grad: FloatTensor<B, D>,
    ) -> FloatTensor<B, D> {
        let value = B::float_mul(
            output.clone(),
            B::float_add_scalar(B::float_neg(output), 1.0.elem()),
        );
        B::float_mul(value, grad)
    }
}
