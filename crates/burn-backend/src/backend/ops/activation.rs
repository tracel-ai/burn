use crate::tensor::FloatTensor;
use crate::{Backend, Scalar, TensorMetadata, get_or_init_device_settings};
use burn_std::IndexingUpdateOp;
use core::f64::consts::SQRT_2;

/// Activation function operations.
///
/// This trait let backend implementations override activation functions for better performance.
pub trait ActivationOps<B: Backend> {
    /// Applies the LeakyReLU activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `negative_slope` - The negative_slope value that values smaller than 0 are multiplied with.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn leaky_relu(tensor: FloatTensor<B>, negative_slope: Scalar) -> FloatTensor<B> {
        let bool_dtype = get_or_init_device_settings::<B>(&tensor.device()).bool_dtype;
        let mask = B::float_lower_elem(tensor.clone(), 0f32.into(), bool_dtype);
        let scaled_tensor = B::float_mul_scalar(tensor.clone(), negative_slope);

        // Update the tensor where the values are `< 0` by `tensor * negative_slope`.
        B::float_mask_where(tensor, mask, scaled_tensor)
    }

    /// Applies the ReLU activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn relu(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let bool_dtype = get_or_init_device_settings::<B>(&tensor.device()).bool_dtype;
        let mask = B::float_lower_equal_elem(tensor.clone(), 0f32.into(), bool_dtype);

        B::float_mask_fill(tensor, mask, 0f32.into())
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
    fn relu_backward(output: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        let bool_dtype = get_or_init_device_settings::<B>(&output.device()).bool_dtype;
        let mask = B::float_lower_equal_elem(output, 0f32.into(), bool_dtype);

        B::float_mask_fill(grad, mask, 0.into())
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
    fn gelu(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let x = B::float_div_scalar(tensor.clone(), SQRT_2.into());
        let x = B::float_erf(x);
        let x = B::float_add_scalar(x, 1f32.into());
        let x = B::float_mul(tensor, x);

        B::float_div_scalar(x, 2f32.into())
    }
    /// Applies the PReLu activation function.
    /// # Arguments
    /// * `tensor` - The input tensor
    /// * `alpha` - The weight tensor
    fn prelu(tensor: FloatTensor<B>, alpha: FloatTensor<B>) -> FloatTensor<B> {
        let bool_dtype = get_or_init_device_settings::<B>(&tensor.device()).bool_dtype;
        let mask = B::float_lower_elem(tensor.clone(), 0f32.into(), bool_dtype);
        let scaled_tensor = B::float_mul(tensor.clone(), alpha);
        B::float_mask_where(tensor, mask, scaled_tensor)
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
    fn gelu_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        // Exact analytical derivative of erf-based GELU:
        // d/dx [0.5 * x * (1 + erf(x / sqrt(2)))]
        // = 0.5 * (1 + erf(x / sqrt(2))) + x * (1 / sqrt(2 * pi)) * exp(-0.5 * x^2)
        const FRAC_1_SQRT_2PI: f64 = 0.3989422804014327;

        let erf_term = B::float_erf(B::float_div_scalar(x.clone(), SQRT_2.into()));
        let cdf = B::float_mul_scalar(B::float_add_scalar(erf_term, 1.0.into()), 0.5.into());

        let x2 = B::float_mul(x.clone(), x.clone());
        let exp_term = B::float_exp(B::float_mul_scalar(x2, (-0.5).into()));
        let pdf_term = B::float_mul(B::float_mul_scalar(x, FRAC_1_SQRT_2PI.into()), exp_term);

        let dgelu = B::float_add(cdf, pdf_term);
        B::float_mul(dgelu, grad)
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
    fn sigmoid(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let dtype = tensor.dtype();
        let tensor_full = B::float_cast(tensor, burn_std::FloatDType::F32);
        let tensor_tmp = B::float_exp(B::float_neg(B::float_log(B::float_add_scalar(
            B::float_exp(B::float_neg(tensor_full)),
            1.0.into(),
        ))));

        B::float_cast(tensor_tmp, dtype.into())
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
    fn sigmoid_backward(output: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        let value = B::float_mul(
            output.clone(),
            B::float_add_scalar(B::float_neg(output), 1.0.into()),
        );
        B::float_mul(value, grad)
    }

    /// Applies the hard Sigmoid activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `alpha` - The alpha value that the tensor is multiplied with.
    /// * `beta` - The beta value that is added to the tensor
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn hard_sigmoid(tensor: FloatTensor<B>, alpha: Scalar, beta: Scalar) -> FloatTensor<B> {
        let dtype = tensor.dtype();
        let tensor_full = B::float_cast(tensor, burn_std::FloatDType::F32);

        let tensor_tmp = B::float_clamp(
            B::float_add_scalar(B::float_mul_scalar(tensor_full, alpha), beta),
            0.0.into(),
            1.0.into(),
        );

        B::float_cast(tensor_tmp, dtype.into())
    }

    /// Applies the LogSigmoid activation function.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn log_sigmoid(tensor: FloatTensor<B>) -> FloatTensor<B> {
        // To avoid overflow, we use the log-sum-exp trick.
        //
        // ```ignore
        // log(sigmoid(x)) = log(1/(1 + exp(-x)))
        //                 = log(1) - log(1 + exp(-x))
        //                 = -log(1 + exp(-x))
        //                 = -log(exp(0) + exp(-x))
        // ```
        // The `exp(t)` of even a moderate-magnitude positive number can be astronomically huge, so we
        // subtract the `max(t, 0)` of each value (where `t = -x` in this case). This results in the
        // following equivalence:
        // ```ignore
        // log(sigmoid(x)) = -(max(-x, 0) + log(exp(-max(-x, 0)) + exp(-x - max(-x, 0))))
        // ```
        //
        // This extends the range of values for which we obtain accurate results.

        // max(-x, 0)
        let bool_dtype = get_or_init_device_settings::<B>(&tensor.device()).bool_dtype;
        let tensor_neg = B::float_neg(tensor);
        let mask = B::float_lower_elem(tensor_neg.clone(), 0f32.into(), bool_dtype);
        let max_elem = B::float_mask_fill(tensor_neg.clone(), mask, 0f32.into());
        let max_elem_neg = B::float_neg(max_elem.clone());

        // z = exp(-max(-x, 0)) + exp(-x - max(-x, 0))
        let z = B::float_add(
            B::float_exp(max_elem_neg.clone()),
            B::float_exp(B::float_sub(tensor_neg, max_elem.clone())),
        );

        // -max(-x, 0) - log(-z)
        B::float_sub(max_elem_neg, B::float_log(z))
    }

    /// Applies the softmax function along the given dimension.
    ///
    /// Exponentiates the stable log-softmax decomposition to preserve small gradients
    /// when the largest probability rounds to one.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension along which softmax is computed.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn softmax(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        B::float_exp(Self::log_softmax(tensor, dim))
    }

    /// Applies the log-softmax function along the given dimension.
    ///
    /// Excludes exactly one maximum from the exponential sum and uses `log1p` to
    /// preserve small log probabilities. The selected maximum remains differentiable.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension along which log-softmax is computed.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn log_softmax(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        let shape = tensor.shape();
        if shape.num_elements() == 0 {
            // Retain backend-specific validation and behavior for empty reductions.
            let max = B::float_max_dim(B::float_detach(tensor.clone()), dim);
            let shifted = B::float_sub(tensor, max);
            let correction = B::float_log(B::float_sum_dim(B::float_exp(shifted.clone()), dim));
            return B::float_sub(shifted, correction);
        }
        let device = tensor.device();
        let settings = get_or_init_device_settings::<B>(&device);
        let indices = B::float_argmax(tensor.clone(), dim, settings.int_dtype);
        let max = B::float_gather(dim, tensor.clone(), indices.clone());
        let selected = B::bool_scatter(
            dim,
            B::bool_zeros(shape, &device, settings.bool_dtype),
            indices,
            B::bool_ones(max.shape(), &device, settings.bool_dtype),
            IndexingUpdateOp::Assign,
        );

        // Preserve NaNs for non-finite maxima, including single-class invalid inputs.
        let finite_max = B::float_lower_elem(
            B::float_abs(max.clone()),
            f64::INFINITY.into(),
            settings.bool_dtype,
        );
        let selected = B::bool_and(selected, finite_max);

        // An exact constant zero at one argmax avoids cancellation in its gradient.
        // Other logits still depend on the gathered maximum; do not detach it.
        let shifted = B::float_mask_fill(B::float_sub(tensor, max), selected.clone(), 0.0.into());
        let others = B::float_mask_fill(B::float_exp(shifted.clone()), selected, 0.0.into());
        let correction = B::float_log1p(B::float_sum_dim(others, dim));
        B::float_sub(shifted, correction)
    }

    /// Applies the softmin function along the given dimension.
    ///
    /// Equivalent to `softmax(-tensor, dim)`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension along which softmin is computed.
    ///
    /// # Returns
    ///
    /// The output tensor.
    fn softmin(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        Self::softmax(B::float_neg(tensor), dim)
    }

    /// Applies the LogSigmoid activation function backward.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor.
    /// * `grad` - The gradient.
    ///
    /// # Returns
    ///
    /// The output gradient.
    fn log_sigmoid_backward(x: FloatTensor<B>, grad: FloatTensor<B>) -> FloatTensor<B> {
        // Derivative of -max(-x, 0) - log(exp(-max(-x, 0)) - exp(-x - max(-x, 0)))) is
        // -max_derive - (-max_derive * exp(-max(-x, 0)) + (-1 - max_derive) * exp(-x - max(-x, 0))) / z
        // where z = exp(-max(-x, 0)) + exp(-x - max(-x, 0))
        //
        // This simplifies to:
        // -max_derive - (z-1)/z if x is >= 0
        // -max_derive + (z-1)/z if x is < 0

        let shape = x.shape();
        let dtype = x.dtype();
        let device = x.device();
        let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;

        // max(-x, 0)
        let x_neg = B::float_neg(x);
        let mask = B::float_lower_elem(x_neg.clone(), 0f32.into(), bool_dtype); // -x < 0 or x >= 0
        let max_elem = B::float_mask_fill(x_neg.clone(), mask.clone(), 0f32.into());

        // z = exp(-max(-x, 0)) + exp(-x - max(-x, 0))
        let z = B::float_add(
            B::float_exp(B::float_neg(max_elem.clone())),
            B::float_exp(B::float_sub(x_neg, max_elem)),
        );

        // Derivative of max(-x, 0) is 1 if x < 0 or 0 if x >= 0
        let ones = B::float_ones(shape, &device, dtype.into());
        let max_derive = B::float_mask_fill(ones.clone(), mask.clone(), 0f32.into());
        let sign = B::float_mask_fill(ones.clone(), mask, (-1f32).into());

        // grad * (max_derive - sign * (1 - (1 / z)))
        B::float_mul(
            grad,
            B::float_sub(
                max_derive,
                B::float_mul(sign, B::float_sub(ones, B::float_recip(z))),
            ),
        )
    }
}
