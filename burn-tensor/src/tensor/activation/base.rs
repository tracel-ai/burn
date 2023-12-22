use crate::backend::Backend;
use crate::check::TensorCheck;
use crate::{check, Tensor};
use crate::{ElementPrecision, Precision};

/// Applies the rectified linear unit function.
pub fn relu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

/// Applies the Gaussian Error Linear Units function as described in the paper in [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
pub fn gelu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(B::gelu(tensor.primitive))
}

/// Applies the softmax function on the input tensor along the given dimension.
///
/// `softmax(x_i) = exp(x_i) / sum_j(exp(x_j))`
///
/// # Notes
///
/// The dimension argument `dim` specifies the dimension along which the function will be computed.
/// It must in the range of `0` and `D-1`.
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    let tensor = tensor.exp();
    let tensor_tmp = tensor.clone().sum_dim(dim);

    tensor.div(tensor_tmp)
}

/// Applies the softplus function
///
/// `softplus(x_i) = log(1 + exp(\beta x_i)) / \beta`
pub fn softplus<const D: usize, B: Backend>(tensor: Tensor<B, D>, beta: f64) -> Tensor<B, D> {
    let tensor = (tensor.mul_scalar(beta).exp() + 1).log();
    tensor.div_scalar(beta)
}

/// Applies the "quiet softmax" function on the input tensor along the given dimension.
/// This function is similar to the softmax function, but it allows for "no selection", e.g.,
/// all outputs can tend to zero.
///
/// `softmax(x_i) = exp(x_i) / [ 1 + sum_j(exp(x_j)) ]`
///
/// # Notes
///
/// The dimension argument `dim` specifies the dimension along which the function will be computed.
/// It must in the range of `0` and `D-1`.
pub fn quiet_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    let tensor = tensor.exp();
    let tensor_tmp = tensor.clone().sum_dim(dim);

    tensor.div(tensor_tmp + 1)
}

/// Applies the log softmax function on the input tensor along the given dimension.
///
/// `log_softmax(x_i) = log(softmax(x_i)) = log(exp(x_i) / sum_j(exp(x_j)))`
///
/// # Notes
///
/// The dimension argument `dim` specifies the dimension along which the function will be computed.
/// It must in the range of `0` and `D-1`.
pub fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("log softmax", dim));

    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    let tensor_tmp = tensor.clone().exp().sum_dim(dim).log();

    tensor.sub(tensor_tmp)
}

/// Applies the sigmoid function.
pub fn sigmoid<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    log_sigmoid(tensor).exp()
}

/// Applies the log sigmoid function.
pub fn log_sigmoid<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    match B::FloatElem::precision() {
        Precision::Half => {
            let tensor_full = tensor.to_full_precision();
            let tensor_tmp = tensor_full.neg().exp().add_scalar(1.0_f32).log().neg();
            Tensor::from_full_precision(tensor_tmp)
        }
        _ => tensor.neg().exp().add_scalar(1.0_f32).log().neg(),
    }
}

/// Applies the silu function
pub fn silu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone().mul(sigmoid(tensor))
}

/// Applies the Mish function as described in the paper in [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).
///
/// `mish(x_i) = x_i \times tanh(softplus(x_i))`
pub fn mish<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone().mul(softplus(tensor, 1.0).tanh())
}

/// Applies the tanh function
pub fn tanh<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.tanh()
}
