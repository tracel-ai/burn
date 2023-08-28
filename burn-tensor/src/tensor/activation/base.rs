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

/// Applies the tanh function
pub fn tanh<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.tanh()
}
