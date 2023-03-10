use crate::backend::Backend;
use crate::Tensor;
use crate::{ElementPrecision, Precision};
use core::f64::consts::SQRT_2;

/// Applies the rectified linear unit function.
pub fn relu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

/// Applies the Gaussian Error Linear Units function as described in the paper in [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
pub fn gelu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let x = tensor.clone().div_scalar(SQRT_2).erf().add_scalar(1.0_f32);

    tensor.mul(x) / 2
}

/// Applies the softmax function.
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(tensor, dim).exp()
}

/// Applies the log softmax function.
pub fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let tensor_tmp = match B::FloatElem::precision() {
        Precision::Half => {
            let tensor_full = tensor.to_full_precision();
            let tensor_tmp = tensor_full.exp().sum_dim(dim).log();
            Tensor::from_full_precision(tensor_tmp)
        }
        _ => tensor.clone().exp().sum_dim(dim).log(),
    };

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
