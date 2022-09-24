use crate::backend::Backend;
use crate::Tensor;
use crate::{ElementConversion, ElementPrecision, Precision};

/// Applies the rectified linear unit function.
pub fn relu<const D: usize, B: Backend>(tensor: &Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

/// Applies the Gaussian Error Linear Units function as described in the paper in [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
pub fn gelu<const D: usize, B: Backend>(tensor: &Tensor<B, D>) -> Tensor<B, D> {
    let x = tensor
        .div_scalar(&2.0_f32.sqrt().to_elem())
        .erf()
        .add_scalar(&1.0_f32.to_elem());
    tensor.mul(&x).mul_scalar(&0.5_f32.to_elem())
}

/// Applies the softmax function.
pub fn softmax<const D: usize, B: Backend>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(tensor, dim).exp()
}

/// Applies the log softmax function.
pub fn log_softmax<const D: usize, B: Backend>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let tensor_tmp = match B::Elem::precision() {
        Precision::Half => {
            let tensor_full = tensor.to_full_precision();
            let tensor_tmp = tensor_full.exp().sum_dim(dim).log();
            Tensor::from_full_precision(tensor_tmp)
        }
        _ => tensor.exp().sum_dim(dim).log(),
    };

    tensor.sub(&tensor_tmp)
}
