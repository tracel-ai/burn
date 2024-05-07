use crate::backend::Backend;
use crate::check::TensorCheck;
use crate::{check, Tensor};

/// Applies the rectified linear unit function as described in the paper [Deep Learning using
/// Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
///
/// `y = max(0, x)`
pub fn relu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

/// Applies the leaky rectified linear unit function.
///
/// f(x) = negative_slope * x for x < 0, f(x) = x for x >= 0
pub fn leaky_relu<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
    negative_slope: f64,
) -> Tensor<B, D> {
    Tensor::from_primitive(B::leaky_relu(
        tensor.primitive,
        crate::ElementConversion::elem(negative_slope),
    ))
}

/// Applies the Gaussian Error Linear Units function as described in the paper [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
pub fn gelu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(B::gelu(tensor.primitive))
}

/// Applies Parametric ReLu activation function as described in the paper [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852).
/// ` PReLu(x) = max(0,x) + \alpha * min(0,x)`
/// tensor is assumed to be of shape \[batch_size, channels, ...\]
/// alpha is assumed to be of shape \[channels\] or \[1\]
pub fn prelu<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
    alpha: Tensor<B, 1>,
) -> Tensor<B, D> {
    check!(TensorCheck::check_prelu_shape::<D>(
        &tensor.shape(),
        &alpha.shape()
    ));

    let weight = if alpha.dims()[0] == 1 {
        // if there is only 1 weight, then reshape it to (1,1,1... D times) so that the rank is D
        alpha.reshape([1; D])
    } else {
        // D>=2 because the case where D==1 and num_weights >1 is handled by check function
        // there is more than 1 weight and rank is more than 2
        let num_weights = alpha.dims()[0];
        let mut s = [1; D];
        s[1] = num_weights;
        // reshape the weights to (1, channels,1 ...)
        alpha.reshape(s)
    };

    Tensor::from_primitive(B::prelu(tensor.primitive, weight.primitive))
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
    Tensor::from_primitive(B::sigmoid(tensor.primitive))
}

/// Applies the log sigmoid function.
pub fn log_sigmoid<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(B::log_sigmoid(tensor.primitive))
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
