use crate::backend::Backend;
use crate::check::TensorCheck;
use crate::{Tensor, TensorPrimitive, check, s};

/// Applies the rectified linear unit function element-wise
/// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
///
#[cfg_attr(doc, doc = "$$\\text{ReLU}\\(x\\) = \\(x\\)^+ = \\max\\(0, x\\)$$")]
#[cfg_attr(not(doc), doc = "`ReLU(x) = max(0, x)`")]
pub fn relu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

/// Applies the leaky rectified linear unit function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{LeakyReLU}\(x\) = \max\(0,x\) + \text{negative\\_slope} \cdot \min\(0, x\)
$$

or

$$
\text{LeakyReLU}(x) =
 \begin{cases}
     x & \text{if } x \geq 0 \newline
     \text{negative\\_slope} \cdot x & \text{otherwise}
 \end{cases}
$$
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`f(x) =`\n- `x for x >= 0`\n- `negative_slope * x if x < 0`"
)]
pub fn leaky_relu<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
    negative_slope: f64,
) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::leaky_relu(
        tensor.primitive.tensor(),
        negative_slope.into(),
    )))
}

/// Applies the Gaussian Error Linear Units function as described in the paper
/// [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{GELU}(x)
= x \cdot \Phi(x)
= x \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

where $\Phi(x)$ is the cumulative distribution function for the Gaussian distribution.
"#
)]
#[cfg_attr(
    not(doc),
    doc = r#"
`GELU(x) = x * Φ(x) = x * 1/2 * (1 + erf(x / sqrt(2)))`

where `Φ(x)` is the cumulative distribution function for the Gaussian distribution.
"#
)]
pub fn gelu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::gelu(tensor.primitive.tensor())))
}

/// Applies Parametric ReLu activation function as described in the paper
/// [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852).
///
/// - The tensor is assumed to be of shape `[batch_size, channels, ...]`.
/// - `alpha` is assumed to be of shape `[channels]` or `[1]`.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{PReLU}\(x\) = \max\(0,x\) + \alpha \cdot \min\(0, x\)
$$

or

$$
\text{PReLU}(x) =
 \begin{cases}
     x & \text{if } x \geq 0 \newline
     \alpha x & \text{otherwise}
 \end{cases}
$$
"#
)]
#[cfg_attr(not(doc), doc = "`PReLu(x) = max(0,x) + alpha * min(0,x)`")]
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

    Tensor::from_primitive(TensorPrimitive::Float(B::prelu(
        tensor.primitive.tensor(),
        weight.primitive.tensor(),
    )))
}

/// Applies the softmax function on the input tensor along the given dimension.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{softmax}\(x_i\) = \frac{\exp\(x_i\)}{\sum_j \exp\(x_j\)}
$$
"#
)]
#[cfg_attr(not(doc), doc = "`softmax(x_i) = exp(x_i) / sum_j(exp(x_j))`")]
///
/// # Arguments
/// - `dim`: the dimension along which Softmax will be computed.
///
/// # Panics
/// - If `dim` is outside [0, D)
pub fn softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    let tensor = tensor.exp();
    let tensor_tmp = tensor.clone().sum_dim(dim);

    tensor.div(tensor_tmp)
}

/// Applies the softmin function on the input tensor along the given dimension.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{softmin}\(x_i\) = \frac{\exp\(-x_i\)}{\sum_j \exp\(-x_j\)}
$$
"#
)]
#[cfg_attr(not(doc), doc = "`softmin(x_i) = exp(-x_i) / sum_j(exp(-x_j)`")]
///
/// # Arguments
/// - `dim`: the dimension along which Softmax will be computed.
///
/// # Panics
/// - If `dim` is outside [0, D)
pub fn softmin<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmin", dim));
    softmax(tensor.neg(), dim)
}

/// Applies the SoftPlus function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{softplus}\(x\) = \frac{1}{\beta}\log\(1 + \exp\(\beta x\)\)
$$
"#
)]
#[cfg_attr(not(doc), doc = "`softplus(x_i) = log(1 + exp(beta * x_i)) / beta`")]
///
/// The SoftPlus function is a smooth approximation of the ReLU function.
pub fn softplus<const D: usize, B: Backend>(tensor: Tensor<B, D>, beta: f64) -> Tensor<B, D> {
    let tensor = (tensor.mul_scalar(beta).exp() + 1).log();
    tensor.div_scalar(beta)
}

/// Applies the "quiet softmax" function on the input tensor along the given dimension.
///
/// Also referred to as [`softmax1`](https://www.evanmiller.org/attention-is-off-by-one.html).
///
/// This function is similar to the softmax function, but it allows for "no selection" when
/// all the outputs are close to zero.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{quiet\\_softmax}\(x_i\) = \frac{\exp\(x_i\)}{1 + \sum_j \exp\(x_j\)}
$$
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`quiet_softmax(x_i) = exp(x_i) / [ 1 + sum_j(exp(x_j)) ]`"
)]
///
/// # Arguments
/// - `dim`: the dimension along which Softmax will be computed.
///
/// # Panics
/// - If `dim` is outside [0, D)
pub fn quiet_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("softmax", dim));

    let max_vals = tensor.clone().detach().max_dim(dim);
    let exp_x = (tensor - max_vals.clone()).exp();
    let sum_exp = exp_x.clone().sum_dim(dim);

    exp_x.div(sum_exp + max_vals.neg().exp())
}

/// Applies the log softmax function on the input tensor along the given dimension.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{log\\_softmax}\(x_i\)
= \log\left(\text{softmax}\(x_i\)\right)
= \log\left(\frac{\exp\(x_i\)}{\sum_j \exp\(x_j\)}\right)
$$
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`log_softmax(x_i) = log(softmax(x_i)) = log(exp(x_i) / sum_j(exp(x_j)))`"
)]
///
/// # Arguments
/// - `dim`: the dimension along which Softmax will be computed.
///
/// # Panics
/// - If `dim` is outside [0, D)
pub fn log_softmax<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    check!(TensorCheck::dim_ops::<D>("log softmax", dim));

    let tensor = tensor.clone() - tensor.detach().max_dim(dim);
    let tensor_tmp = tensor.clone().exp().sum_dim(dim).log();

    tensor.sub(tensor_tmp)
}

/// Applies the sigmoid function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{sigmoid}\(x\)
= \sigma(x)
= \frac{1}{1 + \exp(-x)}
$$
"#
)]
#[cfg_attr(not(doc), doc = "`sigmoid(x) = 1 / (1 + exp(-x))`")]
pub fn sigmoid<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::sigmoid(
        tensor.primitive.tensor(),
    )))
}

/// Applies the hard sigmoid function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{hard\\_sigmoid}\(x\) = \max(0, \min(1, \alpha \cdot x + \beta))
$$
"#
)]
#[cfg_attr(not(doc), doc = "`hard_sigmoid(x) = max(0, min(1, alpha * x + beta))`")]
pub fn hard_sigmoid<const D: usize, B: Backend>(
    tensor: Tensor<B, D>,
    alpha: f64,
    beta: f64,
) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::hard_sigmoid(
        tensor.primitive.tensor(),
        alpha.into(),
        beta.into(),
    )))
}

/// Applies the log sigmoid function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{log\\_sigmoid}\(x\) = \log\left(\frac{1}{1 + \exp(-x)}\right)
$$
"#
)]
#[cfg_attr(not(doc), doc = "`log_sigmoid(x) = log(1 / (1 + exp(-x)))`")]
pub fn log_sigmoid<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(B::log_sigmoid(
        tensor.primitive.tensor(),
    )))
}

/// Applies the SiLU function (also known as the swish function) element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{SiLU}\(x\) = x \cdot \sigma(x) = \frac{x}{1 + \exp(-x)}
$$
"#
)]
#[cfg_attr(not(doc), doc = "`SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`")]
pub fn silu<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone().mul(sigmoid(tensor))
}

/// Applies the hard swish function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{hard\_swish}\(x\) = x \cdot \text{hard\_sigmoid}(x) = x \cdot \max(0, \min(1, \frac{x}{6} + 0.5))
$$
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`hard_swish(x) = x * hard_sigmoid(x) = x * max(0, min(1, x/6 + 0.5))`"
)]
pub fn hard_swish<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone().mul(hard_sigmoid(tensor, 1.0 / 6.0, 0.5))
}

/// Applies the Mish function as described in the paper in
/// [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{Mish}\(x\)
= x \cdot \tanh(\text{Softplus}(x))
= \tanh\left(\log\(1 + \exp\(x\)\)\right)
$$
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`mish(x) = x * tanh(softplus(x)) = tanh(log(1 + exp(x)))`"
)]
pub fn mish<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone().mul(softplus(tensor, 1.0).tanh())
}

/// Applies the tanh function element-wise.
pub fn tanh<const D: usize, B: Backend>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.tanh()
}

/// Applies the Continuously Differentiable Exponential Linear Unit function element-wise.
///
#[cfg_attr(
    doc,
    doc = r#"
$$
\text{CELU}(x) =
 \begin{cases}
     x & \text{if } x \geq 0 \newline
     \alpha \cdot \left(\exp\left(\frac{x}{\alpha}\right) - 1\right) & \text{otherwise}
 \end{cases}
$$
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`celu(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`"
)]
///
/// See also [CELU](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
///
/// # Arguments
/// - `alpha`: scaling parameter for the negative part. Default is 1.0.
pub fn celu<const D: usize, B: Backend>(tensor: Tensor<B, D>, alpha: f64) -> Tensor<B, D> {
    let mask = tensor.clone().greater_elem(0);
    let positive = tensor.clone().mask_fill(mask.clone().bool_not(), 0.0);
    let negative = tensor
        .div_scalar(alpha)
        .exp()
        .sub_scalar(1)
        .mul_scalar(alpha)
        .mask_fill(mask, 0.0);
    positive.add(negative)
}

/// Applies the gated linear unit function.
///
/// GLU(a,b)=a⊗σ(b) where `a` is the first half of the input matrices and `b` is the second half.
///
/// **Note**:
/// * The size of the input tensor along `dim` must be divisible by 2.
///
/// ### Arguments
/// * `tensor` - The input tensor.
///
/// ### Returns
/// * A tensor with the same shape as the input, except the size along `dim` is halved.
pub fn glu<const D: usize, B: Backend>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    // TODO: Handle negative indices with AsIndex for compatibility with Pytorch nn.GLU.

    assert!(
        tensor.dims()[dim].is_multiple_of(2),
        "Input tensor along dimension {dim} must have an even size. N is divisible by 2."
    );
    let new_len = tensor.dims()[dim] / 2;

    let a = tensor.clone().slice_dim(dim, s![0..new_len]);
    let b = tensor.slice_dim(dim, s![new_len..new_len * 2]);

    a.mul(sigmoid(b))
}
