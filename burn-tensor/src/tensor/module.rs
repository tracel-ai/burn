use crate::{backend::Backend, Tensor};

/// Applies the [embedding module](crate::ops::ModuleOps::embedding).
pub fn embedding<B>(weights: Tensor<B, 2>, indexes: Tensor<B::IntegerBackend, 2>) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::embedding(weights.value, indexes.value))
}

/// Applies a [1D convolution](crate::ops::ModuleOps::conv2d).
pub fn conv1d<B>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::conv1d(
        x.value,
        weight.value,
        bias.map(|b| b.value),
        stride,
        padding,
    ))
}

/// Applies a [2D convolution](crate::ops::ModuleOps::conv2d).
pub fn conv2d<B>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::conv2d(
        x.value,
        weight.value,
        bias.map(|b| b.value),
        stride,
        padding,
    ))
}

/// Applies a [2D max pooling](crate::ops::ModuleOps::max_pool2d).
pub fn max_pool2d<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::max_pool2d(x.value, kernel_size, stride, padding))
}

/// Applies a [2D max pooling with indexes](crate::ops::ModuleOps::max_pool2d_with_indexes).
pub fn max_pool2d_with_indexes<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (Tensor<B, 4>, Tensor<B::IntegerBackend, 4>)
where
    B: Backend,
{
    let output = B::max_pool2d_with_indexes(x.value, kernel_size, stride, padding);

    (Tensor::new(output.output), Tensor::new(output.indexes))
}
