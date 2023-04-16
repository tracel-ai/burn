use crate::{backend::Backend, Int, Tensor};

/// Applies the [embedding module](crate::ops::ModuleOps::embedding).
pub fn embedding<B>(weights: Tensor<B, 2>, indexes: Tensor<B, 2, Int>) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::embedding(weights.primitive, indexes.primitive))
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
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
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
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        stride,
        padding,
    ))
}

/// Applies a [1D transposed convolution](crate::ops::ModuleOps::conv_transpose1d).
pub fn conv_transpose1d<B>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
    padding_out: usize,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::conv_transpose1d(
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        stride,
        padding,
        padding_out,
    ))
}

/// Applies a [2D transposed convolution](crate::ops::ModuleOps::conv_transpose2d).
pub fn conv_transpose2d<B>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
    out_padding: [usize; 2],
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::conv_transpose2d(
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        stride,
        padding,
        out_padding,
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
    Tensor::new(B::max_pool2d(x.primitive, kernel_size, stride, padding))
}

/// Applies a [2D max pooling with indexes](crate::ops::ModuleOps::max_pool2d_with_indexes).
pub fn max_pool2d_with_indexes<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (Tensor<B, 4>, Tensor<B, 4, Int>)
where
    B: Backend,
{
    let output = B::max_pool2d_with_indexes(x.primitive, kernel_size, stride, padding);

    (Tensor::new(output.output), Tensor::new(output.indexes))
}
