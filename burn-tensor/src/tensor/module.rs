use crate::{
    backend::Backend,
    ops::{ConvOptions, ConvTransposeOptions},
    Int, Tensor,
};

/// Applies the [embedding module](crate::ops::ModuleOps::embedding).
pub fn embedding<B>(weights: Tensor<B, 2>, indices: Tensor<B, 2, Int>) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::embedding(weights.primitive, indices.primitive))
}

/// Applies a [1D convolution](crate::ops::ModuleOps::conv2d).
pub fn conv1d<B>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    options: ConvOptions<1>,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::conv1d(
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        options,
    ))
}

/// Applies a [2D convolution](crate::ops::ModuleOps::conv2d).
pub fn conv2d<B>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    options: ConvOptions<2>,
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::conv2d(
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        options,
    ))
}

/// Applies a [1D transposed convolution](crate::ops::ModuleOps::conv_transpose1d).
pub fn conv_transpose1d<B>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    options: ConvTransposeOptions<1>,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::conv_transpose1d(
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        options,
    ))
}

/// Applies a [2D transposed convolution](crate::ops::ModuleOps::conv_transpose2d).
pub fn conv_transpose2d<B>(
    x: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    options: ConvTransposeOptions<2>,
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::conv_transpose2d(
        x.primitive,
        weight.primitive,
        bias.map(|b| b.primitive),
        options,
    ))
}

/// Applies a [1D max pooling](crate::ops::ModuleOps::max_pool1d).
pub fn max_pool1d<B>(
    x: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::max_pool1d(
        x.primitive,
        kernel_size,
        stride,
        padding,
        dilation,
    ))
}

/// Applies a [2D max pooling](crate::ops::ModuleOps::max_pool2d).
pub fn max_pool2d<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::max_pool2d(
        x.primitive,
        kernel_size,
        stride,
        padding,
        dilation,
    ))
}

/// Applies a [2D avg pooling](crate::ops::ModuleOps::avg_pool2d).
pub fn avg_pool2d<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::avg_pool2d(
        x.primitive,
        kernel_size,
        stride,
        padding,
        count_include_pad,
    ))
}

/// Applies a [1D avg pooling](crate::ops::ModuleOps::avg_pool1d).
pub fn avg_pool1d<B>(
    x: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::avg_pool1d(
        x.primitive,
        kernel_size,
        stride,
        padding,
        count_include_pad,
    ))
}

/// Applies a [1D max pooling](crate::ops::ModuleOps::max_pool1d).
pub fn max_pool1d_with_indices<B>(
    x: Tensor<B, 3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> (Tensor<B, 3>, Tensor<B, 3, Int>)
where
    B: Backend,
{
    let output = B::max_pool1d_with_indices(x.primitive, kernel_size, stride, padding, dilation);

    (Tensor::new(output.output), Tensor::new(output.indices))
}

/// Applies a [2D max pooling with indices](crate::ops::ModuleOps::max_pool2d_with_indices).
pub fn max_pool2d_with_indices<B>(
    x: Tensor<B, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (Tensor<B, 4>, Tensor<B, 4, Int>)
where
    B: Backend,
{
    let output = B::max_pool2d_with_indices(x.primitive, kernel_size, stride, padding, dilation);

    (Tensor::new(output.output), Tensor::new(output.indices))
}

/// Applies a [2D adaptive avg pooling](crate::ops::ModuleOps::adaptive_avg_pool2d).
pub fn adaptive_avg_pool2d<B>(x: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::adaptive_avg_pool2d(x.primitive, output_size))
}

/// Applies a [1D adaptive avg pooling](crate::ops::ModuleOps::adaptive_avg_pool1d).
pub fn adaptive_avg_pool1d<B>(x: Tensor<B, 3>, output_size: usize) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::adaptive_avg_pool1d(x.primitive, output_size))
}
