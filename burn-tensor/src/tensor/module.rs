use crate::{backend::Backend, Tensor};

pub fn embedding<B>(weights: &Tensor<B, 2>, indexes: &Tensor<B::IntegerBackend, 2>) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::embedding(&weights.value, &indexes.value))
}

pub fn conv1d<B>(
    x: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Option<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::conv1d(
        &x.value,
        &weight.value,
        &bias.clone().map(|b| b.value),
        stride,
        padding,
    ))
}

pub fn conv2d<B>(
    x: &Tensor<B, 4>,
    weight: &Tensor<B, 4>,
    bias: &Option<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Tensor<B, 4>
where
    B: Backend,
{
    Tensor::new(B::conv2d(
        &x.value,
        &weight.value,
        &bias.clone().map(|b| b.value),
        stride,
        padding,
    ))
}
