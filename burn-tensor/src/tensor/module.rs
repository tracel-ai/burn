use crate::{backend::Backend, Tensor};

pub fn embedding<B>(weights: &Tensor<B, 2>, indexes: &Tensor<B::IntegerBackend, 2>) -> Tensor<B, 3>
where
    B: Backend,
{
    Tensor::new(B::embedding(&weights.value, &indexes.value))
}
