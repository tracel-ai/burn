use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

#[derive(Default)]
pub struct TensorCache<B: Backend, const D: usize> {
    pub(crate) state: Option<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> TensorCache<B, D> {
    pub fn new() -> Self {
        Self::default()
    }
}
