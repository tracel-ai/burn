use crate::graph::TensorId;
use burn_tensor::Shape;
use std::sync::Arc;

#[derive(new, Clone, Debug)]
pub struct FusionTensor {
    shape: Vec<usize>,
    id: Arc<TensorId>,
}

impl FusionTensor {
    pub(crate) fn shape<const D: usize>(&self) -> Shape<D> {
        Shape::from(self.shape.clone())
    }
    pub fn can_mut(&self) -> bool {
        Arc::strong_count(&self.id) <= 2
    }
}
