use crate::tensor::backend::Backend;
use crate::tensor::{backend::autodiff::ADTensor, ops::*};

impl<B: Backend, P, const D: usize> TensorOpsDetach<P, D> for ADTensor<D, B> {
    fn detach(self) -> Self {
        let tensor = self.tensor();
        Self::from_tensor(tensor.detach())
    }
}
