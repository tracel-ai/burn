use crate::tensor::backend::backend::Backend;
use crate::tensor::{backend::autodiff::ADTensor, ops::*};

impl<B: Backend, P, const D: usize> TensorOpsDetach<P, D> for ADTensor<D, B> {
    fn detach(self) -> Self {
        let tensor = self.tensor();
        // let device = tensor.device();
        Self::from_tensor(tensor)
    }
}
