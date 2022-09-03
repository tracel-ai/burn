use crate::{backend::tch::TchTensor, ops::activation::ReLU, TchElement};

impl<E, const D: usize> ReLU<E, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn relu(&self) -> Self {
        let tensor = self.tensor.relu();

        Self {
            tensor,
            shape: self.shape,
            kind: self.kind.clone(),
        }
    }
}
