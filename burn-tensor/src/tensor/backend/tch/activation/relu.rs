use crate::{activation::ReLU, backend::tch::TchTensor, TchElement};
use rand::distributions::Standard;

impl<E, const D: usize> ReLU<E, D> for TchTensor<E, D>
where
    E: TchElement,
    Standard: rand::distributions::Distribution<E>,
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
