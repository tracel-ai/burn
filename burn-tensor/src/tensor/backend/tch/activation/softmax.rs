use crate::{activation::Softmax, backend::tch::TchTensor, TchElement};
use rand::distributions::Standard;

impl<E, const D: usize> Softmax<E, D> for TchTensor<E, D>
where
    E: TchElement,
    Standard: rand::distributions::Distribution<E>,
{
    fn softmax(&self, dim: usize) -> Self {
        let tensor = self.tensor.softmax(dim as i64, self.kind.kind());

        Self {
            tensor,
            shape: self.shape,
            kind: self.kind.clone(),
        }
    }
}
