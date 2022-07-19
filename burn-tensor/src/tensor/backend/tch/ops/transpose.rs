use crate::{backend::tch::TchTensor, Shape, TensorOpsTranspose};

impl<P: tch::kind::Element, const D: usize> TensorOpsTranspose<P, D> for TchTensor<P, D> {
    fn transpose(&self) -> Self {
        let tensor = self.tensor.transpose(-2, -1);
        let kind = self.kind.clone();
        let shape = Shape::from(tensor.size());

        Self {
            kind,
            tensor,
            shape,
        }
    }
}
