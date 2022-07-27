use crate::tensor::{backend::tch::TchTensor, ops::*, Shape};

impl<P: tch::kind::Element, const D: usize> TensorOpsMatmul<P, D> for TchTensor<P, D> {
    fn matmul(&self, other: &Self) -> Self {
        let tensor = self.tensor.matmul(&other.tensor);
        let kind = self.kind.clone();
        let shape = Shape::from(tensor.size());

        Self {
            kind,
            tensor,
            shape,
        }
    }
}
