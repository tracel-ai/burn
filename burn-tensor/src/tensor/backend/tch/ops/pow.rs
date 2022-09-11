use crate::{
    tensor::{backend::tch::TchTensor, ops::*},
    TchElement,
};

impl<E, const D: usize> TensorOpsPow<E, D> for TchTensor<E, D>
where
    E: TchElement,
{
    fn powf(&self, value: f32) -> Self {
        let tensor = self.tensor.pow_tensor_scalar(value as f64);
        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            shape,
            kind,
        }
    }
}
