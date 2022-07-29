use crate::tensor::{
    backend::tch::TchTensor,
    ops::{TensorOpsBackend, TensorOpsUtilities},
    Backend, Element, Tensor, TensorType,
};

impl<E, const D: usize, B: Backend> TensorOpsBackend<E, D, B> for TchTensor<E, D>
where
    E: Element + tch::kind::Element,
    B: Backend<E = E> + TensorType<D, B>,
{
    type Output = Tensor<D, B>;

    fn to_backend(&self) -> Self::Output {
        let data = self.to_data();
        <B as TensorType<D, B>>::from_data(data)
    }
}
