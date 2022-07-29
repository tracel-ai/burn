use crate::tensor::{
    backend::tch::TchTensor,
    ops::{TensorOpsBackend, TensorOpsUtilities},
    Backend, Element, Tensor, TensorType,
};

impl<E, const D: usize, B1, B2> TensorOpsBackend<E, D, B1, B2> for TchTensor<E, D>
where
    E: Element + tch::kind::Element,
    B1: Backend<E = E> + TensorType<D, B1>,
    B2: Backend<E = E> + TensorType<D, B2>,
{
    type Output = Tensor<D, B2>;

    fn to_backend(&self) -> Self::Output {
        let data = self.to_data();
        <B2 as TensorType<D, B2>>::from_data(data)
    }
}
