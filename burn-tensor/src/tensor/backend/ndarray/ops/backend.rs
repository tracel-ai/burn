use crate::tensor::{
    backend::ndarray::NdArrayTensor,
    ops::{TensorOpsBackend, TensorOpsUtilities},
    Backend, Element, Tensor, TensorType,
};

impl<E, const D: usize, B1: Backend, B2: Backend> TensorOpsBackend<E, D, B1, B2>
    for NdArrayTensor<E, D>
where
    E: Element,
    B1: Backend<E = E> + TensorType<D, B1>,
    B2: Backend<E = E> + TensorType<D, B2>,
{
    type Output = Tensor<D, B2>;

    fn to_backend(&self) -> Self::Output {
        let data = self.to_data();
        <B2 as TensorType<D, B2>>::from_data(data)
    }
}
