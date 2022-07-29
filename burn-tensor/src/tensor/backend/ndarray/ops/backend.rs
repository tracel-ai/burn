use crate::tensor::{
    backend::ndarray::NdArrayTensor,
    ops::{TensorOpsBackend, TensorOpsUtilities},
    Backend, Element, Tensor, TensorType,
};

impl<E, const D: usize, B: Backend> TensorOpsBackend<E, D, B> for NdArrayTensor<E, D>
where
    E: Element,
    B: Backend<E = E> + TensorType<D, B>,
{
    type Output = Tensor<D, B>;

    fn to_backend(&self) -> Self::Output {
        let data = self.to_data();
        <B as TensorType<D, B>>::from_data(data)
    }
}
