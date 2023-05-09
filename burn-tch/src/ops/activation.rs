use crate::{element::TchElement, TchBackend, TchTensor};
use burn_tensor::ops::ActivationOps;

impl<E: TchElement> ActivationOps<TchBackend<E>> for TchBackend<E> {
    fn relu<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.relu_(), |tensor| tensor.relu())
    }
    fn gelu<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.gelu_("none"),
            |tensor| tensor.gelu("none"),
        )
    }
    fn gelu_backward<const D: usize>(
        tensor: TchTensor<E, D>,
        grad: TchTensor<E, D>,
    ) -> TchTensor<E, D> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.gelu_backward(&grad.tensor, "none");

        TchTensor::from_existing(tensor, storage)
    }
}
