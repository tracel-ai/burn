use crate::{element::TchElement, TchBackend, TchTensor};
use burn_tensor::ops::ActivationOps;

impl<E: TchElement> ActivationOps<TchBackend<E>> for TchBackend<E> {
    fn gelu<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(
            |mut tensor| tensor.gelu_("tanh"),
            |tensor| tensor.gelu("tanh"),
        )
    }
}
