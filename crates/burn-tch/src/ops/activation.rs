use crate::{element::TchElement, LibTorch, TchTensor};
use burn_tensor::ops::ActivationOps;

impl<E: TchElement> ActivationOps<Self> for LibTorch<E> {
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

    fn sigmoid<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.unary_ops(|mut tensor| tensor.sigmoid_(), |tensor| tensor.sigmoid())
    }

    fn log_sigmoid<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, D> {
        // NOTE: we don't override log_sigmoid_backward because Torch has a special backward
        // formula that uses a buffer with computed values from the forward pass

        // no in-place log_sigmoid_
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.log_sigmoid();

        TchTensor::from_existing(tensor, storage)
    }
}
