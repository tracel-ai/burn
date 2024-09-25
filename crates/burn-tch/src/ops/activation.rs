use crate::{element::TchElement, LibTorch, QuantElement, TchTensor};
use burn_tensor::ops::ActivationOps;

impl<E: TchElement, Q: QuantElement> ActivationOps<Self> for LibTorch<E, Q> {
    fn relu(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.relu_(), |tensor| tensor.relu())
    }

    fn gelu(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(
            |mut tensor| tensor.gelu_("none"),
            |tensor| tensor.gelu("none"),
        )
    }

    fn gelu_backward(tensor: TchTensor<E>, grad: TchTensor<E>) -> TchTensor<E> {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.gelu_backward(&grad.tensor, "none");

        TchTensor::from_existing(tensor, storage)
    }

    fn sigmoid(tensor: TchTensor<E>) -> TchTensor<E> {
        tensor.unary_ops(|mut tensor| tensor.sigmoid_(), |tensor| tensor.sigmoid())
    }

    fn log_sigmoid(tensor: TchTensor<E>) -> TchTensor<E> {
        // NOTE: we don't override log_sigmoid_backward because Torch has a special backward
        // formula that uses a buffer with computed values from the forward pass

        // no in-place log_sigmoid_
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.log_sigmoid();

        TchTensor::from_existing(tensor, storage)
    }
}
