use crate::{LibTorch, TchFloatElement, TchIntElement, TchTensor, element::TchElement};
use burn_backend::ops::ActivationOps;

impl<E: TchElement, F: TchFloatElement> ActivationOps<Self> for LibTorch<E, F> {
    fn relu(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.relu_(), |tensor| tensor.relu())
    }

    fn gelu(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.gelu_("none"),
            |tensor| tensor.gelu("none"),
        )
    }

    fn gelu_backward(tensor: TchTensor, grad: TchTensor) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.gelu_backward(&grad.tensor, "none");

        TchTensor::from_existing(tensor, storage)
    }

    fn sigmoid(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(|mut tensor| tensor.sigmoid_(), |tensor| tensor.sigmoid())
    }

    fn log_sigmoid(tensor: TchTensor) -> TchTensor {
        // NOTE: we don't override log_sigmoid_backward because Torch has a special backward
        // formula that uses a buffer with computed values from the forward pass

        // no in-place log_sigmoid_
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.log_sigmoid();

        TchTensor::from_existing(tensor, storage)
    }
}
