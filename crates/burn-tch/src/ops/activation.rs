use crate::{LibTorch, TchTensor};
use burn_backend::ops::ActivationOps;

impl ActivationOps<Self> for LibTorch {
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

    fn softmax(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.softmax(dim as i64, None);
        TchTensor::from_existing(tensor, storage)
    }

    fn log_softmax(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.log_softmax(dim as i64, None);
        TchTensor::from_existing(tensor, storage)
    }

    fn softmin(tensor: TchTensor, dim: usize) -> TchTensor {
        let storage = tensor.storage.clone();
        let tensor = tensor.tensor.neg().softmax(dim as i64, None);
        TchTensor::from_existing(tensor, storage)
    }

    fn prelu(tensor: TchTensor, alpha: TchTensor) -> TchTensor {
        let storage = tensor.storage.clone();
        // `activation::prelu` broadcasts `alpha` to the rank of the input (`[1, C, 1, ...]`)
        // for the default composite implementation. `at::native::prelu` only accepts a
        // scalar or a 1-D weight, and applies a size-`C` weight along dim 1, which is the
        // same semantics that layout encodes.
        let tensor = tensor.tensor.prelu(&alpha.tensor.reshape([-1]));
        TchTensor::from_existing(tensor, storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{TensorData, Tolerance, ops::FloatTensorOps, read_sync};

    type B = crate::LibTorch;

    /// `activation::prelu` hands backends an `alpha` already broadcast to the rank of the
    /// input, so `prelu` must accept more than the scalar or 1-D weight `tch` requires.
    #[test]
    fn prelu_accepts_a_broadcast_alpha() {
        let device = Default::default();
        // [N=1, C=2, H=1, W=2] with a per-channel alpha shaped the way `burn-tensor` ships it.
        let x = B::float_from_data(
            TensorData::new(vec![-2.0f32, 3.0, -4.0, 5.0], [1, 2, 1, 2]),
            &device,
        );
        let alpha = B::float_from_data(TensorData::new(vec![0.1f32, 0.5], [1, 2, 1, 1]), &device);

        let out = read_sync(B::float_into_data(<B as ActivationOps<B>>::prelu(x, alpha))).unwrap();

        let expected = TensorData::new(vec![-0.2f32, 3.0, -2.0, 5.0], [1, 2, 1, 2]);
        out.assert_approx_eq::<f32>(&expected, Tolerance::default());
    }
}
