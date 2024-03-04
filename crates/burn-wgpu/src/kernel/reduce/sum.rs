use crate::{element::JitElement, tensor::JitTensor, Runtime};
use burn_tensor::Shape;

use super::sum_dim;

/// Sum all elements in the input buffer.
pub fn sum<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
) -> JitTensor<R, E, 1> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E, 1> = JitTensor::new(input.client, input.device, shape, input.handle);
    sum_dim(input, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn reduction_sum_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let val = Tensor::<TestBackend, 1>::from_primitive(sum(tensor.into_primitive()));
        let val_ref = tensor_ref.sum();

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }
}
