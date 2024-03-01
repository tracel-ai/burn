use crate::{element::JitElement, tensor::JitTensor, Runtime};
use burn_tensor::Shape;

// #[cfg(not(feature = "autotune"))]
#[cfg(feature = "autotune")]
use super::tune::{mean_dim_autotune, sum_dim_autotune};
use super::{
    argmax_naive, argmin_naive, init_reduce_output, shared_reduce_shader::mean_dim_shared,
};

/// Sum all elements in the input buffer.
pub fn sum<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
) -> JitTensor<R, E, 1> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E, 1> = JitTensor::new(input.client, input.device, shape, input.handle);
    sum_dim(input, 0)
}

/// Sum all elements on one dimension. Autotunable
pub fn sum_dim<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    #[cfg(feature = "autotune")]
    {
        sum_dim_autotune(tensor, dim)
    }

    #[cfg(not(feature = "autotune"))]
    {
        let output = init_reduce_output(&tensor, dim);
        sum_dim_naive(tensor, output, dim)
    }
}

/// Mean of all elements on one dimension. Autotunable
pub fn mean_dim<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, E, D> {
    // #[cfg(feature = "autotune")]
    // {
    //     mean_dim_autotune(tensor, dim)
    // }

    // #[cfg(not(feature = "autotune"))]
    {
        let output = init_reduce_output(&tensor, dim);
        // mean_dim_naive(tensor, output, dim)
        mean_dim_shared(tensor, output, dim)
    }
}

/// Execute the argmax kernel.
pub fn argmax<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, I, D> {
    argmax_naive::<R, E, I, D>(input, dim)
}

/// Execute the argmin kernel.
pub fn argmin<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    dim: usize,
) -> JitTensor<R, I, D> {
    argmin_naive::<R, E, I, D>(input, dim)
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
