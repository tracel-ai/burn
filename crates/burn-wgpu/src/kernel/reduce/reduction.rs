use crate::{element::JitElement, tensor::JitTensor, Runtime};
use burn_tensor::Shape;

#[cfg(feature = "autotune")]
use crate::kernel::reduce::reduce_dim_autotune;

#[cfg(not(feature = "autotune"))]
use super::init_reduce_output;

use super::{reduce_dim_naive, reduce_dim_shared, ArgMax, ArgMin, MeanDim, SumDim};

/// Sum all elements in the input buffer.
pub fn sum<R: Runtime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
) -> JitTensor<R, E, 1> {
    let shape = Shape::new([input.shape.num_elements()]);
    let input: JitTensor<R, E, 1> = JitTensor::new(input.client, input.device, shape, input.handle);
    sum_dim(input, 0)
}

macro_rules! reduce_operation {
    ($name:ident, $ops:ty) => {
        /// Executes $name operation, autotunable
        pub fn $name<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
            tensor: JitTensor<R, EI, D>,
            dim: usize,
        ) -> JitTensor<R, EO, D> {
            #[cfg(feature = "autotune")]
            {
                reduce_dim_autotune::<$ops, R, EI, EO, D>(tensor, dim)
            }

            #[cfg(not(feature = "autotune"))]
            {
                let output = init_reduce_output(&tensor, dim);
                reduce_dim_naive::<$ops, R, ER, EO, D>(tensor, output, dim)
            }
        }
    };
}

reduce_operation!(sum_dim, SumDim);
reduce_operation!(mean_dim, MeanDim);
reduce_operation!(argmin, ArgMin);
reduce_operation!(argmax, ArgMax);

/// Exposed for benchmarks. Prefer the autotunable version for other uses
pub fn sum_dim_naive<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    reduce_dim_naive::<SumDim, R, EI, EO, D>(tensor, output, dim)
}

/// Exposed for benchmarks. Prefer the autotunable version for other uses
pub fn sum_dim_shared<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    tensor: JitTensor<R, EI, D>,
    output: JitTensor<R, EO, D>,
    dim: usize,
) -> JitTensor<R, EO, D> {
    reduce_dim_shared::<SumDim, R, EI, EO, D>(tensor, output, dim)
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
