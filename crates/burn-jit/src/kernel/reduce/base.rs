#[cfg(feature = "autotune")]
use crate::kernel::reduce::reduce_dim_autotune;
use crate::{element::JitElement, tensor::JitTensor, JitRuntime};

use super::{
    naive::{base::ReduceDimNaive, shader::reduce_dim_naive},
    shared::{base::ReduceDimShared, shader::reduce_dim_shared},
};

#[allow(dead_code)]
pub(crate) trait ReduceDimAlgorithm<EI: JitElement>:
    ReduceDimNaive<EI> + ReduceDimShared<EI>
{
}

/// Creates an empty output tensor with reduce output shape
pub fn init_reduce_output<R: JitRuntime, EI: JitElement, EO: JitElement>(
    input: &JitTensor<R, EI>,
    reduce_dim: usize,
) -> JitTensor<R, EO> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[reduce_dim] = 1;

    // Create output handle
    let num_elems_output = shape_out.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<EO>());
    JitTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        handle,
    )
}

#[derive(Copy, Clone, Debug)]
#[allow(missing_docs)]
pub enum ReduceStrategy {
    Naive,
    SharedMemory,
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for ReduceStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return ReduceStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        ReduceStrategy::Naive
    }
}

#[cfg(feature = "autotune")]
#[cfg(not(feature = "autotune"))]
impl Default for ReduceStrategy {
    fn default() -> Self {
        ReduceStrategy::Naive
    }
}

macro_rules! reduce_operation {
    ($name:ident, $ops:ident) => {
        pub(crate) struct $ops;
        impl<EI: JitElement> ReduceDimAlgorithm<EI> for $ops {}

        /// Executes the reduce operation with the given strategy.
        pub fn $name<R: JitRuntime, EI: JitElement, EO: JitElement>(
            tensor: JitTensor<R, EI>,
            dim: usize,
            strategy: ReduceStrategy,
        ) -> JitTensor<R, EO> {
            match strategy {
                ReduceStrategy::Naive => {
                    let output = init_reduce_output(&tensor, dim);
                    reduce_dim_naive::<$ops, R, EI, EO>(tensor, output, dim)
                }
                ReduceStrategy::SharedMemory => {
                    let output = init_reduce_output(&tensor, dim);
                    reduce_dim_shared::<$ops, R, EI, EO>(tensor, output, dim)
                }
                #[cfg(feature = "autotune")]
                ReduceStrategy::Autotune => reduce_dim_autotune::<$ops, R, EI, EO>(tensor, dim),
            }
        }
    };
}

// Autotunable reduce operation variants
reduce_operation!(sum_dim, SumDim);
reduce_operation!(mean_dim, MeanDim);
reduce_operation!(prod_dim, ProdDim);
reduce_operation!(argmin, Argmin);
reduce_operation!(argmax, Argmax);
