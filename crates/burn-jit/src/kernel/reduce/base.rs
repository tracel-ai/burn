use cubecl::prelude::Numeric;

#[cfg(feature = "autotune")]
use crate::kernel::reduce::reduce_dim_autotune;
use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};

use super::{
    naive::{base::ReduceDimNaive, kernel::reduce_dim_naive},
    shared::{base::ReduceDimShared, kernel::reduce_dim_shared},
    subcube::{base::ReduceDimSubcube, kernel::reduce_dim_subcube},
};

#[allow(dead_code)]
pub(crate) trait ReduceDimAlgorithm<EI: JitElement + Numeric, EO: JitElement>:
    core::fmt::Debug + ReduceDimNaive<EI> + ReduceDimShared<EI, EO> + ReduceDimSubcube<EI, EO>
{
}

/// Creates an empty output tensor with reduce output shape
pub fn init_reduce_output<R: JitRuntime, EI: JitElement, EO: JitElement>(
    input: &JitTensor<R>,
    reduce_dim: usize,
) -> JitTensor<R> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[reduce_dim] = 1;

    empty_device::<R, EO>(input.client.clone(), input.device.clone(), shape_out)
}

#[derive(Copy, Clone, Debug)]
#[allow(missing_docs)]
pub enum ReduceStrategy {
    /// Naive
    Naive,
    /// Use shared memory as an accumulator
    SharedMemory,
    /// Use subcube functions
    Subcube,
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

macro_rules! reduce_operation {
    ($name:ident, $ops:ident) => {
        #[derive(Debug)]
        pub(crate) struct $ops;

        impl<EI: JitElement, EO: JitElement> ReduceDimAlgorithm<EI, EO> for $ops {}

        /// Executes the reduce operation with the given strategy.
        pub fn $name<R: JitRuntime, EI: JitElement, EO: JitElement>(
            tensor: JitTensor<R>,
            dim: usize,
            strategy: ReduceStrategy,
        ) -> JitTensor<R> {
            match strategy {
                ReduceStrategy::Naive => reduce_dim_naive::<$ops, R, EI, EO>(tensor, dim),
                ReduceStrategy::SharedMemory => reduce_dim_shared::<$ops, R, EI, EO>(tensor, dim),
                ReduceStrategy::Subcube => reduce_dim_subcube::<$ops, R, EI, EO>(tensor, dim),
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
