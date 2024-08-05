#[cfg(feature = "autotune")]
use crate::kernel::accumulate::accumulate_dim_autotune;
use crate::{element::JitElement, tensor::JitTensor, JitRuntime};

use super::{
    naive::{base::AccumulateDimNaive, shader::accumulate_dim_naive},
    shared::{base::AccumulateDimShared, shader::accumulate_dim_shared},
};

#[allow(dead_code)]
pub(crate) trait AccumulateDimAlgorithm<E: JitElement>:
AccumulateDimNaive<E> + AccumulateDimShared<E>
{
}

/// Creates an empty output tensor with accumulate output shape
pub fn init_accumulate_output<R: JitRuntime, EI: JitElement, EO: JitElement, const D: usize>(
    input: &JitTensor<R, EI, D>,
) -> JitTensor<R, EO, D> {
    let mut shape_out = input.shape.clone();

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
pub enum AccumulateStrategy {
    Naive,
    SharedMemory,
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for AccumulateStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return AccumulateStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        AccumulateStrategy::Naive
    }
}

#[cfg(feature = "autotune")]
#[cfg(not(feature = "autotune"))]
impl Default for AccumulateStrategy {
    fn default() -> Self {
        AccumulateStrategy::Naive
    }
}

macro_rules! accumulate_operation {
    ($name:ident, $ops:ident) => {
        pub(crate) struct $ops;
        impl<E: JitElement> AccumulateDimAlgorithm<E> for $ops {}

        /// Executes the accumulate operation with the given strategy.
        pub fn $name<R: JitRuntime, EI: JitElement, EO: JitElement, const D: usize>(
            tensor: JitTensor<R, EI, D>,
            dim: usize,
            strategy: AccumulateStrategy,
        ) -> JitTensor<R, EO, D> {
            match strategy {
                AccumulateStrategy::Naive => {
                    let output = init_accumulate_output(&tensor, dim);
                    accumulate_dim_naive::<$ops, R, EI, EO, D>(tensor, output, dim)
                }
                AccumulateStrategy::SharedMemory => {
                    let output = init_accumulate_output(&tensor, dim);
                    accumulate_dim_shared::<$ops, R, EI, EO, D>(tensor, output, dim)
                }
                #[cfg(feature = "autotune")]
                AccumulateStrategy::Autotune => accumulate_dim_autotune::<$ops, R, EI, EO, D>(tensor, dim),
            }
        }
    };
}

// Autotunable reduce operation variants
accumulate_operation!(cumsum, CumSum);
