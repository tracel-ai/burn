#[cfg(feature = "autotune")]
use crate::kernel::reduce::reduce_dim_autotune;
use crate::{
    codegen::dialect::gpu::{Item, Scope, Variable},
    element::JitElement,
    tensor::JitTensor,
    Runtime,
};

use super::{reduce_dim_naive, reduce_dim_shared, ArgMax, ArgMin, MeanDim, ProdDim, SumDim};

/// Specifies the reduce dim algorithm in use
pub trait ReduceDimAlgorithm<E: JitElement>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: Copy;

    /// Initialization for naive algorithm
    fn initialize_naive(
        scope: &mut Scope,
        input_item: Item,
        output_item: Item,
    ) -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(
        scope: &mut Scope,
        accumulator: Self::Accumulator,
        current_value: Variable,
        i: Variable,
    );

    /// Assignation for naive algorithm
    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        accumulator: Self::Accumulator,
        shape_reduce_dim: Variable,
    );

    /// Initialization for shared algorithm
    fn initialize_shared(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator;

    /// How to write to shared memory
    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    );

    /// How to read from input in shared algorithm
    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
        i: Variable,
    ) -> Self::Accumulator;

    /// How to read from shared memory
    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator;

    /// How to assign from shared memory
    fn assign_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        shape_reduce_dim: Variable,
    );
}

/// Creates an empty output tensor with reduce output shape
pub fn init_reduce_output<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    input: &JitTensor<R, EI, D>,
    reduce_dim: usize,
) -> JitTensor<R, EO, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[reduce_dim] = 1;

    // Create output handle
    let num_elems_output = shape_out.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<EO>());
    JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        handle,
    )
}

#[derive(Copy, Clone, Debug)]
#[allow(missing_docs)]
#[derive(Default)]
pub enum ReduceStrategy {
    Naive,
    SharedMemory,
    #[cfg(feature = "autotune")]
    #[default]
    Autotune,
}

#[cfg(feature = "autotune")]
#[cfg(not(feature = "autotune"))]
impl Default for ReduceStrategy {
    fn default() -> Self {
        ReduceStrategy::Naive
    }
}

macro_rules! reduce_operation {
    ($name:ident, $ops:ty) => {
        /// Executes the reduce operation with the given strategy.
        pub fn $name<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
            tensor: JitTensor<R, EI, D>,
            dim: usize,
            strategy: ReduceStrategy,
        ) -> JitTensor<R, EO, D> {
            match strategy {
                ReduceStrategy::Naive => {
                    let output = init_reduce_output(&tensor, dim);
                    reduce_dim_naive::<$ops, R, EI, EO, D>(tensor, output, dim)
                }
                ReduceStrategy::SharedMemory => {
                    let output = init_reduce_output(&tensor, dim);
                    reduce_dim_shared::<$ops, R, EI, EO, D>(tensor, output, dim)
                }
                #[cfg(feature = "autotune")]
                ReduceStrategy::Autotune => reduce_dim_autotune::<$ops, R, EI, EO, D>(tensor, dim),
            }
        }
    };
}

// Autotunable reduce operation variants
reduce_operation!(sum_dim, SumDim);
reduce_operation!(mean_dim, MeanDim);
reduce_operation!(prod_dim, ProdDim);
reduce_operation!(argmin, ArgMin);
reduce_operation!(argmax, ArgMax);
