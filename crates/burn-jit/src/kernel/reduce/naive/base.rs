use cubecl::prelude::{Numeric, Tensor, UInt};
use crate::JitElement;

/// Specifies the reduce dim algorithm in use
pub trait ReduceDimNaive<EI: Numeric, EO: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: Copy;

    /// Initialization for naive algorithm
    fn initialize_naive() -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(
        accumulator: &mut Self::Accumulator,
        current_value: EI,
        i: UInt,
    );

    /// Assignation for naive algorithm
    fn assign_naive(
        output: &mut Tensor<EO>,
        accumulator: Self::Accumulator,
        shape_reduce_dim: UInt,
    );
}
