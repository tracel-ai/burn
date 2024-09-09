use cubecl::cube;
use cubecl::frontend::CubeType;
use cubecl::prelude::{Numeric, Tensor};

/// Specifies the reduce dim algorithm in use
#[cube]
pub trait ReduceDimNaive<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: Copy + CubeType;

    /// Initialization for naive algorithm
    fn initialize_naive() -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(accumulator: &mut Self::Accumulator, current_value: EI, i: u32);

    /// Assignation for naive algorithm
    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<EO>,
        accumulator: Self::Accumulator,
        shape_reduce_dim: u32,
    );
}
