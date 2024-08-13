use cubecl::frontend::CubeType;
use cubecl::prelude::{Numeric, Tensor, UInt};

/// Specifies the reduce dim algorithm in use
pub trait ReduceDimNaive<EI: Numeric>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: Copy + CubeType;

    /// Initialization for naive algorithm
    fn initialize_naive() -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(accumulator: &mut Self::Accumulator, current_value: EI, i: UInt);

    /// Assignation for naive algorithm
    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<EO>,
        accumulator: Self::Accumulator,
        shape_reduce_dim: UInt,
    );

    fn __expand_initialize_naive(
        context: &mut cubecl::frontend::CubeContext,
    ) -> <Self::Accumulator as cubecl::frontend::CubeType>::ExpandType;

    fn __expand_inner_loop_naive(
        context: &mut cubecl::frontend::CubeContext,
        accumulator: <Self::Accumulator as cubecl::frontend::CubeType>::ExpandType,
        current_value: <EI as cubecl::frontend::CubeType>::ExpandType,
        _i: <UInt as cubecl::frontend::CubeType>::ExpandType,
    );

    fn __expand_assign_naive<EO: Numeric>(
        context: &mut cubecl::frontend::CubeContext,
        output: <Tensor<EO> as cubecl::frontend::CubeType>::ExpandType,
        accumulator: <Self::Accumulator as cubecl::frontend::CubeType>::ExpandType,
        _shape_reduce_dim: <UInt as cubecl::frontend::CubeType>::ExpandType,
    );
}
