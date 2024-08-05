use crate::{kernel::reduce::MeanDim};
use cubecl::prelude::{ABSOLUTE_POS, Cast, Numeric, Tensor, UInt};

use super::base::ReduceDimNaive;

impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for MeanDim {
    type Accumulator = EI;

    fn initialize_naive() -> EI {
        EI::from(0)
    }

    fn inner_loop_naive(
        accumulator: &mut EI,
        current_value: EI,
        _i: UInt,
    ) {
        *accumulator += current_value;
    }

    fn assign_naive(
        output: &mut Tensor<EO>,
        accumulator: Self::Accumulator,
        shape_reduce_dim: UInt,
    ) {
        output[ABSOLUTE_POS] = EO::cast_from(accumulator) / EO::cast_from(shape_reduce_dim);
    }
}