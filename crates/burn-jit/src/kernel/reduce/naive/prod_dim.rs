use crate::{kernel::reduce::ProdDim, JitElement};
use cubecl::prelude::{ABSOLUTE_POS, Cast, Numeric, Tensor, UInt};

use super::base::ReduceDimNaive;

impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for ProdDim {
    type Accumulator = EI;

    fn initialize_naive() -> EI {
        EI::from(1)
    }

    fn inner_loop_naive(
        accumulator: &mut EI,
        current_value: EI,
        _i: UInt,
    ) {
        *accumulator *= current_value;
    }

    fn assign_naive(
        output: &mut Tensor<EO>,
        accumulator: Self::Accumulator,
        _shape_reduce_dim: UInt,
    ) {
        output[ABSOLUTE_POS] = EO::cast_from(accumulator);
    }
}
