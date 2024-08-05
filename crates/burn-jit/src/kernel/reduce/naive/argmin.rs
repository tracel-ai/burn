use cubecl::prelude::{ABSOLUTE_POS, Cast, Numeric, Tensor, UInt};
use crate::{kernel::reduce::Argmin, JitElement};

use super::base::ReduceDimNaive;

impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for Argmin {

    type Accumulator = (EI, UInt);

    fn initialize_naive() -> (EI, UInt) {
        // TODO: how to get the max value of a Primitive?
        (EI::from(u32::MAX), UInt::from(0))
    }

    fn inner_loop_naive(
        (min, index): &mut Self::Accumulator,
        current_value: EI,
        i: UInt,
    ) {
        if current_value < *min {
            *min = current_value;
            *index = i;
        }
    }

    fn assign_naive(
        output: &mut Tensor<EO>,
        (_, index): Self::Accumulator,
        _shape_reduce_dim: UInt,
    ) {
        output[ABSOLUTE_POS] = EO::cast_from(index);
    }
}
