use cubecl::cube;
use cubecl::prelude::{ABSOLUTE_POS, Cast, Numeric, Tensor, UInt};
use crate::{kernel::reduce::Argmin, JitElement};

use super::base::ReduceDimNaive;

#[cube]
impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for Argmin {

    type Accumulator = (EI, UInt);

    fn initialize_naive() -> (EI, UInt) {
        // TODO: how to get the max value of a Primitive?
        (EI::from(u32::MAX), UInt::from(0))
    }

    fn inner_loop_naive(
        accumulator: &mut (EI, UInt),
        current_value: EI,
        i: UInt,
    ) {
        let (min, index) = accumulator;
        if current_value < *min {
            *min = current_value;
            *index = i;
        }
    }

    fn assign_naive(
        output: &mut Tensor<EO>,
        accumulator: (EI, UInt),
        _shape_reduce_dim: UInt,
    ) {
        let (_, index) = accumulator;
        output[ABSOLUTE_POS] = EO::cast_from(index);
    }
}
