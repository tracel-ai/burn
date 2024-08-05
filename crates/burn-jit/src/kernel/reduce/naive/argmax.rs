use cubecl::frontend::{ABSOLUTE_POS, Tensor, UInt};
use cubecl::prelude::{Cast, Numeric};
use crate::kernel::reduce::Argmax;
use super::base::ReduceDimNaive;


impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for Argmax {

    type Accumulator = (EI, UInt);

    fn initialize_naive() -> (EI, UInt) {
        // TODO: how to get the min value of a Primitive?
        (EI::from(u32::MIN), UInt::from(0))
    }

    fn inner_loop_naive(
        (max, index): &mut Self::Accumulator,
        current_value: EI,
        i: UInt,
    ) {
        if current_value > *max {
            *max = current_value;
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
