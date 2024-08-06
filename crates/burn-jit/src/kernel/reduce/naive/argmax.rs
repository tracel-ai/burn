use cubecl::cube;
use cubecl::frontend::{ABSOLUTE_POS, Tensor, UInt};
use cubecl::prelude::{Cast, Numeric};
use crate::kernel::reduce::Argmax;
use super::base::ReduceDimNaive;


#[cube]
impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for Argmax {

    type Accumulator = (EI, UInt);

    fn initialize_naive() -> (EI, UInt) {
        // TODO: how to get the min value of a Primitive?
        (EI::from(u32::MIN), UInt::from(0))
    }

    fn inner_loop_naive(
        accumulator: &mut (EI, UInt),
        current_value: EI,
        i: UInt,
    ) {
        let (max, index) = accumulator;
        if current_value > *max {
            *max = current_value;
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
