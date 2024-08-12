use cubecl::cube;
use cubecl::prelude::{ABSOLUTE_POS, Cast, Numeric, Tensor, UInt, F32, Float};
use crate::{kernel::reduce::Argmin};

use super::base::ReduceDimNaive;

#[cube]
impl<EI: Numeric, EO: Numeric> ReduceDimNaive<EI, EO> for Argmin {

    type Accumulator = (F32, UInt);

    fn initialize_naive() -> (F32, UInt) {
        // (F32::new(f32::INFINITY), UInt::new(0))
        (F32::new(1000000.0), UInt::new(0))
    }

    fn inner_loop_naive(
        accumulator: &mut (F32, UInt),
        current_value: EI,
        i: UInt,
    ) {
        let (min, index) = accumulator;
        let val = F32::cast_from(current_value);
        if val < *min {
            *min = val;
            *index = i;
        }
    }

    fn assign_naive(
        output: &mut Tensor<EO>,
        accumulator: (F32, UInt),
        _shape_reduce_dim: UInt,
    ) {
        let (_, index) = accumulator;
        output[ABSOLUTE_POS] = EO::cast_from(index);
    }
}
