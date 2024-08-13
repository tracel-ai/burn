use crate::kernel::reduce::Argmin;
use cubecl::cube;
use cubecl::prelude::{Cast, Float, Numeric, Tensor, UInt, ABSOLUTE_POS, F32};

use super::base::ReduceDimNaive;

#[allow(clippy::extra_unused_type_parameters)]
#[cube]
impl<EI: Numeric> ReduceDimNaive<EI> for Argmin {
    type Accumulator = (F32, UInt);

    fn initialize_naive() -> (F32, UInt) {
        // TODO: switch to using f32::INFINITY when it's supported: https://github.com/tracel-ai/cubecl/issues/68
        (F32::new(100000000.0), UInt::new(0))
    }

    fn inner_loop_naive(accumulator: &mut (F32, UInt), current_value: EI, i: UInt) {
        let (min, index) = accumulator;
        let val = F32::cast_from(current_value);
        if val < *min {
            *min = val;
            *index = i;
        }
    }

    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<EO>,
        accumulator: (F32, UInt),
        _shape_reduce_dim: UInt,
    ) {
        let (_, index) = accumulator;
        output[ABSOLUTE_POS] = EO::cast_from(index);
    }
}
