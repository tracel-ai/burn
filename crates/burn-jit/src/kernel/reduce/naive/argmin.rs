use crate::kernel::reduce::Argmin;
use cubecl::cube;
use cubecl::prelude::{Cast, Numeric, Tensor, ABSOLUTE_POS};

use super::base::ReduceDimNaive;

#[allow(clippy::extra_unused_type_parameters)]
#[cube]
impl<EI: Numeric> ReduceDimNaive<EI> for Argmin {
    type Accumulator = (f32, u32);

    fn initialize_naive() -> (f32, u32) {
        // TODO: switch to using f32::INFINITY when it's supported: https://github.com/tracel-ai/cubecl/issues/68
        (100000000f32, 0u32)
    }

    fn inner_loop_naive(accumulator: &mut (f32, u32), current_value: EI, i: u32) {
        let (min, index) = accumulator;
        let val = f32::cast_from(current_value);
        if val < *min {
            *min = val;
            *index = i;
        }
    }

    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<EO>,
        accumulator: (f32, u32),
        _shape_reduce_dim: u32,
    ) {
        let (_, index) = accumulator;
        output[ABSOLUTE_POS] = EO::cast_from(index);
    }
}
