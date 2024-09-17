use crate::kernel::reduce::MeanDim;
use cubecl::cube;
use cubecl::prelude::{Cast, Numeric, Tensor, ABSOLUTE_POS};

use super::base::ReduceDimNaive;

#[cube]
impl<EI: Numeric> ReduceDimNaive<EI> for MeanDim {
    type Accumulator = EI;

    fn initialize_naive() -> EI {
        EI::from_int(0)
    }

    fn inner_loop_naive(accumulator: &mut EI, current_value: EI, _i: u32) {
        *accumulator += current_value;
    }

    fn assign_naive<EO: Numeric>(output: &mut Tensor<EO>, accumulator: EI, shape_reduce_dim: u32) {
        output[ABSOLUTE_POS] = EO::cast_from(accumulator) / EO::cast_from(shape_reduce_dim);
    }
}
