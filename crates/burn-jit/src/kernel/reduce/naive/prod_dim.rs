use crate::kernel::reduce::ProdDim;
use cubecl::cube;
use cubecl::prelude::{Cast, Numeric, Tensor, ABSOLUTE_POS};

use super::base::ReduceDimNaive;

#[cube]
impl<EI: Numeric> ReduceDimNaive<EI> for ProdDim {
    type Accumulator = EI;

    fn initialize_naive() -> EI {
        EI::from_int(1)
    }

    fn inner_loop_naive(accumulator: &mut EI, current_value: EI, _i: u32) {
        *accumulator *= current_value;
    }

    fn assign_naive<EO: Numeric>(output: &mut Tensor<EO>, accumulator: EI, _shape_reduce_dim: u32) {
        output[ABSOLUTE_POS] = EO::cast_from(accumulator);
    }
}
