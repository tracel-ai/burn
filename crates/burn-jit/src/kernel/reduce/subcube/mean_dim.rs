use cubecl::{cube, prelude::*};

use crate::{kernel::reduce::MeanDim, JitElement};

use super::base::ReduceDimSubcube;

#[cube]
impl<EIn: JitElement, EOut: JitElement> ReduceDimSubcube<EIn, EOut> for MeanDim {
    /// The reduction accumulator
    type Accumulator = SharedMemory<EIn>;
    type Value = EIn;

    fn init_shared(#[comptime] size: u32) -> Self::Accumulator {
        SharedMemory::new(size)
    }

    fn init_value() -> Self::Value {
        comptime![EIn::default()].runtime()
    }

    fn read_value(input: &Tensor<EIn>, pos: u32, _i: u32) -> Self::Value {
        input[pos]
    }

    fn read_from_shared(acc: &Self::Accumulator, pos: u32) -> Self::Value {
        acc[pos]
    }

    fn update_value(current: &mut Self::Value, new: Self::Value) {
        *current += new;
    }

    fn reduce_subcube(acc: &mut Self::Accumulator, write_position: u32, value: Self::Value) {
        let sum = plane_sum(value);

        if UNIT_POS % PLANE_DIM == 0 {
            acc[write_position] = sum;
        }
    }

    fn store(acc: &Self::Accumulator, out: &mut Tensor<EOut>, pos: u32, dim_length: u32) {
        let denom = EIn::cast_from(dim_length);
        out[pos] = EOut::cast_from(acc[0] / denom);
    }
}
