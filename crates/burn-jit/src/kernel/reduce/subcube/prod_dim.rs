use cubecl::{cube, prelude::*};

use crate::{kernel::reduce::ProdDim, JitElement};

use super::base::ReduceDimSubcube;

#[cube]
impl<EIn: JitElement, EOut: JitElement> ReduceDimSubcube<EIn, EOut> for ProdDim {
    /// The reduction accumulator
    type Accumulator = SharedMemory<EIn>;
    type Value = EIn;

    fn init_shared(#[comptime] size: u32) -> Self::Accumulator {
        SharedMemory::new(size)
    }

    fn init_value() -> Self::Value {
        comptime![EIn::from_int(1)].runtime()
    }

    fn read_value(input: &Tensor<EIn>, pos: u32, _i: u32) -> Self::Value {
        input[pos]
    }

    fn read_from_shared(acc: &Self::Accumulator, pos: u32) -> Self::Value {
        acc[pos]
    }

    fn update_value(current: &mut Self::Value, new: Self::Value) {
        *current *= new;
    }

    fn reduce_subcube(acc: &mut Self::Accumulator, write_position: u32, value: Self::Value) {
        let prod = plane_prod(value);

        if UNIT_POS % PLANE_DIM == 0 {
            acc[write_position] = prod;
        }
    }

    fn store(acc: &Self::Accumulator, out: &mut Tensor<EOut>, pos: u32, _layout: u32) {
        out[pos] = EOut::cast_from(acc[0]);
    }
}
