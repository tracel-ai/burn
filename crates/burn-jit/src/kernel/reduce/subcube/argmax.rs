use cubecl::{cube, prelude::*};

use crate::{kernel::reduce::Argmax, JitElement};

use super::base::ReduceDimSubcube;

#[cube]
impl<EIn: JitElement, EOut: JitElement> ReduceDimSubcube<EIn, EOut> for Argmax {
    /// The reduction accumulator
    type Accumulator = (SharedMemory<EIn>, SharedMemory<u32>);
    type Value = (EIn, u32);

    fn init_shared(#[comptime] size: u32) -> Self::Accumulator {
        let value_shared = SharedMemory::new(size);
        let index_shared = SharedMemory::new(size);
        (value_shared, index_shared)
    }

    fn init_value() -> Self::Value {
        (comptime![EIn::minimum_value()], 0u32)
    }

    fn read_value(input: &Tensor<EIn>, pos: u32, i: u32) -> Self::Value {
        (input[pos], i)
    }

    fn read_from_shared(acc: &Self::Accumulator, pos: u32) -> Self::Value {
        let (values, indices) = acc;
        (values[pos], indices[pos])
    }

    fn update_value(current: &mut Self::Value, new: Self::Value) {
        let (current_val, current_idx) = current;
        let (new_val, new_idx) = new;
        *current_val = Max::max(*current_val, new_val);
        *current_idx = select(*current_val == new_val, new_idx, *current_idx);
    }

    fn reduce_subcube(acc: &mut Self::Accumulator, write_position: u32, value: Self::Value) {
        let (val, index) = value;
        let (val_smem, index_smem) = acc;
        let max = plane_max(val);

        if max == val {
            val_smem[write_position] = val;
            index_smem[write_position] = index;
        }
    }

    fn store(acc: &Self::Accumulator, out: &mut Tensor<EOut>, pos: u32, _layout: u32) {
        let (_, indices) = acc;
        out[pos] = EOut::cast_from(indices[0]);
    }
}
