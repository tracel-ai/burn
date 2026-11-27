use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::{LinearView, LinearViewMut};
use cubecl::{self as cubecl, cube};

/// Vectors each unit of [`touch_vectors`] reads for the one it writes.
pub const TOUCHED_PER_WRITTEN: u32 = 64;

/// Every unit sums the vectors of `input` at its own position and every `output.shape()` after
/// it, [`TOUCHED_PER_WRITTEN`] of them, and writes the sum: a read of the whole input through the
/// device's caches.
#[cube(launch, address_type = "dynamic")]
pub fn touch_vectors<E: Numeric, N: Size>(
    input: LinearView<'_, Vector<E, N>>,
    mut output: LinearViewMut<'_, Vector<E, N>>,
    #[define(E)] _dtype: ElemType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    // The stride between one unit's reads is the number of units that write,
    // which is the output's own length — never a topology builtin. The
    // runtime shapes the cube, and on CUDA a 256-unit cube is `(32, 8)`, so
    // `CUBE_COUNT_X * CUBE_DIM_X` is the planes and not the units: a stride
    // built from it covers an eighth of the scratch, most of it several
    // times over, and the cache this read exists to empty stays warm.
    let units = output.shape();
    let mut sum = Vector::<E, N>::cast_from(0u32);
    #[unroll]
    for i in 0..TOUCHED_PER_WRITTEN as usize {
        let vector = ABSOLUTE_POS + i * units;
        if input.is_in_bounds(vector) {
            sum += input.read(vector);
        }
    }
    output.write(ABSOLUTE_POS, sum);
}
