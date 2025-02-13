use burn_tensor::Shape;
use cubecl::{prelude::*, CubeCount, CubeDim};

use crate::{
    ops::{max_vectorization, numeric::empty_device},
    tensor::JitTensor,
    JitElement, JitRuntime,
};

/// Efficiently transpose an NCHW tensor to NHWC for use in kernels that prefer NHWC for performance.
/// Faster than `into_contiguous`, but specialized only for this specific permutation.
///
/// # Arguments
///
/// * `input` - The input in NCHW format
///
/// # Output
///
/// The input in NHWC format
///
pub fn nchw_to_nhwc<R: JitRuntime, E: JitElement>(input: JitTensor<R>) -> JitTensor<R> {
    let tiles_per_block = 8;
    let warp_size = 32;
    let tile_dim = 16;

    let [batch_size, in_c, h, w] = input.shape.dims();
    let hw = h * w;

    let out_shape = Shape::new([batch_size, h, w, in_c]);
    let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), out_shape);

    let tiles_channel = in_c.div_ceil(tile_dim) as u32;
    let tiles_hw = hw.div_ceil(tile_dim) as u32;

    let block_tiles_y = Ord::min(tiles_channel.next_power_of_two(), tiles_per_block);
    let block_tiles_x = Ord::min(tiles_per_block / block_tiles_y, tiles_hw);

    let cube_count_y = tiles_channel.div_ceil(block_tiles_y);
    let cube_count_x = tiles_hw.div_ceil(block_tiles_x);
    let cube_count_z = batch_size as u32;

    let config = ComptimeConfig {
        tiles_x: block_tiles_x,
        warps_per_cube: tiles_per_block,
        tile_dim: tile_dim as u32,
        warp_size,
        num_banks: 32,
    };

    let cube_dim = CubeDim {
        x: block_tiles_x * warp_size,
        y: block_tiles_y,
        z: 1,
    };
    let cube_count = CubeCount::Static(cube_count_x, cube_count_y, cube_count_z);

    let in_vec = max_vectorization(&input);
    let out_vec = R::supported_line_sizes()
        .iter()
        .copied()
        .find(|vec| in_c % *vec as usize == 0)
        .unwrap_or(1);

    unsafe {
        nchw_to_nhwc_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<E>(in_vec),
            out.as_tensor_arg::<E>(out_vec),
            ScalarArg::new(hw as u32),
            config,
        )
    };

    out
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ComptimeConfig {
    tiles_x: u32,
    warps_per_cube: u32,
    tile_dim: u32,
    warp_size: u32,
    num_banks: i32,
}

#[cube(launch_unchecked)]
fn nchw_to_nhwc_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    out: &mut Tensor<Line<E>>,
    shape_hw: u32,
    #[comptime] config: ComptimeConfig,
) {
    let ComptimeConfig {
        tiles_x,
        warps_per_cube,
        tile_dim,
        warp_size,
        num_banks,
    } = config;

    let tile_elems = tile_dim * tile_dim;

    let unit_pos = UNIT_POS;
    let intra_warp_unit_idx = unit_pos % 32;
    let batch = CUBE_POS_Z;

    if batch >= input.shape(0) {
        terminate!();
    }

    let batch_offset = batch * input.stride(0);

    let warp_id = plane_broadcast(unit_pos / 32, 0);
    let warp_id_x = warp_id % tiles_x;

    let tile_x = CUBE_POS_X * tiles_x + warp_id_x;
    let tile_y = ABSOLUTE_POS_Y;

    let mut shared = SharedMemory::<E>::new(warps_per_cube * tile_elems);
    let shared_start = warp_id * tile_elems;

    let base_hw = tile_x * tile_dim;
    let base_c = tile_y * tile_dim;

    let elems_per_unit = tile_elems / warp_size;
    let unit_start = intra_warp_unit_idx * elems_per_unit;

    let mat_hw_start = unit_start % tile_dim;

    let mat_c = unit_start / tile_dim;
    let channel = base_c + mat_c;
    let offset = channel * input.stride(1) + batch_offset;

    let input_vec = input.line_size();
    let out_vec = out.line_size();
    let in_max = input.buffer_len() - 1;

    let channels = input.shape(1);

    let mat_offset_base = shared_start + mat_c * tile_dim;

    #[unroll]
    for hw in range_stepped(0, elems_per_unit, input_vec) {
        let mat_hw = mat_hw_start + hw;
        let hw = base_hw + mat_hw;
        let offset = Min::min((offset + hw) / input_vec, in_max);
        let value = input[offset];

        let mat_idx = mat_offset_base + mat_hw;

        #[unroll]
        for v in 0..input_vec {
            let shared_idx = swizzle(mat_idx + v, num_banks);
            shared[shared_idx] = value[v];
        }
    }

    sync_units();

    let mat_hw = mat_c;
    let hw = base_hw + mat_hw;

    if hw >= shape_hw {
        terminate!();
    }

    let mat_c_start = mat_hw_start;
    let offset = hw * out.stride(2) + batch_offset;
    let mat_base = shared_start + mat_hw;

    #[unroll]
    for ch in range_stepped(0, elems_per_unit, out_vec) {
        let mat_c = mat_c_start + ch;
        let ch = base_c + mat_c;

        let mat_idx = mat_base + mat_c * tile_dim;
        let mut value = Line::empty(out_vec);
        let offset = (offset + ch) / out_vec;

        #[unroll]
        for v in 0..out_vec {
            let shared_idx = swizzle(mat_idx + v * tile_dim, num_banks);
            value[v] = shared[shared_idx];
        }

        if ch < channels {
            out[offset] = value;
        }
    }
}

#[cube]
pub fn swizzle(offset: u32, #[comptime] bank_count: i32) -> u32 {
    let num_bits = comptime!(i32::BITS - bank_count.leading_zeros() - 1);
    let bit_mask = (1 << num_bits) - 1;
    let yyy_mask = bit_mask << (num_bits);
    let mask_shift = num_bits;

    offset ^ ((offset & yyy_mask) >> mask_shift)
}
