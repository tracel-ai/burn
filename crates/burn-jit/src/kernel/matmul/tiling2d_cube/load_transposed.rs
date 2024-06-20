use burn_cube::prelude::*;

use super::config::CubeTiling2dConfig;

#[cube]
fn load_transposed<F: Float>(
    tensor: Tensor<F>,
    cube_offset: UInt, // counts k's impact + offset from cube and batches
    mut shared_memory: SharedMemory<F>,
    unit_row: UInt,
    unit_col: UInt,
    tensor_stride: UInt,
    sm_stride: UInt,
    sm_position_base: UInt,
    dim_k: UInt,
    dim_n: UInt,
    check_bottom_bounds: Comptime<bool>,
    check_right_bounds: Comptime<bool>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let skip_row = CUBE_POS_X * Comptime::runtime(block_size_m);
    let skip_col = CUBE_POS_Y * Comptime::runtime(block_size_n);
    let row = skip_row + unit_row;
    let col = skip_col + unit_col;

    let tensor_position_base = unit_row * tensor_stride + unit_col + cube_offset;

    // let lhs_sm_stride_row = Comptime::runtime(block_size_m) / Comptime::runtime(tile_size);
    // let rhs_sm_stride_row = Comptime::runtime(block_size_n) / Comptime::runtime(tile_size);

    // let lhs_sm_position_base = unit_col * lhs_sm_stride_row + unit_row;
    // let rhs_sm_position_base = unit_col * rhs_sm_stride_row + unit_row;

    // Read entries
    let mut entries = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));

    if Comptime::get(check_bottom_bounds) {
        if Comptime::get(check_right_bounds) {
            // We assume whole vectorization is out of bound
            if col >= dim_n {
                read_zeros(entries, config);
            } else {
                read_partial(
                    tensor,
                    dim_k,
                    row,
                    tensor_position_base,
                    tensor_stride,
                    entries,
                    config,
                );
            }
        } else {
            read_partial(
                tensor,
                dim_k,
                row,
                tensor_position_base,
                tensor_stride,
                entries,
                config,
            );
        }
    } else {
        if Comptime::get(check_n_bounds) {
            // We assume whole vectorization is out of bound
            if col >= dim_n {
                read_zeros(entries, config);
            } else {
                read_whole(tensor, tensor_position_base, tensor_stride, entries, config);
            }
        } else {
            read_whole(tensor, tensor_position_base, tensor_stride, entries, config);
        }
    }

    let is_scalar = Comptime::map(tile_size, |c| c.val == 1);
    if Comptime::get(is_scalar) {
        shared_memory[sm_position_base] = entries[0];
    } else {
        // TODO: there's a bug if we reuse i inside else clause
        for w in range(0u32, Comptime::get(tile_size), unroll) {
            let mut transposed = Array::<F>::new(Comptime::get(tile_size));
            for j in range(0u32, Comptime::get(tile_size), unroll) {
                transposed[j] = entries[j][w];
            }
            let sm_position = sm_position_base + w * sm_stride;
            shared_memory[sm_position] = transposed.to_vectorized(tile_size);
        }
    }
}

#[cube]
fn load_lhs_transposed<F: Float>(
    lhs: Tensor<F>,
    unit_row: UInt,
    unit_col: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let tile_size = Comptime::map(config, |c| c.tile_size);
    let rank = lhs.rank();
    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_k = lhs.shape(rank - UInt::new(1));

    let tensor_stride = dim_m / Comptime::runtime(tile_size);
    let sm_stride = block_size_m / tile_size;
    let sm_position_base = unit_col * sm_stride + unit_row;

    load_transposed(
        Comptime::map(config, |c| c.check_m_bounds),
        Comptime::map(config, |c| c.check_k_bounds),
    )
}

#[cube]
fn load_rhs_transposed(config: Comptime<CubeTiling2dConfig>) {
    // let rhs_sm_position_base = unit_col * rhs_sm_stride_row + unit_row;
    // let rhs_sm_stride_row = Comptime::runtime(block_size_n) / Comptime::runtime(tile_size);
    load_transposed(
        Comptime::map(config, |c| c.check_m_bounds),
        Comptime::map(config, |c| c.check_k_bounds),
    )
}

#[cube]
fn read_zeros<F: Float>(mut entries: Array<F>, config: Comptime<CubeTiling2dConfig>) {
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    for i in range(0u32, Comptime::get(tile_size), unroll) {
        entries[i] = F::vectorized(0., Comptime::get(tile_size));
    }
}

#[cube]
fn read_partial<F: Float>(
    tensor: Tensor<F>,
    dim_k: UInt,
    row: UInt,
    position_base: UInt,
    stride: UInt,
    mut entries: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let tile_size = Comptime::map(config, |c| c.tile_size);

    let num_reads = UInt::min(dim_k - row, Comptime::runtime(tile_size));
    for i in range(0u32, num_reads, Comptime::new(false)) {
        entries[i] = tensor[position_base + i * stride];
    }
    for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
        entries[i] = F::vectorized(0., Comptime::get(tile_size));
    }
}

#[cube]
fn read_whole<F: Float>(
    tensor: Tensor<F>,
    position_base: UInt,
    stride: UInt,
    mut entries: Array<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::map(config, |c| c.tile_size);

    for i in range(0u32, Comptime::get(tile_size), unroll) {
        entries[i] = tensor[position_base + i * stride];
    }
}
