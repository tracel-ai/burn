use crate::{kernel::into_contiguous, tensor::JitTensor, FloatElement, JitRuntime};
use burn_cube::prelude::*;

use super::{tiling2d_launch_options, Tiling2dConfig};

impl Init for CubeTiling2dConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Debug, Clone, Copy)]
/// Tiling 2D parameters
pub struct CubeTiling2dConfig {
    /// Block size along dimension of lhs
    pub block_size_m: UInt,
    /// Block size along common dimension
    pub block_size_k: UInt,
    /// Block size along dimension of rhs
    pub block_size_n: UInt,
    /// Loop unrolling
    pub unroll: bool,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Shared memory size lhs: technically derivable from others, but needs comptime arithmetic
    pub sm_size_lhs: UInt,
    /// Shared memory size rhs: technically derivable from others, but needs comptime arithmetic
    pub sm_size_rhs: UInt,
}

impl CubeTiling2dConfig {
    fn new(config: Tiling2dConfig, m: usize, k: usize, n: usize) -> Self {
        let tile_size = config.tile_size_m;
        let sm_size_lhs = config.block_size_m * config.block_size_k * tile_size;
        let sm_size_rhs = config.block_size_n * config.block_size_k * tile_size;

        CubeTiling2dConfig {
            block_size_m: UInt::new(config.block_size_m as u32),
            block_size_k: UInt::new(config.block_size_k as u32),
            block_size_n: UInt::new(config.block_size_n as u32),
            unroll: config.unroll,
            check_m_bounds: m % config.block_size_m != 0,
            check_k_bounds: k % config.block_size_k != 0,
            check_n_bounds: n % config.block_size_n != 0,
            sm_size_lhs: UInt::new(sm_size_lhs as u32),
            sm_size_rhs: UInt::new(sm_size_rhs as u32),
        }
    }
}

#[derive(CubeType, Copy, Clone)]
struct Tiling2dState<F: Float> {
    pub n_loops: UInt,
    pub k: UInt,
    pub lhs: Tensor<F>,
    pub rhs: Tensor<F>,
    pub out: Tensor<F>,
    pub offset_lhs: UInt,
    pub offset_rhs: UInt,
    pub offset_output: UInt,
    pub row: UInt,
    pub col: UInt,
    pub dim_m: UInt,
    pub dim_k: UInt,
    pub dim_n: UInt,
    pub unit_col: UInt,
    pub unit_row: UInt,
    pub shared_lhs: SharedMemory<F>,
    pub shared_rhs: SharedMemory<F>,
    pub register_m: F,
    pub register_n: F,
    pub results: Array<F>,
    pub lhs_stride_col: UInt,
    pub lhs_stride_row: UInt,
    pub rhs_stride_col: UInt,
    pub rhs_stride_row: UInt,
    pub out_stride_row: UInt,
    pub out_stride_col: UInt,
}

#[cube]
fn gather_kernel_information<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) -> Tiling2dState<F> {
    // Config variables
    let block_size_m = Comptime::map(config, |c| c.block_size_m);
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let sm_size_lhs = Comptime::map(config, |c| c.sm_size_lhs);
    let sm_size_rhs = Comptime::map(config, |c| c.sm_size_rhs);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    // Assumes lhs and rhs share vectorization factor
    let tile_size = Comptime::vectorization(lhs);

    // Topology info
    let n_threads_per_row = ((Comptime::runtime(block_size_n) - UInt::new(1))
        / Comptime::runtime(tile_size))
        + UInt::new(1);
    let local_idx = UNIT_POS;
    let batch = ABSOLUTE_POS_Z;

    // Shapes
    let rank = out.rank();
    let dim_m = lhs.shape(rank - UInt::new(2));
    let dim_k = lhs.shape(rank - UInt::new(1));
    let dim_n = rhs.shape(rank - UInt::new(1));

    // Strides
    let lhs_stride_row = lhs.stride(rank - UInt::new(2));
    let lhs_stride_col = lhs.stride(rank - UInt::new(1));
    let rhs_stride_row = rhs.stride(rank - UInt::new(2));
    let rhs_stride_col = rhs.stride(rank - UInt::new(1));
    let out_stride_row = out.stride(rank - UInt::new(2));
    let out_stride_col = out.stride(rank - UInt::new(1));

    // Cube offset
    let skip_row = CUBE_POS_X * Comptime::runtime(block_size_m);
    let skip_col = CUBE_POS_Y * Comptime::runtime(block_size_n);

    // Position of the first element of the unit, relative to the cube
    let unit_row = (local_idx / n_threads_per_row) * Comptime::runtime(tile_size);
    let unit_col = (local_idx % n_threads_per_row) * Comptime::runtime(tile_size);

    // Position of the first element of the unit, in absolute (in one batch)
    let row = skip_row + unit_row;
    let col = skip_col + unit_col;

    // Batch offset for output
    let offset_output = dim_m * dim_n * batch / Comptime::runtime(tile_size);

    // Calculate offset for lhs and rhs, without regards to batches
    let mut offset_lhs = skip_row * lhs_stride_row;
    let mut offset_rhs = skip_col * rhs_stride_col;

    // Batch offset for lhs, rhs
    for b in range(0u32, rank - UInt::new(2), unroll) {
        let tmp = offset_output / out.stride(b);
        offset_lhs += tmp % lhs.shape(b) * lhs.stride(b);
        offset_rhs += tmp % rhs.shape(b) * rhs.stride(b);
    }

    offset_lhs /= Comptime::runtime(tile_size);
    offset_rhs /= Comptime::runtime(tile_size);

    let tile_squared = Comptime::zip(tile_size, tile_size, |c1, c2| UInt::new(c1.val * c2.val));
    let results = Array::<F>::new(Comptime::get(tile_squared));

    let shared_lhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_lhs), Comptime::get(tile_size));
    let shared_rhs =
        SharedMemory::<F>::vectorized(Comptime::get(sm_size_rhs), Comptime::get(tile_size));

    // Calculate exact number of loop iterations
    let mut n_loops = UInt::new(0); // TODO support syntax let x = if ... else ...
    if Comptime::get(check_k_bounds) {
        n_loops = UInt::cast_from(F::ceil(
            F::cast_from(dim_k) / F::cast_from(Comptime::runtime(block_size_k)),
        ));
    } else {
        n_loops = dim_k / Comptime::runtime(block_size_k);
    }

    // Dummy declarations
    let k = UInt::new(0);
    let register_m = F::vectorized(0., Comptime::get(tile_size));
    let register_n = F::vectorized(0., Comptime::get(tile_size));

    Tiling2dState {
        n_loops,
        k,
        lhs,
        rhs,
        out,
        offset_lhs,
        offset_rhs,
        offset_output,
        row,
        col,
        dim_m,
        dim_k,
        dim_n,
        unit_col,
        unit_row,
        shared_lhs,
        shared_rhs,
        register_m,
        register_n,
        results,
        lhs_stride_col,
        lhs_stride_row,
        rhs_stride_col,
        rhs_stride_row,
        out_stride_row,
        out_stride_col,
    }
}

#[cube]
fn load_shared_memory<F: Float>(
    kernel_state: Tiling2dState<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    load_lhs_tensor_plain(
        kernel_state.lhs,
        kernel_state.offset_lhs,
        kernel_state.shared_lhs,
        kernel_state.unit_col,
        kernel_state.unit_row,
        kernel_state.lhs_stride_col,
        kernel_state.lhs_stride_row,
        kernel_state.dim_m,
        kernel_state.row,
        kernel_state.col,
        kernel_state.k,
        kernel_state.dim_k,
        config,
    );
    load_rhs_tensor_transposed(
        kernel_state.rhs,
        kernel_state.offset_rhs,
        kernel_state.shared_rhs,
        kernel_state.unit_row,
        kernel_state.unit_col,
        kernel_state.rhs_stride_row,
        kernel_state.rhs_stride_col,
        kernel_state.dim_n,
        kernel_state.row,
        kernel_state.col,
        kernel_state.k,
        kernel_state.dim_k,
        config,
    )
}

#[cube]
/// Assumes vectorization is in the same orientation we need in shared memory
fn load_lhs_tensor_plain<F: Float>(
    lhs: Tensor<F>,
    offset_lhs: UInt,
    mut shared_lhs: SharedMemory<F>,
    unit_col: UInt,
    unit_row: UInt,
    lhs_stride_col: UInt,
    lhs_stride_row: UInt,
    dim_m: UInt,
    row: UInt,
    col: UInt,
    k: UInt,
    dim_k: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::vectorization(lhs);
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);

    let position_in_lhs_base = (k + unit_col) * lhs_stride_col
        + unit_row * (lhs_stride_row / Comptime::runtime(tile_size))
        + offset_lhs;
    let sm_position_base =
        (unit_row * Comptime::runtime(block_size_k) + unit_col) / Comptime::runtime(tile_size);
    let sm_stride = Comptime::runtime(block_size_k) / Comptime::runtime(tile_size);

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_k_bounds) {
            if col >= dim_k {
                for i in range(0u32, Comptime::get(tile_size), unroll) {
                    let sm_position = sm_position_base + i * sm_stride;
                    shared_lhs[sm_position] = F::vectorized(0., Comptime::get(tile_size));
                }
            } else {
                let num_reads = UInt::min(dim_m - row, Comptime::runtime(tile_size));
                for i in range(0u32, num_reads, Comptime::new(false)) {
                    let sm_position = sm_position_base + i * sm_stride;
                    let position_in_lhs =
                        position_in_lhs_base + i * (lhs_stride_row / Comptime::runtime(tile_size));
                    shared_lhs[sm_position] = lhs[position_in_lhs];
                }
                for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
                    let sm_position = sm_position_base + i * sm_stride;
                    shared_lhs[sm_position] = F::vectorized(0., Comptime::get(tile_size));
                }
            }
        } else {
            let num_reads = UInt::min(dim_m - row, Comptime::runtime(tile_size));
            for i in range(0u32, num_reads, Comptime::new(false)) {
                let sm_position = sm_position_base + i * sm_stride;
                let position_in_lhs =
                    position_in_lhs_base + i * (lhs_stride_row / Comptime::runtime(tile_size));
                shared_lhs[sm_position] = lhs[position_in_lhs];
            }
            for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
                let sm_position = sm_position_base + i * sm_stride;
                shared_lhs[sm_position] = F::vectorized(0., Comptime::get(tile_size));
            }
        }
    } else {
        if Comptime::get(check_k_bounds) {
            if col >= dim_k {
                for i in range(0u32, Comptime::get(tile_size), unroll) {
                    let sm_position = sm_position_base + i * sm_stride;
                    shared_lhs[sm_position] = F::vectorized(0., Comptime::get(tile_size));
                }
            } else {
                for i in range(0u32, Comptime::get(tile_size), unroll) {
                    let sm_position = sm_position_base + i * sm_stride;
                    let position_in_lhs =
                        position_in_lhs_base + i * (lhs_stride_row / Comptime::runtime(tile_size));
                    shared_lhs[sm_position] = lhs[position_in_lhs];
                }
            }
        } else {
            for i in range(0u32, Comptime::get(tile_size), unroll) {
                let sm_position = sm_position_base + i * sm_stride;
                let position_in_lhs =
                    position_in_lhs_base + i * (lhs_stride_row / Comptime::runtime(tile_size));
                shared_lhs[sm_position] = lhs[position_in_lhs];
            }
        }
    }
}

#[cube]
fn load_rhs_tensor_transposed<F: Float>(
    rhs: Tensor<F>,
    offset_rhs: UInt,
    mut shared_rhs: SharedMemory<F>,
    unit_row: UInt,
    unit_col: UInt,
    rhs_stride_row: UInt,
    rhs_stride_col: UInt,
    dim_n: UInt,
    row: UInt,
    col: UInt,
    k: UInt,
    dim_k: UInt,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::vectorization(rhs);
    let check_k_bounds = Comptime::map(config, |c| c.check_k_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    let position_base = (k + unit_row) * rhs_stride_row + unit_col * rhs_stride_col + offset_rhs;
    let sm_position_base =
        (unit_col * Comptime::runtime(block_size_k) + unit_row) / Comptime::runtime(tile_size);
    let sm_stride = Comptime::runtime(block_size_k) / Comptime::runtime(tile_size);

    // Read entries
    let mut entries = Array::<F>::vectorized(Comptime::get(tile_size), Comptime::get(tile_size));
    if Comptime::get(check_k_bounds) {
        if Comptime::get(check_n_bounds) {
            // We assume whole vectorization is out of bound
            if col >= dim_n {
                for i in range(0u32, Comptime::get(tile_size), unroll) {
                    entries[i] = F::vectorized(0., Comptime::get(tile_size));
                }
            } else {
                let num_reads = UInt::min(dim_k - row, Comptime::runtime(tile_size));
                for i in range(0u32, num_reads, Comptime::new(false)) {
                    entries[i] = rhs[position_base + i * rhs_stride_row];
                }
                for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
                    entries[i] = F::vectorized(0., Comptime::get(tile_size));
                }
            }
        } else {
            let num_reads = UInt::min(dim_k - row, Comptime::runtime(tile_size));
            for i in range(0u32, num_reads, Comptime::new(false)) {
                entries[i] = rhs[position_base + i * rhs_stride_row];
            }
            for i in range(num_reads, Comptime::get(tile_size), Comptime::new(false)) {
                entries[i] = F::vectorized(0., Comptime::get(tile_size));
            }
        }
    } else {
        if Comptime::get(check_n_bounds) {
            // We assume whole vectorization is out of bound
            if col >= dim_n {
                for i in range(0u32, Comptime::get(tile_size), unroll) {
                    entries[i] = F::vectorized(0., Comptime::get(tile_size));
                }
            } else {
                for i in range(0u32, Comptime::get(tile_size), unroll) {
                    entries[i] = rhs[position_base + i * rhs_stride_row];
                }
            }
        } else {
            for i in range(0u32, Comptime::get(tile_size), unroll) {
                entries[i] = rhs[position_base + i * rhs_stride_row];
            }
        }
    }

    // Decompose vectorization then recompose as transposed
    for i in range(0u32, Comptime::get(tile_size), unroll) {
        let mut transposed = Array::<F>::new(Comptime::get(tile_size));
        for j in range(0u32, Comptime::get(tile_size), unroll) {
            transposed[j] = entries[j][i];
        }
        let sm_position = sm_position_base + i * sm_stride;
        shared_rhs[sm_position] = transposed.to_vectorized(tile_size);
    }
}

#[cube]
fn computation_loop<F: Float>(
    kernel_state: Tiling2dState<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let unit_col = kernel_state.unit_col;
    let unit_row = kernel_state.unit_row;
    let shared_lhs = kernel_state.shared_lhs;
    let shared_rhs = kernel_state.shared_rhs;
    let mut register_m = kernel_state.register_m;
    let mut register_n = kernel_state.register_n;
    let mut results = kernel_state.results;
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let block_size_n = Comptime::map(config, |c| c.block_size_n);
    let unroll = Comptime::map(config, |c| c.unroll);
    let tile_size = Comptime::vectorization(kernel_state.lhs);

    // TODO this would greatly beneficiate from comptime arithmetic so we could unroll
    let num_compute = Comptime::runtime(block_size_k) / Comptime::runtime(tile_size);
    let lhs_pos_base = unit_row / Comptime::runtime(tile_size) * Comptime::runtime(block_size_k);
    let rhs_pos_base = unit_col / Comptime::runtime(tile_size) * Comptime::runtime(block_size_k);

    for dot_index in range(0u32, num_compute, Comptime::new(false)) {
        let dot_index = Comptime::runtime(tile_size) * dot_index;
        let lhs_pos = lhs_pos_base + dot_index;
        let rhs_pos = rhs_pos_base + dot_index;

        register_m = shared_lhs[lhs_pos];
        register_n = shared_rhs[rhs_pos];

        // Naive version that decomposes vectorization
        for res_idx_m in range(0u32, Comptime::get(tile_size), unroll) {
            let res_pos_base = res_idx_m * Comptime::runtime(tile_size);
            for res_idx_n in range(0u32, Comptime::get(tile_size), unroll) {
                let mul = register_m[res_idx_m] * register_n[res_idx_n];
                results[res_pos_base + res_idx_n] += mul;
            }
        }
    }
}

#[cube]
fn write_to_output<F: Float>(
    mut kernel_state: Tiling2dState<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    // No bounds check version
    let row = kernel_state.row;
    let col = kernel_state.col;
    let out_stride_row = kernel_state.out_stride_row;
    let out_stride_col = kernel_state.out_stride_col;
    let offset_output = kernel_state.offset_output;
    let results = kernel_state.results;
    let unroll = Comptime::map(config, |c| c.unroll);
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);
    let tile_size = Comptime::vectorization(kernel_state.lhs);

    for res_idx_m in range(0u32, Comptime::get(tile_size), unroll) {
        let results_pos_m = res_idx_m * Comptime::runtime(tile_size);

        // TODO just reinterpret the array if possible
        let mut array = Array::<F>::new(Comptime::get(tile_size));
        for res_idx_n in range(0u32, Comptime::get(tile_size), unroll) {
            array[res_idx_n] = results[results_pos_m + res_idx_n];
        }

        let row_index = (row + res_idx_m) * out_stride_row;
        let col_index = col * out_stride_col;

        // FOR DEBUGGING
        // TODO: it's a pain to put a debug value in output if it's vectorized
        let print_value = out_stride_row;
        let mut out = Array::<F>::new(Comptime::get(tile_size));
        for i in range(0u32, Comptime::get(tile_size), unroll) {
            out[i] = F::cast_from(print_value) + F::new(10.);
        }

        kernel_state.out[row_index + col_index + offset_output] = out.to_vectorized(tile_size);
        // kernel_state.out[res_idx_m] = out.to_vectorized(tile_size);
        // F::vectorized(2., Comptime::get(tile_size));
    }
}

#[cube(launch)]
/// Kernel for tiling2d matmul
pub fn tiling2d_matmul_kernel<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    let block_size_k = Comptime::map(config, |c| c.block_size_k);
    let mut kernel_state = gather_kernel_information::<F>(lhs, rhs, out, config);

    for i in range(0u32, kernel_state.n_loops, Comptime::new(false)) {
        kernel_state.k = i * Comptime::runtime(block_size_k);

        load_shared_memory(kernel_state, config);

        sync_units();

        computation_loop(kernel_state, config);

        sync_units();
    }

    write_to_output(kernel_state, config);
}

/// Matrix multiplication using tiling 2d algorithm with
/// written in Cube
pub fn matmul_tiling_2d_cube<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let lhs = match lhs.batch_swapped_with_row_col() {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match rhs.batch_swapped_with_row_col() {
        true => into_contiguous(rhs),
        false => rhs,
    };

    let cube_count = tiling2d_launch_options(&out.shape, config.clone());

    let vectorization_factor = 1;
    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization_factor)
        .vectorize_input(1, vectorization_factor)
        .vectorize_output(0, vectorization_factor);

    tiling2d_matmul_kernel_launch::<E::CubeElement, R>(
        client,
        cube_count,
        settings,
        TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        CubeTiling2dConfig::new(config, m, k, n),
    );

    out
}
