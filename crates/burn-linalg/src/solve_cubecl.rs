//! Portable LU factorization and triangular solves for CubeCL devices.
//!
//! Only the singularity status leaves the device. A is factored before batch
//! broadcasting, and the original input allocations are never modified.
use alloc::vec::Vec;
use burn_core::backend::{DType, Shape, TensorMetadata, cubecl::dtype_to_storage_type};
use burn_cubecl::{
    cubecl::{self, prelude::*, std::FastDivmod},
    kernel::{
        matmul::{MatmulStrategy, matmul},
        slice, untile,
    },
    ops::{into_data_sync, numeric::empty_device_contiguous_dtype},
    tensor::CubeTensor,
};

const PANEL: usize = 32;

// WGSL may implement division through a reciprocal that flushes subnormal
// intermediates. Scaling both operands by an exact power of two keeps the
// reciprocal of even the largest finite F32 denominator in the normal range.
#[cube]
fn divide<F: Float>(numerator: F, denominator: F) -> F {
    let mut numerator = numerator;
    let mut denominator = denominator;
    if F::abs(denominator) > F::new(1.0_f32 / f32::MIN_POSITIVE) {
        numerator *= F::new(0.25_f32);
        denominator *= F::new(0.25_f32);
    }
    numerator / denominator
}

#[cube]
fn pivot_candidate_wins<F: Float>(
    value: F,
    row: u32,
    previous_value: F,
    previous_row: u32,
    invalid: u32,
) -> bool {
    let is_nan = value == F::new(-2.0_f32);
    let previous_is_nan = previous_value == F::new(-2.0_f32);
    let mut wins = false;
    if row < invalid {
        if previous_row == invalid {
            wins = true;
        } else if is_nan {
            wins = !previous_is_nan || row < previous_row;
        } else if !previous_is_nan {
            wins = value > previous_value || (value == previous_value && row < previous_row);
        }
    }
    wins
}

#[cube]
fn reduce_pivot<F: Float>(
    value: F,
    row: u32,
    invalid: u32,
    max_values: &mut Shared<[F]>,
    max_rows: &mut Shared<[u32]>,
    #[comptime] use_planes: bool,
) -> (F, u32) {
    let lane = UNIT_POS as usize;
    let units = CUBE_DIM as usize;
    let mut ordered_value = value;
    if bool::cast_from(value.is_nan()) {
        ordered_value = F::new(-2.0_f32);
    }
    if comptime!(use_planes) {
        // Three reductions: finite maximum, its smallest row, and a NaN row.
        // The extra integer reduction makes NaN handling independent of the
        // target's subgroupMax NaN behavior.
        let mut plane_value = plane_max(ordered_value);
        let mut winning_row = invalid;
        if ordered_value == plane_value {
            winning_row = row;
        }
        let mut plane_row = plane_min(winning_row);
        let mut nan_row = invalid;
        if ordered_value == F::new(-2.0_f32) {
            nan_row = row;
        }
        let plane_nan_row = plane_min(nan_row);
        if plane_nan_row < invalid {
            plane_value = F::new(-2.0_f32);
            plane_row = plane_nan_row;
        }

        if UNIT_POS_PLANE == 0 {
            max_values[PLANE_POS as usize] = plane_value;
            max_rows[PLANE_POS as usize] = plane_row;
        }
        sync_cube();

        // Every lane scans the few subgroup candidates. Returning the result
        // directly removes the second workgroup barrier needed to broadcast a
        // lane-zero result. The caller's row-exchange barrier fences these
        // reads before the next pivot can overwrite the candidate arrays.
        let plane_dim = PLANE_DIM as usize;
        let planes = units.div_ceil(plane_dim);
        let mut best_value = F::new(-1.0_f32);
        let mut best_row = invalid;
        for plane in 0..planes {
            let candidate_value = max_values[plane];
            let candidate_row = max_rows[plane];
            if pivot_candidate_wins::<F>(
                candidate_value,
                candidate_row,
                best_value,
                best_row,
                invalid,
            ) {
                best_value = candidate_value;
                best_row = candidate_row;
            }
        }
        (best_value, best_row)
    } else {
        max_values[lane] = ordered_value;
        max_rows[lane] = row;
        sync_cube();
        let mut distance = units / 2;
        while distance > 0 {
            if lane < distance {
                let other_value = max_values[lane + distance];
                let other_row = max_rows[lane + distance];
                if pivot_candidate_wins::<F>(
                    other_value,
                    other_row,
                    max_values[lane],
                    max_rows[lane],
                    invalid,
                ) {
                    max_values[lane] = other_value;
                    max_rows[lane] = other_row;
                }
            }
            sync_cube();
            distance /= 2;
        }
        (max_values[0], max_rows[0])
    }
}

// Store LU column-major: neighboring lanes own neighboring rows during panel
// factorization and substitution. Pack from arbitrary input strides in one pass.
#[cube(launch, address_type = "dynamic")]
fn pack<F: Float>(
    input: &Tensor<F>,
    lu: &mut Tensor<F>,
    permutation: &mut Tensor<u32>,
    status: &mut Tensor<u32>,
    #[define(F)] _dtype: ElemType,
) {
    let index = ABSOLUTE_POS;
    if index >= lu.len() {
        terminate!();
    }
    let n = lu.shape(lu.rank() - 1);
    let batch = index / (n * n);
    let row = index % n;
    let col = index / n % n;
    let mut logical = (batch * n + row) * n + col;
    let mut offset = 0usize;
    for rev in 0..input.rank() {
        let dim = input.rank() - 1 - rev;
        offset += logical % input.shape(dim) * input.stride(dim);
        logical /= input.shape(dim);
    }
    lu[index] = input[offset];
    if col == 0 {
        permutation[batch * n + row] = row as u32;
        if row == 0 {
            status[batch] = 0;
        }
    }
}

fn shared_panel_config(
    client: &cubecl::client::Client,
    n: usize,
    units: usize,
    elem_bytes: usize,
) -> Option<(usize, usize)> {
    let rows_capacity = (n.div_ceil(128) * 128) | 1;
    let budget = client.properties().hardware.max_shared_memory_size;
    let reduction_bytes = units.checked_mul(elem_bytes.checked_add(size_of::<u32>())?)?;
    let overhead = reduction_bytes.checked_add(256)?;
    let available = budget.checked_sub(overhead)?;
    // Each panel column needs one tall shared column plus two exchanged values.
    let bytes_per_column = rows_capacity.checked_add(2)?.checked_mul(elem_bytes)?;
    let max_panel = (available / bytes_per_column).min(if n <= 64 { 64 } else { 32 });
    if max_panel == 0 {
        return None;
    }
    let panel = 1usize << max_panel.ilog2();
    Some((rows_capacity, panel))
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn factor_panel_shared<F: Float>(
    lu: &mut Tensor<F>,
    pivots: &mut Tensor<u32>,
    permutation: &mut Tensor<u32>,
    status: &mut Tensor<u32>,
    start: u32,
    #[comptime] rows_capacity: usize,
    #[comptime] panel: usize,
    #[comptime] units: usize,
    #[comptime] use_planes: bool,
    #[define(F)] _dtype: ElemType,
) {
    let lane = UNIT_POS as usize;
    let n = lu.shape(lu.rank() - 1);
    let batch = CUBE_POS;
    if batch >= lu.len() / (n * n) {
        terminate!();
    }
    let base = batch * n * n;
    let start = start as usize;
    let remaining = n - start;
    let width = usize::min(panel, remaining);
    let mut rows = Shared::<[F]>::new_slice(rows_capacity * panel);
    let mut max_values = Shared::<[F]>::new_slice(units);
    let mut max_rows = Shared::<[u32]>::new_slice(units);
    let mut exchange = Shared::<[F]>::new_slice(2 * panel + 1);

    // Neighboring lanes load neighboring rows of each column.
    let mut local_row = lane;
    while local_row < remaining {
        for col in 0..width {
            rows[col * rows_capacity + local_row] =
                lu[base + (start + col) * n + start + local_row];
        }
        local_row += units;
    }

    for k in 0..width {
        let mut max_value = F::new(-1.0_f32);
        let mut max_row = n as u32;
        let mut local_row = lane;
        while local_row < remaining {
            if local_row >= k {
                let global_row = (start + local_row) as u32;
                let mut candidate = F::abs(rows[k * rows_capacity + local_row]);
                if bool::cast_from(candidate.is_nan()) {
                    candidate = F::new(-2.0_f32);
                }
                if pivot_candidate_wins::<F>(candidate, global_row, max_value, max_row, n as u32) {
                    max_value = candidate;
                    max_row = global_row;
                }
            }
            local_row += units;
        }
        let (pivot_value, pivot_row) = reduce_pivot::<F>(
            max_value,
            max_row,
            n as u32,
            &mut max_values,
            &mut max_rows,
            use_planes,
        );

        let pivot = pivot_row as usize;
        let pivot_local = pivot - start;
        if lane == 0 {
            pivots[batch * n + start + k] = pivot as u32;
            if pivot != start + k {
                let previous = permutation[batch * n + start + k];
                permutation[batch * n + start + k] = permutation[batch * n + pivot];
                permutation[batch * n + pivot] = previous;
            }
            if pivot_value == F::new(0.0_f32) {
                status[batch] = 1;
            }
        }

        // The reduction barrier makes the full panel visible. Threads now
        // cooperatively copy the two rows; nobody changes the panel yet.
        let mut col = lane;
        while col < width {
            exchange[col] = rows[col * rows_capacity + pivot_local];
            exchange[panel + col] = rows[col * rows_capacity + k];
            col += units;
        }
        if lane == 0 {
            let diagonal = rows[k * rows_capacity + pivot_local];
            let magnitude = F::abs(diagonal);
            let mut inverse = F::new(0.0_f32);
            if magnitude >= F::new(f32::MIN_POSITIVE)
                && magnitude <= F::new(1.0_f32 / f32::MIN_POSITIVE)
            {
                inverse = F::new(1.0_f32) / diagonal;
            }
            exchange[2 * panel] = inverse;
        }
        sync_cube();
        let diagonal = exchange[k];
        let inverse = exchange[2 * panel];

        let mut local_row = lane;
        while local_row < remaining {
            if pivot_local != k {
                if local_row == k {
                    for col in 0..width {
                        rows[col * rows_capacity + local_row] = exchange[col];
                    }
                } else if local_row == pivot_local {
                    for col in 0..width {
                        rows[col * rows_capacity + local_row] = exchange[panel + col];
                    }
                }
            }
            if local_row > k {
                let mut multiplier = F::new(0.0_f32);
                if inverse != F::new(0.0_f32) {
                    multiplier = rows[k * rows_capacity + local_row] * inverse;
                } else if diagonal != F::new(0.0_f32) {
                    multiplier = divide::<F>(rows[k * rows_capacity + local_row], diagonal);
                }
                rows[k * rows_capacity + local_row] = multiplier;
                for col in k + 1..width {
                    rows[col * rows_capacity + local_row] -= multiplier * exchange[col];
                }
            }
            local_row += units;
        }
        // No end barrier: the next reduction fences these updates before
        // exchange[] is overwritten or another lane's row is read.
    }

    let mut local_row = lane;
    while local_row < remaining {
        for col in 0..width {
            lu[base + (start + col) * n + start + local_row] =
                rows[col * rows_capacity + local_row];
        }
        local_row += units;
    }
}

// Each workgroup owns one panel. Rows live in private registers and only the
// pivot reduction and the two rows being exchanged use shared memory. This
// avoids fitting the entire tall panel into limited workgroup memory.
#[cube(launch, address_type = "dynamic")]
fn factor_panel<F: Float>(
    lu: &mut Tensor<F>,
    pivots: &mut Tensor<u32>,
    permutation: &mut Tensor<u32>,
    status: &mut Tensor<u32>,
    start: u32,
    #[comptime] rows_per_unit: usize,
    #[comptime] panel: usize,
    #[comptime] threads: usize,
    #[comptime] use_planes: bool,
    #[define(F)] _dtype: ElemType,
) {
    let lane = UNIT_POS as usize;
    let units = threads;
    let n = lu.shape(lu.rank() - 1);
    let batch = CUBE_POS;
    if batch >= lu.len() / (n * n) {
        terminate!();
    }
    let base = batch * n * n;
    let start = start as usize;
    let width = usize::min(panel, n - start);
    let mut rows = Array::<F>::new(rows_per_unit * panel);
    let mut max_values = Shared::<[F]>::new_slice(threads);
    let mut max_rows = Shared::<[u32]>::new_slice(threads);
    let mut exchange = Shared::<[F]>::new_slice(2 * panel);

    #[unroll]
    for r in 0..rows_per_unit {
        let row = start + lane + r * units;
        #[unroll]
        for col in 0..panel {
            let mut value = F::new(0.0_f32);
            if row < n && col < width {
                value = lu[base + (start + col) * n + row];
            }
            rows[r * panel + col] = value;
        }
    }

    for k in 0..width {
        let mut max_value = F::new(-1.0_f32);
        let mut max_row = n as u32;
        #[unroll]
        for r in 0..rows_per_unit {
            let row = start + lane + r * units;
            let mut candidate = F::new(0.0_f32);
            #[unroll]
            for col in 0..panel {
                if col == k {
                    candidate = F::abs(rows[r * panel + col]);
                }
            }
            if row >= start + k && row < n {
                if bool::cast_from(candidate.is_nan()) {
                    candidate = F::new(-2.0_f32);
                }
                if pivot_candidate_wins::<F>(candidate, row as u32, max_value, max_row, n as u32) {
                    max_value = candidate;
                    max_row = row as u32;
                }
            }
        }
        let (pivot_value, pivot_row) = reduce_pivot::<F>(
            max_value,
            max_row,
            n as u32,
            &mut max_values,
            &mut max_rows,
            use_planes,
        );
        let pivot = pivot_row as usize;
        if lane == 0 {
            pivots[batch * n + start + k] = pivot as u32;
            let old = permutation[batch * n + start + k];
            permutation[batch * n + start + k] = permutation[batch * n + pivot];
            permutation[batch * n + pivot] = old;
            if pivot_value == F::new(0.0_f32) {
                status[batch] = 1;
            }
        }
        #[unroll]
        for r in 0..rows_per_unit {
            let row = start + lane + r * units;
            #[unroll]
            for col in 0..panel {
                if row == pivot {
                    exchange[col] = rows[r * panel + col];
                }
                if row == start + k {
                    exchange[panel + col] = rows[r * panel + col];
                }
            }
        }
        sync_cube();
        let diagonal = exchange[k];
        #[unroll]
        for r in 0..rows_per_unit {
            let row = start + lane + r * units;
            #[unroll]
            for col in 0..panel {
                if row == start + k {
                    rows[r * panel + col] = exchange[col];
                } else if row == pivot {
                    rows[r * panel + col] = exchange[panel + col];
                }
            }
            let mut value = F::new(0.0_f32);
            #[unroll]
            for col in 0..panel {
                if col == k {
                    value = rows[r * panel + col];
                }
            }
            // Continue safely after a zero pivot; the final status check reports
            // singularity. Avoid forming an overflowing reciprocal for tiny pivots.
            let mut multiplier = F::new(0.0_f32);
            if diagonal != F::new(0.0_f32) {
                multiplier = divide::<F>(value, diagonal);
            }
            if row > start + k && row < n {
                #[unroll]
                for col in 0..panel {
                    if col == k {
                        rows[r * panel + col] = multiplier;
                    } else if col > k {
                        rows[r * panel + col] -= multiplier * exchange[col];
                    }
                }
            }
        }
        // Every lane must finish reading the exchanged rows before the next
        // pivot can overwrite the shared reduction and exchange buffers.
        sync_cube();
    }
    #[unroll]
    for r in 0..rows_per_unit {
        let row = start + lane + r * units;
        #[unroll]
        for col in 0..panel {
            if row < n && col < width {
                lu[base + (start + col) * n + row] = rows[r * panel + col];
            }
        }
    }
}

// Each lane owns one column. Apply the panel's swaps to columns outside the
// panel, then solve L11 U12 = A12. No workgroup-to-workgroup barriers are needed.
#[cube(launch, address_type = "dynamic")]
fn swap_and_solve_upper<F: Float>(
    lu: &mut Tensor<F>,
    pivots: &Tensor<u32>,
    start: u32,
    #[comptime] panel: usize,
    #[comptime] update_upper: bool,
    #[define(F)] _dtype: ElemType,
) {
    let n = lu.shape(lu.rank() - 1);
    let col = ABSOLUTE_POS % n;
    let batch = ABSOLUTE_POS / n;
    let start = start as usize;
    let width = usize::min(panel, n - start);
    if batch >= lu.len() / (n * n) || (col >= start && col < start + width) {
        terminate!();
    }
    let base = batch * n * n;
    for k in 0..width {
        let pivot = pivots[batch * n + start + k] as usize;
        if pivot != start + k {
            let old = lu[base + col * n + start + k];
            lu[base + col * n + start + k] = lu[base + col * n + pivot];
            lu[base + col * n + pivot] = old;
        }
    }
    if comptime!(update_upper) {
        if col < start + width {
            terminate!();
        }
        let mut values = Array::<F>::new(panel);
        #[unroll]
        for row in 0..panel {
            let mut value = F::new(0.0_f32);
            if row < width {
                value = lu[base + col * n + start + row];
            }
            values[row] = value;
        }
        #[unroll]
        for k in 0..panel {
            let current = values[k];
            #[unroll]
            for row in 0..panel {
                if row > k && row < width {
                    values[row] -= lu[base + (start + k) * n + start + row] * current;
                }
            }
        }
        #[unroll]
        for row in 0..panel {
            if row < width {
                lu[base + col * n + start + row] = values[row];
            }
        }
    }
}

#[cube(launch, address_type = "dynamic")]
fn update_trailing<F: Float, N: Size>(
    product: &Tensor<Vector<F, N>>,
    lu: &mut Tensor<Vector<F, N>>,
    start: u32,
    dimension: FastDivmod<usize>,
    #[define(F)] _dtype: ElemType,
) {
    let n = lu.shape(lu.rank() - 1);
    let start = start as usize;
    let remaining = n - start;
    let index = ABSOLUTE_POS;
    if index >= product.len() {
        terminate!();
    }
    let vector_size = product.vector_size();
    let (batch_col, row) = dimension.div_mod(index * vector_size);
    let (batch, col) = dimension.div_mod(batch_col);
    // Matmul allocates a row-major output with optional row padding. Its
    // leading batch dimensions are contiguous, so no rank-wise division is
    // needed for each element. The product is transposed relative to LU.
    let source = (batch * remaining + col) * product.stride(product.rank() - 2)
        + row * product.stride(product.rank() - 1);
    let target = batch * n * n + (start + col) * n + start + row;
    lu[target / vector_size] -= product[source / vector_size];
}

// Small systems keep an RHS tile in shared memory within one workgroup.
/// Result: (threads per workgroup, shared leading dimension, RHS tile width).
fn shared_rhs_config(
    client: &cubecl::client::Client,
    n: usize,
    rhs: usize,
    elem_bytes: usize,
    preferred_units: usize,
    preferred_tile: usize,
) -> Option<(usize, usize, usize)> {
    if n == 0 || rhs == 0 || elem_bytes == 0 {
        return None;
    }
    let hardware = &client.properties().hardware;
    let max_units = (hardware.max_units_per_cube.min(hardware.max_cube_dim.0) as usize)
        .min(preferred_units.max(1))
        .max(1);
    let units = n.next_power_of_two().min(1usize << max_units.ilog2());
    let rows_capacity = n | 1;
    // Include both solved-row buffers and reserve alignment slack.
    let available = hardware.max_shared_memory_size.checked_sub(256)?;
    let bytes_per_column = rows_capacity.checked_add(2)?.checked_mul(elem_bytes)?;
    let max_tile = (available / bytes_per_column)
        .min(preferred_tile.max(1))
        .min(rhs.next_power_of_two());
    if max_tile == 0 {
        return None;
    }
    let rhs_tile = 1usize << max_tile.ilog2();
    Some((units, rows_capacity, rhs_tile))
}

#[cube(launch, address_type = "dynamic")]
fn solve_rhs_shared<F: Float>(
    lu: &Tensor<F>,
    permutation: &Tensor<u32>,
    b: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] rows_capacity: usize,
    #[comptime] rhs_tile: usize,
    #[comptime] units: usize,
    #[define(F)] _dtype: ElemType,
) {
    let rank = output.rank();
    let n = output.shape(rank - 2);
    let rhs = output.shape(rank - 1);
    let lane = UNIT_POS as usize;
    let tiles = rhs.div_ceil(rhs_tile);
    let batch = CUBE_POS / tiles;
    let first_col = (CUBE_POS % tiles) * rhs_tile;
    if batch >= output.len() / (n * rhs) {
        terminate!();
    }
    let width = usize::min(rhs_tile, rhs - first_col);
    let output_base = batch * n * rhs;
    let mut a_base = 0usize;
    let mut b_base = 0usize;
    for dim in 0..rank - 2 {
        let coordinate = output_base / output.stride(dim) % output.shape(dim);
        a_base += coordinate % lu.shape(dim) * lu.stride(dim);
        b_base += coordinate % b.shape(dim) * b.stride(dim);
    }
    let permutation_base = a_base / n;
    let mut values = Shared::<[F]>::new_slice(rows_capacity * rhs_tile);
    let mut row = lane;
    while row < n {
        let source_row = permutation[permutation_base + row] as usize;
        for col in 0..width {
            values[col * rows_capacity + row] = b
                [b_base + source_row * b.stride(rank - 2) + (first_col + col) * b.stride(rank - 1)];
        }
        row += units;
    }

    let mut current = Shared::<[F]>::new_slice(2 * rhs_tile);
    for k in 0..n {
        let slot = (k % 2) * rhs_tile;
        if lane == k % units {
            for col in 0..width {
                current[slot + col] = values[col * rows_capacity + k];
            }
        }
        sync_cube();
        // First row owned by this lane strictly below k. Avoid scanning
        // already solved rows or testing ownership against private arrays.
        let mut row = lane + ((k + units - lane) / units) * units;
        while row < n {
            let lower = lu[a_base + k * n + row];
            for col in 0..width {
                let index = col * rows_capacity + row;
                values[index] -= lower * current[slot + col];
            }
            row += units;
        }
    }
    for reverse in 0..n {
        let k = n - 1 - reverse;
        let slot = ((n + reverse) % 2) * rhs_tile;
        // Keep normalization on the owning lane: other lanes may still
        // be finishing the prior update before this barrier.
        if lane == k % units {
            let diagonal = lu[a_base + k * n + k];
            for col in 0..width {
                let index = col * rows_capacity + k;
                let solved = divide::<F>(values[index], diagonal);
                values[index] = solved;
                current[slot + col] = solved;
            }
        }
        sync_cube();
        let mut row = lane;
        while row < k {
            let upper = lu[a_base + k * n + row];
            for col in 0..width {
                let index = col * rows_capacity + row;
                values[index] -= upper * current[slot + col];
            }
            row += units;
        }
    }
    let mut row = lane;
    while row < n {
        for col in 0..width {
            output[output_base + row * rhs + first_col + col] = values[col * rows_capacity + row];
        }
        row += units;
    }
}

struct BlockedRhsConfig {
    units: usize,
    panel: usize,
    rhs_tile: usize,
    row_tile: usize,
    use_shuffle: bool,
}

fn blocked_rhs_config(
    client: &cubecl::client::Client,
    dtype: DType,
    rhs: usize,
    preferred_units: usize,
    preferred_tile: usize,
    row_tile: usize,
) -> BlockedRhsConfig {
    let hardware = &client.properties().hardware;
    let max_units = (hardware.max_units_per_cube.min(hardware.max_cube_dim.0) as usize)
        .min(preferred_units.max(1))
        .max(1);
    let plane = hardware.plane_size_min as usize;
    let use_shuffle = plane > 1
        && plane == hardware.plane_size_max as usize
        && plane <= 64
        && plane <= max_units
        && client
            .features()
            .plane
            .contains(cubecl::ir::features::Plane::Ops)
        && dtype == DType::F32;
    let panel = if use_shuffle { plane } else { 32 };
    let mut rhs_tile = rhs.next_power_of_two().min(preferred_tile.max(1)).min(8);
    if use_shuffle {
        rhs_tile = rhs_tile.min(max_units / panel);
    }
    let shared_cap = hardware.max_shared_memory_size.saturating_sub(256) / (panel * dtype.size());
    rhs_tile = rhs_tile.min(shared_cap).max(1);
    rhs_tile = 1usize << rhs_tile.ilog2();
    let units = if use_shuffle {
        panel * rhs_tile
    } else {
        1usize << max_units.ilog2()
    };
    BlockedRhsConfig {
        units,
        panel,
        rhs_tile,
        row_tile,
        use_shuffle,
    }
}

/// Gather P*B once from arbitrary input strides into broadcast column-major B.
#[cube(launch, address_type = "dynamic")]
fn gather_rhs<F: Float>(
    lu: &Tensor<F>,
    permutation: &Tensor<u32>,
    b: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let index = ABSOLUTE_POS;
    if index >= output.len() {
        terminate!();
    }
    let rank = output.rank();
    let n = output.shape(rank - 2);
    let rhs = output.shape(rank - 1);
    let batch = index / (n * rhs);
    let row = index % n;
    let col = index / n % rhs;
    let batch_base = batch * n * rhs;
    let mut a_base = 0usize;
    let mut b_base = 0usize;
    for dim in 0..rank - 2 {
        let coordinate = batch_base / output.stride(dim) % output.shape(dim);
        a_base += coordinate % lu.shape(dim) * lu.stride(dim);
        b_base += coordinate % b.shape(dim) * b.stride(dim);
    }
    let source_row = permutation[a_base / n + row] as usize;
    output[index] = b[b_base + source_row * b.stride(rank - 2) + col * b.stride(rank - 1)];
}

/// One panel step of L*Y=P*B (forward) or U*X=Y (backward).
#[cube(launch_unchecked, address_type = "dynamic")]
fn blocked_rhs<F: Float>(
    lu: &Tensor<F>,
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    start: u32,
    #[comptime] units: usize,
    #[comptime] panel: usize,
    #[comptime] rhs_tile: usize,
    #[comptime] row_tile: usize,
    #[comptime] use_shuffle: bool,
    #[comptime] forward: bool,
    #[define(F)] _dtype: ElemType,
) {
    let rank = output.rank();
    let n = output.shape(rank - 2);
    let rhs = output.shape(rank - 1);
    let lane = UNIT_POS as usize;
    let row_tiles = n.div_ceil(row_tile);
    let rhs_tiles = rhs.div_ceil(rhs_tile);
    let group = CUBE_POS;
    let batch = group / (row_tiles * rhs_tiles);
    if batch >= output.len() / (n * rhs) {
        terminate!();
    }
    let row_begin = (group % row_tiles) * row_tile;
    let first_col = (group / row_tiles % rhs_tiles) * rhs_tile;
    let cols = usize::min(rhs_tile, rhs - first_col);
    let start = start as usize;
    let width = usize::min(panel, n - start);
    let end = start + width;
    let batch_base = batch * n * rhs;
    let mut a_base = 0usize;
    for dim in 0..rank - 2 {
        let coordinate = batch_base / output.stride(dim) % output.shape(dim);
        a_base += coordinate % lu.shape(dim) * lu.stride(dim);
    }
    let mut solved = Shared::<[F]>::new_slice(panel * rhs_tile);

    if comptime!(use_shuffle) {
        // Host only enables this when the hardware subgroup size is fixed and
        // equals panel, and units is exactly panel*rhs_tile.
        let row = UNIT_POS_PLANE as usize;
        let col = PLANE_POS as usize;
        let mut value = F::new(0.0_f32);
        if row < width && col < cols {
            value = input[batch_base + (first_col + col) * n + start + row];
        }
        if comptime!(forward) {
            #[unroll]
            for k in 0..panel {
                let current = plane_shuffle(value, k as u32);
                if row > k && row < width {
                    value -= lu[a_base + (start + k) * n + start + row] * current;
                }
            }
        } else {
            #[unroll]
            for reverse in 0..panel {
                let k = panel - 1 - reverse;
                if row == k && k < width {
                    value = divide::<F>(value, lu[a_base + (start + k) * n + start + k]);
                }
                let current = plane_shuffle(value, k as u32);
                if row < k && k < width {
                    value -= lu[a_base + (start + k) * n + start + row] * current;
                }
            }
        }
        if row < width && col < cols {
            solved[col * panel + row] = value;
        }
        sync_cube();
    } else {
        let mut entry = lane;
        while entry < panel * rhs_tile {
            let row = entry % panel;
            let col = entry / panel;
            if row < width && col < cols {
                solved[entry] = input[batch_base + (first_col + col) * n + start + row];
            }
            entry += units;
        }
        sync_cube();
        if comptime!(forward) {
            for k in 0..width {
                let mut entry = lane;
                while entry < panel * rhs_tile {
                    let row = entry % panel;
                    let col = entry / panel;
                    if row > k && row < width && col < cols {
                        let current = solved[col * panel + k];
                        solved[entry] -= lu[a_base + (start + k) * n + start + row] * current;
                    }
                    entry += units;
                }
                sync_cube();
            }
        } else {
            for reverse in 0..width {
                let k = width - 1 - reverse;
                let mut col = lane;
                while col < cols {
                    let value = solved[col * panel + k];
                    solved[col * panel + k] =
                        divide::<F>(value, lu[a_base + (start + k) * n + start + k]);
                    col += units;
                }
                sync_cube();
                let mut entry = lane;
                while entry < panel * rhs_tile {
                    let row = entry % panel;
                    let col = entry / panel;
                    if row < k && col < cols {
                        let current = solved[col * panel + k];
                        solved[entry] -= lu[a_base + (start + k) * n + start + row] * current;
                    }
                    entry += units;
                }
                sync_cube();
            }
        }
    }

    // One lane handles one row at a time and reuses each LU coefficient across
    // the RHS tile. Array indices become constants after the tiny column loop
    // unroll; only the RHS tile, never n, determines register storage.
    let mut local_row = lane;
    while local_row < row_tile {
        let row = row_begin + local_row;
        if row < n {
            let mut values = Array::<F>::new(rhs_tile);
            #[unroll]
            for col in 0..rhs_tile {
                let mut value = F::new(0.0_f32);
                if col < cols {
                    value = input[batch_base + (first_col + col) * n + row];
                }
                values[col] = value;
            }
            if row >= start && row < end {
                #[unroll]
                for col in 0..rhs_tile {
                    if col < cols {
                        values[col] = solved[col * panel + row - start];
                    }
                }
            } else {
                let mut update = row < start;
                if comptime!(forward) {
                    update = row >= end;
                }
                if update {
                    #[unroll]
                    for k in 0..panel {
                        if k < width {
                            let coefficient = lu[a_base + (start + k) * n + row];
                            #[unroll]
                            for col in 0..rhs_tile {
                                if col < cols {
                                    values[col] -= coefficient * solved[col * panel + k];
                                }
                            }
                        }
                    }
                }
            }
            #[unroll]
            for col in 0..rhs_tile {
                if col < cols {
                    output[batch_base + (first_col + col) * n + row] = values[col];
                }
            }
        }
        local_row += units;
    }
}

// Include every allocation used by a launch: broadcast results can require
// wider indexing than either input, and input views can retain larger handles.
fn address_type(tensors: &[&CubeTensor]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}

fn allocate(reference: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    if shape.num_elements() == 0 {
        return CubeTensor::new_contiguous(
            reference.client.clone(),
            reference.device.clone(),
            shape,
            reference.client.empty(dtype.size()),
            dtype,
        );
    }
    empty_device_contiguous_dtype(
        reference.client.clone(),
        reference.device.clone(),
        shape,
        dtype,
    )
}

pub(crate) fn solve(a: CubeTensor, b: CubeTensor) -> CubeTensor {
    let a = untile(a);
    let b = untile(b);
    let shape = a.shape();
    let rank = shape.num_dims();
    let n = shape[rank - 1];
    let matrices = shape.num_elements() / (n * n);
    let client = a.client.clone();
    let dtype = dtype_to_storage_type(a.dtype);
    let mut lu = allocate(&a, shape.clone(), a.dtype);
    lu.meta.swap(rank - 2, rank - 1);
    let pivots = allocate(&a, Shape::new([matrices, n]), DType::U32);
    let permutation = allocate(&a, Shape::new([matrices, n]), DType::U32);
    let status = allocate(&a, Shape::new([matrices]), DType::U32);
    let max_units = client
        .properties()
        .hardware
        .max_units_per_cube
        .min(client.properties().hardware.max_cube_dim.0)
        .min(256) as usize;
    let units = n
        .next_power_of_two()
        .min(1usize << max_units.ilog2())
        .max(1);
    let cube = CubeDim::new_1d(units as u32);
    let factor_address = address_type(&[&lu, &pivots, &permutation, &status]);
    let pack_address = factor_address.max(a.required_address_type());
    pack::launch(
        &client,
        cubecl::calculate_cube_count_elemwise(&client, shape.num_elements(), cube),
        cube,
        pack_address,
        a.clone().into_tensor_arg(),
        lu.clone().into_tensor_arg(),
        permutation.clone().into_tensor_arg(),
        status.clone().into_tensor_arg(),
        dtype,
    );
    let mut start = 0;
    while start < n {
        let panel_units = units.min((n - start).next_power_of_two());
        let panel_cube = CubeDim::new_1d(panel_units as u32);
        let shared_panel = shared_panel_config(&client, n - start, panel_units, a.dtype.size());
        let panel = shared_panel.map_or(PANEL, |(_, panel)| panel);
        if let Some((capacity, _)) = shared_panel {
            // SAFETY: capacity covers every remaining row; the kernel guards
            // batches/columns and always selects an existing pivot. Each group
            // owns one matrix, and its exchange barrier protects shared reuse.
            unsafe {
                factor_panel_shared::launch_unchecked(
                    &client,
                    cubecl::calculate_cube_count_elemwise(
                        &client,
                        matrices * panel_units,
                        panel_cube,
                    ),
                    panel_cube,
                    factor_address,
                    lu.clone().into_tensor_arg(),
                    pivots.clone().into_tensor_arg(),
                    permutation.clone().into_tensor_arg(),
                    status.clone().into_tensor_arg(),
                    start as u32,
                    capacity,
                    panel,
                    panel_units,
                    client
                        .features()
                        .plane
                        .contains(cubecl::ir::features::Plane::Ops)
                        && a.dtype == DType::F32,
                    dtype,
                );
            }
        } else {
            factor_panel::launch(
                &client,
                cubecl::calculate_cube_count_elemwise(&client, matrices * panel_units, panel_cube),
                panel_cube,
                factor_address,
                lu.clone().into_tensor_arg(),
                pivots.clone().into_tensor_arg(),
                permutation.clone().into_tensor_arg(),
                status.clone().into_tensor_arg(),
                start as u32,
                (n - start).div_ceil(panel_units),
                panel,
                panel_units,
                client
                    .features()
                    .plane
                    .contains(cubecl::ir::features::Plane::Ops)
                    && a.dtype == DType::F32,
                dtype,
            );
        }
        let end = (start + panel).min(n);
        if n > panel {
            swap_and_solve_upper::launch(
                &client,
                cubecl::calculate_cube_count_elemwise(&client, n * matrices, cube),
                cube,
                factor_address,
                lu.clone().into_tensor_arg(),
                pivots.clone().into_tensor_arg(),
                start as u32,
                panel,
                end < n,
                dtype,
            );
        }
        if end < n {
            let mut lower_ranges = shape.iter().map(|dim| 0..*dim).collect::<Vec<_>>();
            lower_ranges[rank - 2] = end..n;
            lower_ranges[rank - 1] = start..end;
            let mut upper_ranges = shape.iter().map(|dim| 0..*dim).collect::<Vec<_>>();
            upper_ranges[rank - 2] = start..end;
            upper_ranges[rank - 1] = end..n;
            let mut lower = slice(lu.clone(), &lower_ranges);
            let mut upper = slice(lu.clone(), &upper_ranges);
            lower.meta.swap(rank - 2, rank - 1);
            upper.meta.swap(rank - 2, rank - 1);
            let product = matmul(upper, lower, None, MatmulStrategy::Cube, a.dtype)
                .expect("linalg::solve: trailing matrix multiplication failed");
            let vector_size = if n.is_multiple_of(4)
                && end.is_multiple_of(4)
                && product.meta.strides()[rank - 2].is_multiple_of(4)
            {
                4
            } else {
                1
            };
            let update_address = address_type(&[&product, &lu]);
            update_trailing::launch(
                &client,
                cubecl::calculate_cube_count_elemwise(
                    &client,
                    product.shape().num_elements() / vector_size,
                    cube,
                ),
                cube,
                update_address,
                vector_size,
                product.into_tensor_arg(),
                lu.clone().into_tensor_arg(),
                end as u32,
                n - end,
                dtype,
            );
        }
        start = end;
    }
    let mut output_shape = b.shape();
    for dim in 0..rank - 2 {
        output_shape[dim] = shape[dim].max(output_shape[dim]);
    }
    let rhs = output_shape[rank - 1];
    let output = if rhs == 0 {
        allocate(&a, output_shape.clone(), a.dtype)
    } else {
        let small_rhs = if n < 32 {
            shared_rhs_config(&client, n, rhs, a.dtype.size(), 256, 8)
        } else {
            None
        };
        if let Some((solve_units, capacity, rhs_tile)) = small_rhs {
            let output = allocate(&a, output_shape.clone(), a.dtype);
            let rhs_address = address_type(&[&lu, &permutation, &b, &output]);
            let solve_cube = CubeDim::new_1d(solve_units as u32);
            let groups = output_shape.num_elements() / (n * rhs) * rhs.div_ceil(rhs_tile);
            solve_rhs_shared::launch(
                &client,
                cubecl::calculate_cube_count_elemwise(&client, groups * solve_units, solve_cube),
                solve_cube,
                rhs_address,
                lu.into_tensor_arg(),
                permutation.into_tensor_arg(),
                b.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                capacity,
                rhs_tile,
                solve_units,
                dtype,
            );
            output
        } else {
            let rhs_address = address_type(&[&lu, &permutation, &b]);
            let config = blocked_rhs_config(&client, a.dtype, rhs, 256, 1, 128);
            let mut physical_shape = output_shape.clone();
            physical_shape.swap(rank - 2, rank - 1);
            let mut read = allocate(&a, physical_shape.clone(), a.dtype);
            let mut write = allocate(&a, physical_shape, a.dtype);
            read.meta.swap(rank - 2, rank - 1);
            write.meta.swap(rank - 2, rank - 1);
            let rhs_address = rhs_address.max(address_type(&[&read, &write]));
            let cube = CubeDim::new_1d(config.units as u32);
            gather_rhs::launch(
                &client,
                cubecl::calculate_cube_count_elemwise(&client, output_shape.num_elements(), cube),
                cube,
                rhs_address,
                lu.clone().into_tensor_arg(),
                permutation.into_tensor_arg(),
                b.into_tensor_arg(),
                read.clone().into_tensor_arg(),
                dtype,
            );
            let groups = output_shape.num_elements() / (n * rhs)
                * rhs.div_ceil(config.rhs_tile)
                * n.div_ceil(config.row_tile);
            let count = cubecl::calculate_cube_count_elemwise(&client, groups * config.units, cube);
            for start in (0..n).step_by(config.panel) {
                // SAFETY: separate input/output allocations prevent intergroup
                // hazards. Guarded row/RHS tiles partition the output, and all
                // diagonal-panel reads are initialized before the shared barrier.
                unsafe {
                    blocked_rhs::launch_unchecked(
                        &client,
                        count.clone(),
                        cube,
                        rhs_address,
                        lu.clone().into_tensor_arg(),
                        read.clone().into_tensor_arg(),
                        write.clone().into_tensor_arg(),
                        start as u32,
                        config.units,
                        config.panel,
                        config.rhs_tile,
                        config.row_tile,
                        config.use_shuffle,
                        true,
                        dtype,
                    );
                }
                core::mem::swap(&mut read, &mut write);
            }
            for block in (0..n.div_ceil(config.panel)).rev() {
                // SAFETY: the same tile bounds and distinct-buffer invariants
                // hold while panels run in reverse; partial panels are guarded.
                unsafe {
                    blocked_rhs::launch_unchecked(
                        &client,
                        count.clone(),
                        cube,
                        rhs_address,
                        lu.clone().into_tensor_arg(),
                        read.clone().into_tensor_arg(),
                        write.clone().into_tensor_arg(),
                        (block * config.panel) as u32,
                        config.units,
                        config.panel,
                        config.rhs_tile,
                        config.row_tile,
                        config.use_shuffle,
                        false,
                        dtype,
                    );
                }
                core::mem::swap(&mut read, &mut write);
            }

            read
        }
    };
    let status = into_data_sync(status);
    assert!(
        status
            .as_slice::<u32>()
            .expect("linalg::solve: failed to read status")
            .iter()
            .all(|value| *value == 0),
        "linalg::solve: A is singular"
    );
    output
}
