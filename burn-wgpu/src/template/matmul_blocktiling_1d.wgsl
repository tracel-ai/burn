@group(0)
@binding(0)
var<storage, read> lhs: array<elem>;

@group(0)
@binding(1)
var<storage, read> rhs: array<elem>;

@group(0)
@binding(2)
var<storage, read_write> output: array<elem>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

const BLOCK_M = {{BLOCK_M}}u;
const BLOCK_N = {{BLOCK_N}}u;
const BLOCK_K = {{BLOCK_K}}u;
const BLOCK_MK = {{BLOCK_MK}}u;
const BLOCK_KN = {{BLOCK_KN}}u;

const TILE_M = {{TILE_M}}u;

var<workgroup> shared_lhs: array<elem, BLOCK_MK>;
var<workgroup> shared_rhs: array<elem, BLOCK_KN>;

@compute
@workgroup_size({{BLOCK_M}}, {{BLOCK_N}}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Indexes
    let thread_row = local_idx / BLOCK_N;
    let thread_col = local_idx % BLOCK_N;
    let batch = global_id.z;


    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u];
    let n_cols = info[6u * dim];
    let K = info[5u * dim - 1u];


    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = workgroup_id.x * BLOCK_M * K;
    var offset_rhs: u32 = workgroup_id.y * BLOCK_N;

    let batch_dims = dim - 2u;
    for (var b: u32 = 0u; b < batch_dims; b++) {
        let stride_lhs = info[b + 1u];
        let stride_rhs = info[b + 1u * dim + 1u];
        let stride_output = info[b + 2u * dim + 1u];
        let shape_lhs = info[b + 3u * dim + 1u];
        let shape_rhs = info[b + 4u * dim + 1u];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }

    let shared_row_lhs = local_idx / BLOCK_K;
    let shared_col_lhs = local_idx % BLOCK_K;
    let shared_row_rhs = local_idx / BLOCK_N;
    let shared_col_rhs = local_idx % BLOCK_N;

    var results: array<elem, TILE_M>;

    for (var k: u32 = 0u; k < K; k += BLOCK_K) {
        if shared_row_lhs < n_rows && shared_col_lhs + k < K {
            let index_lhs = shared_row_lhs * K + shared_col_lhs + k;
            shared_lhs[shared_row_lhs * BLOCK_K + shared_col_lhs] = lhs[index_lhs + offset_lhs];
        }

        if shared_row_rhs + k < K && shared_col_lhs < n_cols {
            let index_rhs = (shared_row_rhs + k) * n_cols + shared_col_rhs;
            shared_rhs[shared_row_rhs * BLOCK_N + shared_col_rhs] = rhs[index_rhs + offset_rhs];
        }

        workgroupBarrier();

        for (var bk: u32 = 0u; bk < BLOCK_K; bk++) {
            let tmp_rhs = shared_rhs[bk * BLOCK_N + thread_col];

            for (var tile_index = 0u; tile_index < TILE_M; tile_index++) {
                results[tile_index] += shared_lhs[(thread_row * TILE_M + tile_index) * BLOCK_K + bk] * tmp_rhs;
            }
        }

        workgroupBarrier();
    }


    for (var tile_index = 0u; tile_index < TILE_M; tile_index++) {
        let tile_row = thread_row * TILE_M + tile_index;
        let row = workgroup_id.x * BLOCK_M + tile_row;
        let col = workgroup_id.y * BLOCK_N + thread_col;

        if row < n_rows && col < n_cols {
            let output_index = row * n_cols + col;
            output[offset_output + output_index] = results[tile_index];
        }
    }
}
