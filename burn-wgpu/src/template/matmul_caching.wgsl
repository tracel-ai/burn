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

const BLOCK_SIZE = {{ block_size }}u;
const BLOCK_SIZE_2X = {{ block_size_2x }}u;

var<workgroup> shared_lhs: array<elem, BLOCK_SIZE_2X>;
var<workgroup> shared_rhs: array<elem, BLOCK_SIZE_2X>;

@compute
@workgroup_size({{ block_size }}, {{ block_size }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Indexes
    let thread_row = local_idx / BLOCK_SIZE;
    let thread_col = local_idx % BLOCK_SIZE;
    let batch = global_id.z;
    let row = workgroup_id.x * BLOCK_SIZE + thread_row;
    let col = workgroup_id.y * BLOCK_SIZE + thread_col;

    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u];
    let n_cols = info[6u * dim];
    let K = info[5u * dim - 1u];

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = workgroup_id.x * BLOCK_SIZE * K;
    var offset_rhs: u32 = workgroup_id.y * BLOCK_SIZE;
    let num_elems_lhs = n_rows * K;
    let num_elems_rhs = K * n_cols;

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

    var sum = 0.0;

    for (var block_index: u32 = 0u; block_index < K; block_index += BLOCK_SIZE) {
        let index_lhs = thread_row * K + thread_col + block_index;
        let index_rhs = thread_row * n_cols + thread_col + block_index * n_cols;

        if shared_row_lhs < n_rows && shared_col_lhs + k < K {
            shared_lhs[thread_row * BLOCK_SIZE + thread_col] = lhs[index_lhs + offset_lhs];
        }
        if index_rhs < num_elems_rhs {
            shared_rhs[thread_row * BLOCK_SIZE + thread_col] = rhs[index_rhs + offset_rhs];
        }

        workgroupBarrier();

        for (var dot_index: u32 = 0u; dot_index < BLOCK_SIZE; dot_index++) {
            sum += shared_lhs[thread_row * BLOCK_SIZE + dot_index] * shared_rhs[dot_index * BLOCK_SIZE + thread_col];
        }

        workgroupBarrier();
    }

    if row >= n_rows || col >= n_cols {
        return;
    }
    let output_index = row * n_rows + col;
    output[offset_output + output_index] = sum;
}
