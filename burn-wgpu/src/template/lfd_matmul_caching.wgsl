@group(0)
@binding(0)
var<storage, read> lhs: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> rhs: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

const BLOCK_SIZE = {{ block_size }}u;
const BLOCK_SIZE_2X = {{ block_size_2x }}u;

// BLOCK_SIZE_2X is in fact squared. so it's big as a block
var<workgroup> shared_lhs: array<{{ elem }}, BLOCK_SIZE_2X>; // likely shared memory for block in lhs
var<workgroup> shared_rhs: array<{{ elem }}, BLOCK_SIZE_2X>; // likely shared memory for block in lhs

@compute
@workgroup_size({{ block_size }}, {{ block_size }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // part to skip because not done by this block
    let skip_row = workgroup_id.x * BLOCK_SIZE;
    let skip_col = workgroup_id.y * BLOCK_SIZE;

    // what row/col this thread is working with, relative to the block
    let thread_row = local_idx / BLOCK_SIZE;
    let thread_col = local_idx % BLOCK_SIZE;

    // what row/col this thread is working with, in absolute
    let row = skip_row + thread_row;
    let col = skip_col + thread_col;

    let batch = global_id.z;

    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u];
    let n_cols = info[6u * dim];
    let K = info[5u * dim - 1u];

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    // pointer towards block row in lhs. skip_row gives the row, but K gives number of elements per row, i.e. stride
    var offset_lhs: u32 = skip_row * K;
    // pointer towards block col in rhs. K is not needed, the stride is 1 because they're columns
    var offset_rhs: u32 = skip_col * 1u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 1u; b <= batch_dims; b++) {
        let stride_lhs = info[b];
        let stride_rhs = info[b + dim];
        let stride_output = info[b + 2u * dim ];
        let shape_lhs = info[b + 3u * dim];
        let shape_rhs = info[b + 4u * dim];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }

    // in this version, one thread has one sum to compute, it is responsible for one number
    var sum = 0.0;

    let lhs_column_stride = 1u;
    let lhs_row_stride = K;
    let rhs_column_stride = 1u;
    let rhs_row_stride = n_cols;
    let output_column_stride = 1u;
    let output_row_stride = n_cols;
    let block_column_stride = 1u;
    let block_row_stride = BLOCK_SIZE;

    let south_of_lhs = row >= n_rows;
    let east_of_rhs = col >= n_cols;

    for (var block_index: u32 = 0u; block_index < K; block_index += BLOCK_SIZE) {
        let sm_position = thread_row * block_row_stride + thread_col * block_column_stride;

        let lhs_block_ptr = block_index * lhs_column_stride;
        let lhs_row_rel = thread_row * lhs_row_stride;
        let lhs_col_rel = thread_col * lhs_column_stride;
        let east_of_lhs = lhs_block_ptr + lhs_col_rel >= K * lhs_column_stride;
        if east_of_lhs || south_of_lhs {
            shared_lhs[sm_position] = 0.0;
        } else {
            let lhs_position = offset_lhs + lhs_block_ptr + lhs_row_rel + lhs_col_rel;
            shared_lhs[sm_position] = lhs[lhs_position];
            
        }

        let rhs_block_ptr = block_index * rhs_row_stride;
        let rhs_row_rel = thread_row * rhs_row_stride;
        let rhs_col_rel = thread_col * rhs_column_stride;
        let south_of_rhs = rhs_block_ptr + rhs_row_rel >= K * rhs_row_stride;
        if east_of_rhs || south_of_rhs {
            shared_rhs[sm_position] = 0.0;
        } else {
            let rhs_position = offset_rhs + rhs_block_ptr + rhs_row_rel + rhs_col_rel;
            shared_rhs[sm_position] = rhs[rhs_position];
        }

        workgroupBarrier();

        if !south_of_lhs && !east_of_rhs {
            for (var dot_index: u32 = 0u; dot_index < BLOCK_SIZE; dot_index++) {
                let lhs_elem = shared_lhs[thread_row * block_row_stride + dot_index * block_column_stride];
                let rhs_elem = shared_rhs[dot_index * block_row_stride + thread_col * block_column_stride];
                sum += lhs_elem * rhs_elem;
            }
        }

        workgroupBarrier();
    }

    if !south_of_lhs && !east_of_rhs {
        let output_index = row * n_cols + col;
        output[offset_output + output_index] = sum;
    }
}
