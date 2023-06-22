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

const B_M = {{block_size}}u;
const B_N = {{block_size}}u;
const B_K = {{block_k}}u;
const BLOCK_SIZE_X_BLOCK_K = {{block_size_x_block_k}}u;
const TILE_M = {{tile_m}}u;

var<workgroup> shared_lhs: array<{{ elem }}, BLOCK_SIZE_X_BLOCK_K>; // in fact: B_M * B_K
var<workgroup> shared_rhs: array<{{ elem }}, BLOCK_SIZE_X_BLOCK_K>; // in fact: B_K * B_N

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let skip_row = workgroup_id.x * B_M; 
    let skip_col = workgroup_id.y * B_N;

    // localIdx goes up to B_M * B_N / T_M

    // still 1 thread per output elem, but only for loading into sm
    let thread_row = local_idx / B_N;
    let thread_col = local_idx % B_N;

    // will get corresponding entries
    // but if B_K > B_N last values will be ignored
    // if B_K < B_N we will read outside block
    let rhs_thread_row = local_idx / B_N;
    let rhs_thread_col = local_idx % B_N;

    // will get offsetted entries
    // and if B_K > B_N last values will be ignored
    // if B_K < B_N we will read outside block
    let lhs_thread_row = local_idx / B_K;
    let lhs_thread_col = local_idx % B_K;
    
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
    var offset_lhs: u32 = skip_row * K;
    var offset_rhs: u32 = skip_col * 1u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 1u; b <= batch_dims; b++) {
        let stride_lhs = info[b];
        let stride_rhs = info[b + dim];
        let stride_output = info[b + 2u * dim];
        let shape_lhs = info[b + 3u * dim];
        let shape_rhs = info[b + 4u * dim];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }

    let lhs_column_stride = 1u;
    let lhs_row_stride = K;
    let rhs_column_stride = 1u;
    let rhs_row_stride = n_cols;
    let output_column_stride = 1u;
    let output_row_stride = n_cols;
    let lhs_block_column_stride = 1u;
    let lhs_block_row_stride = B_K;
    let rhs_block_column_stride = 1u;
    let rhs_block_row_stride = B_N;

    let south_of_lhs = row >= n_rows;
    let east_of_rhs = col >= n_cols;

    var results: array<{{ elem }}, TILE_M>;

    for (var k: u32 = 0u; k < K; k += B_K) {
        let lhs_sm_position = lhs_thread_row * lhs_block_row_stride + lhs_thread_col * lhs_block_column_stride;
        let lhs_block_ptr = k * lhs_column_stride;
        let lhs_row_rel = lhs_thread_row * lhs_row_stride;
        let lhs_col_rel = lhs_thread_col * lhs_column_stride;
        let east_of_lhs = lhs_block_ptr + lhs_col_rel >= K * lhs_column_stride;
        if east_of_lhs || south_of_lhs {
            shared_lhs[lhs_sm_position] = 0.0;
        } else {
            let lhs_position = offset_lhs + lhs_block_ptr + lhs_row_rel + lhs_col_rel;
            shared_lhs[lhs_sm_position] = lhs[lhs_position];
        }

        let rhs_sm_position = rhs_thread_row * rhs_block_row_stride + rhs_thread_col * rhs_block_column_stride;
        let rhs_block_ptr = k * rhs_row_stride;
        let rhs_row_rel = rhs_thread_row * rhs_row_stride;
        let rhs_col_rel = rhs_thread_col * rhs_column_stride;
        let south_of_rhs = rhs_block_ptr + rhs_row_rel >= K * rhs_row_stride;
        if east_of_rhs || south_of_rhs {
            shared_rhs[rhs_sm_position] = 0.0;
        } else {
            let rhs_position = offset_rhs + rhs_block_ptr + rhs_row_rel + rhs_col_rel;
            shared_rhs[rhs_sm_position] = rhs[rhs_position];
        }

        workgroupBarrier();

        if thread_row == 0u {
            for (var dot_index: u32 = 0u; dot_index < B_K; dot_index++) {
            let tmp_rhs = shared_rhs[dot_index * rhs_block_row_stride + rhs_thread_col * rhs_block_column_stride];

            for (var tile_index = 0u; tile_index < TILE_M; tile_index++) {
                let lhs_position = tile_index * lhs_block_row_stride + dot_index * lhs_block_column_stride;
                results[tile_index] += shared_lhs[lhs_position] * tmp_rhs;
            }
        }

        }
        
        workgroupBarrier();
    }


    if !south_of_lhs && !east_of_rhs {
        for (var tile_index = 0u; tile_index < TILE_M; tile_index++) {
            let row = skip_row + tile_index;
            let output_index = row * n_cols + col;
            output[offset_output + output_index] = results[tile_index];
        }
    }
}
