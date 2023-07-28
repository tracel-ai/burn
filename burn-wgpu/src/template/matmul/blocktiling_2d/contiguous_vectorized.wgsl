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

const B_M = {{b_m}}u;
const B_N = {{b_n}}u;
const B_K = {{b_k}}u;
const B_M_X_B_K = {{bm_x_bk}}u;
const B_K_X_B_N = {{bk_x_bn}}u;
const T_M = {{t_m}}u;
const T_N = {{t_n}}u;
const T_M_X_T_N = {{tm_x_tn}}u;

var<workgroup> shared_lhs: array<{{ elem }}, B_M_X_B_K>; 
var<workgroup> shared_rhs: array<{{ elem }}, B_K_X_B_N>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let skip_row = workgroup_id.x * B_M;
    let skip_col = workgroup_id.y * B_N;

    let n_thread_per_row = ((B_N - 1u) / T_N) + 1u;
    let n_thread_per_col = ((B_M - 1u) / T_M) + 1u;
    let n_threads = n_thread_per_row * n_thread_per_col;

    let thread_row = (local_idx / n_thread_per_row) * T_M;
    let thread_col = (local_idx % n_thread_per_row) * T_N;
    
    let row = skip_row + thread_row;
    let col = skip_col + thread_col;

    let batch = global_id.z;

    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u]; 
    let n_cols = info[6u * dim]; 
    let K = info[5u * dim - 1u];

    // Row / col strides
    let lhs_stride_row = info[dim - 1u];
    let lhs_stride_col = info[dim];
    let rhs_stride_row = info[2u * dim - 1u];
    let rhs_stride_col = info[2u * dim];
    let out_stride_row = info [3u * dim - 1u];
    let out_stride_col = info [3u * dim];

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols; 
    var offset_lhs: u32 = skip_row * lhs_stride_row; 
    var offset_rhs: u32 = skip_col * rhs_stride_col;

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
    
    var results: array<{{ elem }}, T_M_X_T_N>;
    var register_M: array<{{ elem }}, T_M>;
    var register_N: array<{{ elem }}, T_N>;

    for (var k = 0u; k < K; k += B_K) {
        for (var load_index = 0u; load_index < T_M_X_T_N; load_index ++) {
            let lhs_sm_position = local_idx + load_index * n_threads;
            let block_row = lhs_sm_position % B_M;
            let block_col = lhs_sm_position / B_M;
            let lhs_position = offset_lhs + block_row * lhs_stride_row + (k + block_col) * lhs_stride_col;

            if block_col < B_K {
                shared_lhs[lhs_sm_position] = lhs[lhs_position];
            } else {
                // Bugfix
                // On Mac OS, when blocks are too large, output will be not be written to, 
                // unless we add this line in which we write in the output. 
                // This value will be overwritten, but allows the output to be writable at the end. 
                output[offset_output + row * out_stride_row + col * out_stride_col] = 0.0;
            }
        }

        for (var load_index = 0u; load_index < T_M_X_T_N; load_index ++) {
            let rhs_sm_position = local_idx + load_index * n_threads;
            let block_row = rhs_sm_position / B_N;
            let block_col = rhs_sm_position % B_N;
            let rhs_position = offset_rhs + (k + block_row) * rhs_stride_row + block_col * rhs_stride_col;

            if block_row < B_K {
                shared_rhs[rhs_sm_position] = rhs[rhs_position];
            }
        } 

        workgroupBarrier();

        // Compute intermediate results
        // Results are cumulated in results array and updated at each block
        // Outer loop indicates which subcolumns/subrows to read from shared memories
        for (var dot_index = 0u; dot_index < B_K; dot_index++) {
            // Load a subcolumn of values from lhs
            for (var tile_index = 0u; tile_index < T_M; tile_index++) {
                let lhs_sm_position = thread_row + tile_index + dot_index * B_M;
                register_M[tile_index] = shared_lhs[lhs_sm_position];
            }
            // Load a subrow of values from rhs
            for (var tile_index = 0u; tile_index < T_N; tile_index++) {
                let rhs_sm_position = thread_col + tile_index + dot_index * B_N;
                register_N[tile_index] = shared_rhs[rhs_sm_position];
            }
            // Multiply subcolumn and subrow and store results
            for (var res_idx_M = 0u; res_idx_M < T_M; res_idx_M++) {
                for (var res_idx_N = 0u; res_idx_N < T_N; res_idx_N++) {
                    results[res_idx_M * T_N + res_idx_N] += register_M[res_idx_M] * register_N[res_idx_N];
                }
            }
        }
        
        workgroupBarrier();
    }

    // Write output matrix
    // Each thread is responsible of writing T_M x T_N results
    for (var res_idx_M = 0u; res_idx_M < T_M; res_idx_M++) {
        for (var res_idx_N = 0u; res_idx_N < T_N; res_idx_N++) {
            let result_position = res_idx_M * T_N + res_idx_N;
            let output_position = offset_output + (row + res_idx_M) * out_stride_row + (col + res_idx_N) * out_stride_col;
            output[output_position] = results[result_position];
        }
    }
}
