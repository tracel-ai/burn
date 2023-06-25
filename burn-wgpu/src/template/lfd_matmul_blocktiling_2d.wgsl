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
const B_M_X_B_N = {{bm_x_bn}}u;
const T_M = {{t_m}}u;
const T_N = {{t_n}}u;
const T_M_X_T_N = {{tm_x_tn}}u;

var<workgroup> shared_lhs: array<{{ elem }}, B_M_X_B_N>; 
var<workgroup> shared_rhs: array<{{ elem }}, B_M_X_B_N>;

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

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols; 
    var offset_lhs: u32 = skip_row * K; 
    var offset_rhs: u32 = skip_col;

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
    
    let computing_thread = row < n_rows && col < n_cols; 

    // in case T_ does not divide B_ evenly
    let actual_T_M = min(B_M - thread_row, T_M);
    let actual_T_N = min(B_N - thread_col, T_N);

    var results: array<{{ elem }}, T_M_X_T_N>;
    var register_M: array<{{ elem }}, T_M>;
    var register_N: array<{{ elem }}, T_N>;

    for (var k = 0u; k < K; k += B_K) { 
        for (var i = 0u; i < actual_T_M; i++) {
            for (var j = 0u; j < actual_T_N; j++) {
                let sm_position = (thread_row + i) * B_N + thread_col + j;
                if thread_col + k + j >= K {
                    shared_lhs[sm_position] = 0.0;
                } else {
                    shared_lhs[sm_position] = lhs[offset_lhs + k + (thread_row + i) * K + thread_col + j];
                }
                if thread_row + k + i >= K {
                    shared_rhs[sm_position] = 0.0;
                } else {
                    shared_rhs[sm_position] = rhs[offset_rhs + (k + thread_row + i) * n_cols + thread_col + j];
                }   
            }
        }

        workgroupBarrier();

        if computing_thread {
            for (var dot_index = 0u; dot_index < B_K; dot_index++) {
                for (var tile_index = 0u; tile_index < actual_T_M; tile_index++) {
                    let lhs_sm_position = (thread_row + tile_index) * B_N + dot_index;
                    register_M[tile_index] = shared_lhs[lhs_sm_position];
                }
                for (var tile_index = 0u; tile_index < actual_T_N; tile_index++) {
                    let rhs_sm_position = thread_col + tile_index + dot_index * B_N;
                    register_N[tile_index] = shared_rhs[rhs_sm_position];
                }
                for (var res_idx_M = 0u; res_idx_M < actual_T_M; res_idx_M++) {
                    for (var res_idx_N = 0u; res_idx_N < actual_T_N; res_idx_N++) {
                        results[res_idx_M * actual_T_N + res_idx_N] += register_M[res_idx_M] * register_N[res_idx_N];
                    }
                }
            }
        }
        
        workgroupBarrier();
    }

    if computing_thread {
        for (var res_idx_M = 0u; res_idx_M < actual_T_M; res_idx_M++) {
            for (var res_idx_N = 0u; res_idx_N < actual_T_N; res_idx_N++) {
                if row + res_idx_M < n_rows && col + res_idx_N < n_cols { 
                    let res = results[res_idx_M * actual_T_N + res_idx_N];
                    output[offset_output + (row + res_idx_M) * n_cols + col + res_idx_N] = res;
                }
            }
        }
    }
}
