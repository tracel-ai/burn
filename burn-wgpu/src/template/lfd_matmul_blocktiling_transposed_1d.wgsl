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
const T_N = {{t_n}}u;

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

    let thread_row = local_idx / B_N;
    let thread_col = local_idx % B_N;
    
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
    
    let computing_thread = thread_col % T_N == 0u && row < n_rows && col < n_cols; 

    let actual_T_N = min(B_N - thread_col, T_N);
    var results: array<{{ elem }}, T_N>; // useless if not computing_thread, but must be declared anyway...

    let sm_position = thread_row * B_N + thread_col;
    let lhs_row_rel = thread_row * K;
    let rhs_row_rel = thread_row * n_cols;

    for (var k: u32 = 0u; k < K; k += B_K) { 
        if thread_col >= K {
            shared_lhs[sm_position] = 0.0;
        } else {
            shared_lhs[sm_position] = lhs[offset_lhs + k + lhs_row_rel + thread_col];
        }   
        if thread_row >= K {
            shared_rhs[sm_position] = 0.0;
        } else {
            shared_rhs[sm_position] = rhs[offset_rhs + k * n_cols + rhs_row_rel + thread_col];        
        }
        

        workgroupBarrier();

        if computing_thread {
            for (var dot_index: u32 = 0u; dot_index < B_K; dot_index++) {
                let tmp_lhs = shared_lhs[dot_index + thread_row * B_N];

                for (var tile_index = 0u; tile_index < actual_T_N; tile_index++) {
                    let rhs_sm_position = thread_col + tile_index + dot_index * B_N;
                    results[tile_index] += shared_rhs[rhs_sm_position] * tmp_lhs;
                }
            }
        }
        
        workgroupBarrier();
    }

    if computing_thread {
        for (var tile_index = 0u; tile_index < actual_T_N; tile_index++) {
            if col + tile_index < n_cols { 
                output[offset_output + row * n_cols + col + tile_index] = results[tile_index];
            }
        }
    }
}
