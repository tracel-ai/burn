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
const B_M_X_B_K_4 = {{bm_x_bk_4}}u;
const B_K_X_B_N_4 = {{bk_x_bn_4}}u;

const T_M = 4u;
const T_N = 4u;
const T_M_X_T_N = 16u;

var<workgroup> shared_lhs: array<vec4<{{ elem }}>, B_M_X_B_K_4>; 
var<workgroup> shared_rhs: array<vec4<{{ elem }}>, B_K_X_B_N_4>; 

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
    
    // Position of the first element of the thread, relative to the block
    let thread_row = (local_idx / n_thread_per_row) * T_M;
    let thread_col = (local_idx % n_thread_per_row) * T_N;
    
    // Position of the first element of the thread, in absolute (in one batch)
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
    
    // Registers used in the compute pass
    var results: array<{{ elem }}, T_M_X_T_N>;
    var register_M: vec4<{{ elem }}>;
    var register_N: vec4<{{ elem }}>;
    
    // How close is the thread to the end of the matrix. 
    // If < 4 then it is an edge case
    let remain_row_lhs = n_rows - row;
    let remain_col_rhs = n_cols - col;

    for (var k = 0u; k < K; k += B_K) {

        // LHS LOAD PASS 

        // For the 4 vec4 columns of this thread
        for (var j = 0u; j < 4u; j++) {

            // The precise 
            let current_col = thread_col + j;
            
            // Position of the column vec4 in shared memory
            let lhs_sm_position = (thread_row/4u) * B_K + current_col;

            // To avoid overwriting following row in share memory
            if current_col < B_K { 
                // To pad with zeros if outside lhs 
                if current_col + k < K && remain_row_lhs >= 1u {
                    let lhs_position0 = offset_lhs + (k + current_col) * lhs_stride_col + thread_row * lhs_stride_row;
                    let lhs_position1 = lhs_position0 + lhs_stride_row;
                    let lhs_position2 = lhs_position1 + lhs_stride_row;
                    let lhs_position3 = lhs_position2 + lhs_stride_row;

                    if remain_row_lhs >= 4u {
                        shared_lhs[lhs_sm_position] = vec4(
                            lhs[lhs_position0],
                            lhs[lhs_position1],
                            lhs[lhs_position2],
                            lhs[lhs_position3],
                        );
                    } else if remain_row_lhs == 3u {
                        shared_lhs[lhs_sm_position] = vec4(
                            lhs[lhs_position0],
                            lhs[lhs_position1],
                            lhs[lhs_position2],
                            0.
                        ); 
                    } else if remain_row_lhs == 2u {
                        shared_lhs[lhs_sm_position] = vec4(
                            lhs[lhs_position0],
                            lhs[lhs_position1],
                            0.,
                            0.
                        ); 
                    } else if remain_row_lhs == 1u {
                        shared_lhs[lhs_sm_position] = vec4(
                            lhs[lhs_position0],
                            0.,
                            0.,
                            0.
                        );  
                    } 
                } else {
                    shared_lhs[lhs_sm_position] = vec4(0.,0.,0.,0.);
                }
            }
        }

        // RHS LOAD PASS

        for (var i = 0u; i < 4u; i++) {
            let current_row = thread_row + i;
            
            let rhs_sm_position = (current_row * B_N + thread_col) / 4u;
            
            if current_row < B_K {
                if current_row + k < K && remain_col_rhs >= 1u {

                    let rhs_position0 = offset_rhs + (k + current_row) * rhs_stride_row + thread_col * rhs_stride_col;
                    let rhs_position1 = rhs_position0 + rhs_stride_col;
                    let rhs_position2 = rhs_position1 + rhs_stride_col;
                    let rhs_position3 = rhs_position2 + rhs_stride_col;

                    if remain_col_rhs >= 4u {
                        shared_rhs[rhs_sm_position] = vec4(
                            rhs[rhs_position0],
                            rhs[rhs_position1],
                            rhs[rhs_position2],
                            rhs[rhs_position3],
                        );
                    } else if remain_col_rhs == 3u {
                        shared_rhs[rhs_sm_position] = vec4(
                            rhs[rhs_position0],
                            rhs[rhs_position1],
                            rhs[rhs_position2],
                            0.
                        ); 
                    } else if remain_col_rhs == 2u {
                        shared_rhs[rhs_sm_position] = vec4(
                            rhs[rhs_position0],
                            rhs[rhs_position1],
                            0.,
                            0.
                        ); 
                    } else if remain_col_rhs == 1u {
                        shared_rhs[rhs_sm_position] = vec4(
                            rhs[rhs_position0],
                            0.,
                            0.,
                            0.
                        );  
                    }
                } else {
                    shared_rhs[rhs_sm_position] = vec4(0.,0.,0.,0.);
                }
            }
        } 

        workgroupBarrier();

        // COMPUTE PASS

        // Compute intermediate results
        // Results are cumulated in results array and updated at each block
        // Outer loop indicates which subcolumns/subrows to read from shared memories
        for (var dot_index = 0u; dot_index < B_K; dot_index++) {
            
            // Load a subcolumn of values from lhs
            let lhs_sm_position = (thread_row/4u) * B_K + dot_index;
            register_M = shared_lhs[lhs_sm_position];
            
            // Load a subrow of values from rhs
            let rhs_sm_position = (dot_index * B_N + thread_col) / 4u;
            register_N = shared_rhs[rhs_sm_position];

            // Multiply subcolumn and subrow and store results
            for (var res_idx_M = 0u; res_idx_M < T_M; res_idx_M++) {
                for (var res_idx_N = 0u; res_idx_N < T_N; res_idx_N++) {
                    results[res_idx_M * T_N + res_idx_N] += register_M[res_idx_M] * register_N[res_idx_N];
                }
            }
        }
        
        workgroupBarrier();
    }

    // OUTPUT PASS

    // Write output matrix
    // Each thread is responsible of writing T_M x T_N results
    for (var res_idx_M = 0u; res_idx_M < T_M; res_idx_M++) {
        for (var res_idx_N = 0u; res_idx_N < T_N; res_idx_N++) {
            let row_index = row + res_idx_M;
            let col_index = col + res_idx_N;
            if row_index < n_rows && col_index < n_cols {
                let result_position = res_idx_M * T_N + res_idx_N;
                let output_position = offset_output + row_index * out_stride_row + col_index * out_stride_col;
                output[output_position] = results[result_position];
            }
        }
    }
}
