@group(0)
@binding(0)
var<storage, read> input_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read> input_1_global: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

var<workgroup> shared_memory_0: array<vec4<f32>, 512>;

var<workgroup> shared_memory_1: array<vec4<f32>, 512>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    var results: array<f32, 16>;

    let rank: u32 = info[0];
    let rank_2: u32 = rank * 2u;
    var l_0_0: u32;
    var M: u32;
    var K: u32;
    var N: u32;
    var lhs_stride_row: u32;
    var lhs_stride_col: u32;
    var rhs_stride_row: u32;
    var rhs_stride_col: u32;
    var out_stride_row: u32;
    var out_stride_col: u32;
    var skip_row: u32;
    var skip_col: u32;
    var thread_row: u32;
    var thread_col: u32;
    var row: u32;
    var col: u32;
    var offset_lhs: u32;
    var offset_rhs: u32;
    var offset_output: u32;
    var l_0_19: u32;
    var register_m: vec4<f32>;
    var register_n: vec4<f32>;
    var n_loops: u32;
    l_0_0 = rank - 1u;
    M = info[(0u * rank_2) + rank + l_0_0 + 1u];
    K = info[(1u * rank_2) + rank + l_0_0 + 1u];
    N = info[(1u * rank_2) + rank + rank + 1u];
    lhs_stride_row = info[(0u * rank_2) + l_0_0 + 1u];
    lhs_stride_col = info[(0u * rank_2) + rank + 1u];
    rhs_stride_row = info[(1u * rank_2) + l_0_0 + 1u];
    rhs_stride_col = info[(1u * rank_2) + rank + 1u];
    out_stride_row = info[(2u * rank_2) + l_0_0 + 1u];
    out_stride_col = info[(2u * rank_2) + rank + 1u];
    skip_row = u32(workgroup_id.x) * 64u;
    skip_col = u32(workgroup_id.y) * 64u;
    thread_row = (local_idx / 16u)*4u;
    thread_col = (local_idx % 16u)*4u;
    row = skip_row + thread_row;
    col = skip_col + thread_col;
    offset_lhs = skip_row * lhs_stride_row;
    offset_rhs = skip_col * rhs_stride_col;
    offset_output = M * N;
    offset_output = offset_output * global_id.z;
    l_0_19 = rank - 2u;

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_19; l_1_0++) {
        var l_1_1: u32;
        var l_1_2: u32;
        var l_1_3: u32;
        var l_1_4: u32;
        var l_1_5: u32;
        var l_1_6: u32;
        var l_1_7: u32;
        var l_1_8: u32;
        l_1_1 = info[(0u * rank_2) + l_1_0 + 1u];
        l_1_2 = info[(1u * rank_2) + l_1_0 + 1u];
        l_1_3 = info[(2u * rank_2) + l_1_0 + 1u];
        l_1_4 = info[(0u * rank_2) + rank + l_1_0 + 1u];
        l_1_5 = info[(1u * rank_2) + rank + l_1_0 + 1u];
        l_1_6 = offset_output / l_1_3;
        l_1_7 = l_1_6 % l_1_4;
        l_1_7 = l_1_7 * l_1_1;
        offset_lhs = offset_lhs + l_1_7;
        l_1_8 = l_1_6 % l_1_5;
        l_1_8 = l_1_8 * l_1_2;
        offset_rhs = offset_rhs + l_1_8;
    }
    n_loops = K / 32u;

    for (var i: u32 = 0u; i < n_loops; i++) {
        var k: u32;
        k = i * 32u;

        for (var j: u32 = 0u; j < 4u; j++) {
            var current_col: u32;
            var l_2_2: bool;
            current_col = thread_col + j;
            if current_col < 32u{
                var l_3_0: u32;
                var l_3_1: u32;
                var l_3_2: u32;
                var l_3_3: u32;
                var l_3_4: u32;
                var l_3_5: u32;
                var l_3_6: f32;
                var l_3_7: f32;
                var l_3_8: f32;
                var l_3_9: f32;
                var l_3_10: vec4<f32>;
                l_3_0 = (thread_row / 4u) * 32u + current_col;
                lhs_position0 = offset_lhs + (k + current_col) * lhs_stride_col + thread_row * lhs_stride_row;
                lhs_position1 = lhs_position0 + lhs_stride_row;
                lhs_position2 = lhs_position1 + lhs_stride_row;
                lhs_position3 = lhs_position2 + lhs_stride_row;
                l_3_6 = f32(input_0_global[lhs_position0]);
                l_3_7 = f32(input_0_global[lhs_position1]);
                l_3_8 = f32(input_0_global[lhs_position2]);
                l_3_9 = f32(input_0_global[lhs_position3]);
                l_3_10 = vec4(l_3_6, l_3_7, l_3_8, l_3_9);
                shared_memory_0[l_3_0] = vec4<f32>(l_3_10);
            } else {
                break;
            }
        }

        for (var i: u32 = 0u; i < 4u; i++) {
            var current_row: u32;
            current_row = thread_row + i;
            if current_row < 32u {
                var l_3_0: u32;
                var rhs_position0: u32;
                var l_3_2: u32;
                var l_3_3: u32;
                var l_3_4: u32;
                var l_3_5: u32;
                var l_3_6: f32;
                var l_3_7: f32;
                var l_3_8: f32;
                var l_3_9: f32;
                var l_3_10: vec4<f32>;
                rhs_sm_position = (current_row * 64u + thread_col) / 4u;
                rhs_position0 = offset_rhs + (k + current_row) * rhs_stride_row + thread_col * rhs_stride_col;
                rhs_position1 = rhs_position0 + rhs_stride_col;
                rhs_position2 = rhs_position1 + rhs_stride_col;
                rhs_position3 = rhs_position2 + rhs_stride_col;
                l_3_6 = f32(input_1_global[rhs_position0]);
                l_3_7 = f32(input_1_global[rhs_position1]);
                l_3_8 = f32(input_1_global[rhs_position2]);
                l_3_9 = f32(input_1_global[rhs_position3]);
                l_3_10 = vec4(l_3_6, l_3_7, l_3_8, l_3_9);
                shared_memory_1[rhs_sm_position] = vec4<f32>(l_3_10);
            } else {
                break;
            }
        }
        workgroupBarrier();

        for (var dot_index: u32 = 0u; dot_index < 32u; dot_index++) {
            var lhs_sm_position: u32;
            var rhs_sm_position: u32;
            lhs_sm_position = (thread_row / 4u)*32u+dot_index;
            register_m = vec4<f32>(shared_memory_0[lhs_sm_position]);
            rhs_sm_position = (dot_index * 64u + thread_col) / 4u;
            register_n = vec4<f32>(shared_memory_1[rhs_sm_position]);

            for (var res_idx_m: u32 = 0u; res_idx_m < 4u; res_idx_m++) {
                for (var res_idx_n: u32 = 0u; res_idx_n < 4u; res_idx_n++) {
                    var left: f32;
                    var right: f32;
                    var multiplied: f32;
                    var results_position: u32;
                    var old: f32;
                    left = f32(register_m[res_idx_m]);
                    right = f32(register_n[res_idx_n]);
                    multiplied = left * right;
                    results_position = res_idx_m * 4u + res_idx_n;
                    old = f32(results[results_position]);
                    results[results_position] = f32(old + multiplied);
                }
            }
        }
        workgroupBarrier();
    }

    for (var res_idx_m: u32 = 0u; res_idx_m < 4u; res_idx_m++) {
        for (var res_idx_n: u32 = 0u; res_idx_n < 4u; res_idx_n++) {
            var result_position: u32;
            var result: f32;
            var output_position: u32;
            result_position = res_idx_m * 4u + res_idx_n;
            result = f32(results[result_position]);
            output_position = (row + res_idx_m) * out_stride_row + (col + res_idx_n) * out_stride_col + offset_output;
            output_0_global[output_position] = f32(result);
        }
    }
}