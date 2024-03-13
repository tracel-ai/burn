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
    var a_0_0: array<f32, 16>;

    let rank: u32 = info[0];
    let rank_2: u32 = rank * 2u;
    var l_0_0: u32;
    var l_0_1: u32;
    var l_0_2: u32;
    var l_0_3: u32;
    var l_0_4: u32;
    var l_0_5: u32;
    var l_0_6: u32;
    var l_0_7: u32;
    var l_0_8: u32;
    var l_0_9: u32;
    var l_0_10: u32;
    var l_0_11: u32;
    var l_0_12: u32;
    var l_0_13: u32;
    var l_0_14: u32;
    var l_0_15: u32;
    var l_0_16: u32;
    var l_0_17: u32;
    var l_0_18: u32;
    var l_0_19: u32;
    var l_0_20: vec4<f32>;
    var l_0_21: vec4<f32>;
    var l_0_22: u32;
    l_0_0 = rank - 1u;
    l_0_1 = info[(0u * rank_2) + rank + l_0_0 + 1u];
    l_0_2 = info[(1u * rank_2) + rank + l_0_0 + 1u];
    l_0_3 = info[(1u * rank_2) + rank + rank + 1u];
    l_0_4 = info[(0u * rank_2) + l_0_0 + 1u];
    l_0_5 = info[(0u * rank_2) + rank + 1u];
    l_0_6 = info[(1u * rank_2) + l_0_0 + 1u];
    l_0_7 = info[(1u * rank_2) + rank + 1u];
    l_0_8 = info[(2u * rank_2) + l_0_0 + 1u];
    l_0_9 = info[(2u * rank_2) + rank + 1u];
    l_0_10 = u32(workgroup_id.x);
    l_0_10 = l_0_10 * 64u;
    l_0_11 = u32(workgroup_id.y);
    l_0_11 = l_0_11 * 64u;
    l_0_12 = local_idx / 16u;
    l_0_12 = l_0_12 * 4u;
    l_0_13 = local_idx % 16u;
    l_0_13 = l_0_13 * 4u;
    l_0_14 = l_0_10 + l_0_12;
    l_0_15 = l_0_11 + l_0_13;
    l_0_16 = l_0_10 * l_0_4;
    l_0_17 = l_0_11 * l_0_7;
    l_0_18 = l_0_1 * l_0_3;
    l_0_18 = l_0_18 * global_id.z;
    l_0_19 = rank - 2u;
    l_0_16 = u32(0u);
    l_0_17 = u32(0u);

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_19; l_1_0++) {
        var l_1_1: u32;
        var l_1_2: u32;
        var l_1_3: u32;
        var l_1_4: u32;
        var l_1_5: u32;
        var l_1_6: u32;
        var l_1_7: u32;
        var l_1_8: u32;
        l_1_1 = info[(2u * rank_2) + l_1_0 + 1u];
        l_1_2 = l_0_18 * 1u;
        l_1_2 = l_1_2 / l_1_1;
        l_1_3 = info[(0u * rank_2) + l_1_0 + 1u];
        l_1_4 = info[(0u * rank_2) + rank + l_1_0 + 1u];
        l_1_5 = l_1_2 % l_1_4;
        l_1_5 = l_1_5 * l_1_3;
        l_0_16 = l_0_16 + l_1_5;
        l_1_6 = info[(1u * rank_2) + l_1_0 + 1u];
        l_1_7 = info[(1u * rank_2) + rank + l_1_0 + 1u];
        l_1_8 = l_1_2 % l_1_7;
        l_1_8 = l_1_8 * l_1_6;
        l_0_17 = l_0_17 + l_1_8;
    }
    l_0_16 = l_0_16 / 1u;
    l_0_17 = l_0_17 / 1u;
    l_0_22 = l_0_2 / 32u;

    for (var l_1_0: u32 = 0u; l_1_0 < l_0_22; l_1_0++) {
        var l_1_1: u32;
        l_1_1 = l_1_0 * 32u;

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            var l_2_1: u32;
            var l_2_2: bool;
            l_2_1 = l_0_13 + l_2_0;
            l_2_2 = l_2_1 < 32u;
            if l_2_2 {
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
                l_3_0 = l_0_12 / 4u;
                l_3_0 = l_3_0 * 32u;
                l_3_0 = l_3_0 + l_2_1;
                l_3_1 = l_1_1 + l_2_1;
                l_3_1 = l_3_1 * l_0_5;
                l_3_2 = l_0_12 * l_0_4;
                l_3_1 = l_3_1 + l_3_2;
                l_3_1 = l_3_1 + l_0_16;
                l_3_3 = l_3_1 + l_0_4;
                l_3_4 = l_3_3 + l_0_4;
                l_3_5 = l_3_4 + l_0_4;
                l_3_6 = f32(input_0_global[l_3_1]);
                l_3_7 = f32(input_0_global[l_3_3]);
                l_3_8 = f32(input_0_global[l_3_4]);
                l_3_9 = f32(input_0_global[l_3_5]);
                l_3_10 = vec4(l_3_6, l_3_7, l_3_8, l_3_9);
                shared_memory_0[l_3_0] = vec4<f32>(l_3_10);
            } else {
                break;
            }
        }

        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            var l_2_1: u32;
            var l_2_2: bool;
            l_2_1 = l_0_12 + l_2_0;
            l_2_2 = l_2_1 < 32u;
            if l_2_2 {
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
                l_3_0 = l_2_1 * 64u;
                l_3_0 = l_3_0 + l_0_13;
                l_3_0 = l_3_0 / 4u;
                l_3_1 = l_1_1 + l_2_1;
                l_3_1 = l_3_1 * l_0_6;
                l_3_2 = l_0_13 * l_0_7;
                l_3_1 = l_3_1 + l_3_2;
                l_3_1 = l_3_1 + l_0_17;
                l_3_3 = l_3_1 + l_0_7;
                l_3_4 = l_3_3 + l_0_7;
                l_3_5 = l_3_4 + l_0_7;
                l_3_6 = f32(input_1_global[l_3_1]);
                l_3_7 = f32(input_1_global[l_3_3]);
                l_3_8 = f32(input_1_global[l_3_4]);
                l_3_9 = f32(input_1_global[l_3_5]);
                l_3_10 = vec4(l_3_6, l_3_7, l_3_8, l_3_9);
                shared_memory_1[l_3_0] = vec4<f32>(l_3_10);
            } else {
                break;
            }
        }
        workgroupBarrier();

        for (var l_2_0: u32 = 0u; l_2_0 < 32u; l_2_0++) {
            var l_2_1: u32;
            var l_2_2: u32;
            l_2_1 = l_0_12 / 4u;
            l_2_1 = l_2_1 * 32u;
            l_2_1 = l_2_1 + l_2_0;
            l_0_20 = vec4<f32>(shared_memory_0[l_2_1]);
            l_2_2 = l_2_0 * 64u;
            l_2_2 = l_2_2 + l_0_13;
            l_2_2 = l_2_2 / 4u;
            l_0_21 = vec4<f32>(shared_memory_1[l_2_2]);

            for (var l_3_0: u32 = 0u; l_3_0 < 4u; l_3_0++) {
                for (var l_4_0: u32 = 0u; l_4_0 < 4u; l_4_0++) {
                    var l_4_1: f32;
                    var l_4_2: f32;
                    var l_4_3: f32;
                    var l_4_4: u32;
                    var l_4_5: f32;
                    var l_4_6: f32;
                    l_4_1 = f32(l_0_20[l_3_0]);
                    l_4_2 = f32(l_0_21[l_4_0]);
                    l_4_3 = l_4_1 * l_4_2;
                    l_4_4 = l_3_0 * 4u;
                    l_4_4 = l_4_4 + l_4_0;
                    l_4_5 = f32(a_0_0[l_4_4]);
                    l_4_6 = l_4_5 + l_4_3;
                    a_0_0[l_4_4] = f32(l_4_6);
                }
            }
        }
        workgroupBarrier();
    }

    for (var l_1_0: u32 = 0u; l_1_0 < 4u; l_1_0++) {
        for (var l_2_0: u32 = 0u; l_2_0 < 4u; l_2_0++) {
            var l_2_1: u32;
            var l_2_2: f32;
            var l_2_3: u32;
            var l_2_4: u32;
            var l_2_5: u32;
            l_2_1 = l_1_0 * 4u;
            l_2_1 = l_2_1 + l_2_0;
            l_2_2 = f32(a_0_0[l_2_1]);
            l_2_4 = l_0_14 + l_1_0;
            l_2_4 = l_2_4 * l_0_8;
            l_2_5 = l_0_15 + l_2_0;
            l_2_5 = l_2_5 * l_0_9;
            l_2_3 = l_2_4 + l_2_5;
            l_2_3 = l_2_3 + l_0_18;
            output_0_global[l_2_3] = f32(l_2_2);
        }
    }
}