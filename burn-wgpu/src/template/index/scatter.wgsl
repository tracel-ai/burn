@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> indices: array<{{ int }}>;

@group(0)
@binding(2)
var<storage, read> value: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
    let rank = info[0];
    let dim = info[3u * rank + 1u];

    let shape = info[dim + rank + 1u];
    let stride = info[dim + 1u];

    var num_elems = 1u;
    var index_offset = 0u;

    for (var i = 1u; i <= rank; i++) {
        if i - 1u != dim {
            let stride_input = info[i];
            let shape_input = info[i + rank];
            let stride_tmp = info[i + 2u * rank];

            num_elems *= shape_input;
            index_offset += id / stride_tmp % shape_input * stride_input;
        }
    }

    if id >= num_elems {
        return;
    }

    for (var i = 0u; i < shape; i++) {
        let index = i * stride + index_offset;
        input[index_offset + stride * u32(indices[index])] += value[index];
    }
}
