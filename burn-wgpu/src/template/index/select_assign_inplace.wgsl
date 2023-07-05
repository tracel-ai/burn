@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> indices: array<{{ int }}>;

@group(0)
@binding(2)
var<storage, read> values: array<{{ elem }}>;

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
    let dim = info[5u * rank + 1u];

    let dim_stride_input = info[dim + 1u];
    let dim_stride_value = info[dim + rank + 1u];
    let dim_shape_value = info[dim + 3u * rank + 1u];

    var num_elems = 1u;
    var index_input_offset = 0u;
    var index_value_offset = 0u;

    var num_elem = 1u;

    for (var i = 1u; i <= rank; i++) {
        if i - 1u != dim {
            let stride_input = info[i];
            let stride_value = info[i + rank];
            let shape_input = info[i + 2u * rank];
            let shape_value = info[i + 3u * rank];
            let stride_tmp = info[i + 4u * rank];

            num_elem *= shape_input;
            index_input_offset += id / stride_tmp % shape_input * stride_input;
            index_value_offset += id / stride_tmp % shape_value * stride_value;
        }
    }

    if id >= num_elem {
        return;
    }

    for (var i = 0u; i < dim_shape_value; i++) {
        let index = u32(indices[i]);
        input[index_input_offset + index * dim_stride_input] += values[index_value_offset + i * dim_stride_value];
    }
}
