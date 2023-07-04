@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> indexes: array<{{ int }}>;

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
    let rank = info[0];
    let dim = info[4u * rank + 1u];

    var index_input_offset = 0u;
    var index_values_offset = 0u;

    let stride_input_dim = info[dim + 1u];
    let stride_values_dim = info[dim + rank + 1u];

    let shape_input_dim = info[dim + 2u * rank + 1u];
    let shape_values_dim = info[dim + 3u * rank + 1u];
    let id_local = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
    let id_global = id_local * shape_input_dim;

    var num_elem = 1u;

    for (var i = 1u; i <= rank; i++) {
        if i - 1u != dim {
            let stride_input = info[i];
            let stride_values = info[i + rank];
            let shape_input = info[i + 2u * rank];
            let shape_values = info[i + 3u * rank];


            num_elem *= shape_input;
            index_input_offset += id_global / stride_input % shape_input * stride_input;
            index_values_offset += id_global / stride_values % shape_values * stride_values;
        }
    }

    if id_local >= num_elem {
        return;
    }

    for (var i = 0u; i < shape_values_dim; i++) {
        let index = u32(indexes[i]);
        input[index_input_offset + index * stride_input_dim] += values[index_values_offset + i * stride_values_dim];
    }
}
