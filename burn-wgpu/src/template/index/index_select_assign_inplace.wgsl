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

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rank = info[0];
    let dim = info[4u * rank + 1u];

    var index_input_offset = 0u;
    var index_values_offset = 0u;

    var stride_input_dim = 0u;
    var stride_values_dim = 0u;

    var shape_input_dim = 0u;
    var shape_values_dim = 0u;

    var num_elem = 1u;

    for (var i = 1u; i <= rank; i++) {
        let stride_input = info[i];
        let stride_values = info[i + rank];
        let shape_input = info[i + 2u * rank];
        let shape_values = info[i + 3u * rank];

        if i - 1u != dim {
            index_input_offset += global_id.x / stride_input % shape_input * stride_input;
            index_values_offset += global_id.x / stride_values % shape_values * stride_values;
            num_elem += shape_input;
        } else {
            shape_input_dim = shape_input;
            shape_values_dim = shape_values;

            stride_input_dim = stride_input;
            stride_values_dim = stride_values;
        }
    }

    if global_id.x > num_elem {
        return;
    }

    for (var i = 0u; i < shape_values_dim; i++) {
        let index = u32(indexes[i]);
        input[index_input_offset + index * stride_input_dim] += values[index_values_offset + i * stride_values_dim];
    }
}

