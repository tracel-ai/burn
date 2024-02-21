@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ int }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
    let dim: u32 = info[0];
    let dim_reduce = info[4u * dim + 1u];
    var index_offset: u32 = 0u;
    var stride_dim: u32 = 0u;
    var shape_dim: u32 = 0u;

    for (var i: u32 = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_output = info[i + dim];
        let shape_output = info[i + 3u * dim];

        let num_block = id / stride_output % shape_output;

        if i - 1u != dim_reduce {
            index_offset += num_block * stride_input;
        } else {
            let shape_input = info[i + 2u * dim];
            index_offset += num_block;
            stride_dim = stride_input;
            shape_dim = shape_input;
        }
    }

    var current_value = {{ elem }}({{ initial }});
    var index = {{ int }}(0);

    for (var i = 0u; i < shape_dim; i++) {
        let index_input = i * stride_dim;
        let value = input[index_input + index_offset];

        if (value {{ cmp }} current_value) {
            current_value = value;
            index = {{ int }}(i);

        }
    }

    output[id] = index;
}
