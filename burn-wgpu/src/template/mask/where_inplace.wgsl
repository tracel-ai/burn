@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> value: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> mask: array<u32>;

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
    let dim = info[0];
    let reverse = info[6u * dim + 1u];

    var index_input = 0u;
    var index_value = 0u;
    var index_mask = 0u;

    for (var i = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_value = info[i + dim];
        let stride_mask = info[i + 2u * dim];

        let shape_input = info[i + 3u * dim];
        let shape_value = info[i + 4u * dim];
        let shape_mask = info[i + 5u * dim];

        index_input += id / stride_input % shape_input * stride_input;
        index_value += id / stride_input % shape_value * stride_value;
        index_mask += id / stride_input % shape_mask * stride_mask;
    }

    var condition = mask[index_mask] != 0u;

    if reverse == 1u {
        condition = !condition;
    }

    if condition {
        input[index_input] = value[index_value];
    } else {
        input[index_input] = input[index_input];
    }
}
