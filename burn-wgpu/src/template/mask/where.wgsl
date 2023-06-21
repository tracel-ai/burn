@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> value: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> mask: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(4)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = info[0];
    var index_input = 0u;
    var index_value = 0u;
    var index_mask = 0u;

    for (var i = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_value = info[i + dim];
        let stride_mask = info[i + 2u * dim];
        let stride_output = info[i + 3u * dim];

        let shape_input = info[i + 4u * dim];
        let shape_value = info[i + 5u * dim];
        let shape_mask = info[i + 6u * dim];

        index_input += global_id.x / stride_output % shape_input * stride_input;
        index_value += global_id.x / stride_output % shape_value * stride_value;
        index_mask += global_id.x / stride_output % shape_mask * stride_mask;
    }


    if mask[index_mask] != 0u {
        output[global_id.x] = value[index_value];
    } else {
        output[global_id.x] = input[index_input];
    }
}
