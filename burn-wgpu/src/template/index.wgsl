@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim: u32 = info[0];
    var index_input: u32 = 0u;

    for (var i: u32 = 0u; i < dim; i++) {
        let stride_input = info[i + 1u];
        let stride_output = info[i + dim + 1u];
        let shape_output = info[i + 3u * dim + 1u];
        let start = info[i + 4u * dim + 1u];

        let num_block = global_id.x / stride_output % shape_output + start;

        index_input += num_block * stride_input;
    }

    output[global_id.x] = input[index_input];
}
