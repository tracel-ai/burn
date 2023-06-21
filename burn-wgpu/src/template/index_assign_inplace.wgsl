@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> value: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim: u32 = info[0];
    var index_input: u32 = 0u;
    var index_value: u32 = 0u;

    for (var i: u32 = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_value = info[i + dim];
        let shape_input = info[i + 2u * dim];
        let shape_value = info[i + 3u * dim];
        let start = info[i + 4u * dim];

        let num_block = global_id.x / stride_value % shape_value;

        index_input += (num_block + start) * stride_input;
        index_value += num_block * stride_value;
    }

    input[index_input] = value[index_value];
}
