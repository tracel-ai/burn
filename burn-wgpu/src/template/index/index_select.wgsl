@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> indexes: array<{{ int }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rank = info[0];
    let dim = info[4u * rank + 1u];
    var index_input = 0u;

    for (var i = 1u; i <= rank; i++) {
        let stride_input = info[i];
        let stride_output = info[i + rank];
        let shape_input = info[i + 2u * rank];
        let shape_output = info[i + 3u * rank];

        let index = global_id.x / stride_output % shape_output;

        if i - 1u == dim {
            index_input += u32(indexes[index]) * stride_input;
        } else {
            index_input += index * stride_input;
        }
    }

    output[global_id.x] = input[index_input];
}

