@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> value: {{ elem }};


@group(0)
@binding(2)
var<storage, read> mask: array<u32>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = info[0];
    var index_input = 0u;
    var index_mask = 0u;

    for (var i = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_mask = info[i + dim];
        let shape_input = info[i + 2u * dim];
        let shape_mask = info[i + 3u * dim];

        index_input += global_id.x / stride_input % shape_input * stride_input;
        index_mask += global_id.x / stride_input % shape_mask * stride_mask;
    }


    if mask[index_mask] != 0u {
        input[index_input] = value;
    }
}
