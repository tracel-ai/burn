@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> value: {{ elem }};

@group(0)
@binding(2)
var<storage, read> mask: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(4)
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
    var index_input = 0u;
    var index_mask = 0u;

    for (var i = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_mask = info[i + dim];
        let stride_output = info[i + 2u * dim];
        let shape_input = info[i + 3u * dim];
        let shape_mask = info[i + 4u * dim];

        index_input += id / stride_output % shape_input * stride_input;
        index_mask += id / stride_output % shape_mask * stride_mask;
    }


    if mask[index_mask] != 0u {
        output[id] = value;
    } else {
        output[id] = input[index_input];
    }
}
