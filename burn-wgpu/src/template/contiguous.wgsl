@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

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
    var index_input: u32 = 0u;

    for (var i: u32 = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_output = info[i + dim];
        let shape_input = info[i + 2u * dim];

        index_input += id / stride_output % shape_input * stride_input;
    }

    output[id] = input[index_input];
}
