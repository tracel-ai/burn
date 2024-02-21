@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> indices: array<{{ int }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

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
    let rank = info[0];
    let dim = info[4u * rank + 1u];

    var index_input = 0u;
    var stride = 0u;

    for (var i = 1u; i <= rank; i++) {
        let stride_input = info[i];

        if i - 1u == dim {
            stride = stride_input;
        } else {
            let stride_output = info[i + rank];
            let shape_output = info[i + 3u * rank];
            index_input += id / stride_output % shape_output * stride_input;
        }
    }

    output[id] = input[index_input + u32(indices[id]) * stride];
}
