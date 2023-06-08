@group(0)
@binding(0)
var<storage, read> input: array<elem>;

@group(0)
@binding(1)
var<storage, read_write> output: array<elem>;

@group(0)
@binding(2)
var<storage, read> info: array<u32>;

var<workgroup> data: array<elem, WORKGROUP_SIZE_X>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let dim: u32 = info[0];
    let dim_reduce = info[4u * dim + 1u];
    var index_offset: u32 = 0u;
    var stride_dim: u32 = 0u;
    var shape_dim: u32 = 0u;

    for (var i: u32 = 0u; i < dim; i++) {
        let stride_input = info[i + 1u];
        let stride_output = info[i + dim + 1u];
        let shape_output = info[i + 3u * dim + 1u];

        let num_block = global_id.x / stride_output % shape_output;

        if i != dim_reduce {
            index_offset += num_block * stride_input;
        } else {
            let shape_input = info[i + 2u * dim + 1u];
            index_offset += num_block;
            stride_dim = stride_input;
            shape_dim = shape_input;
        }
    }

    var sum = elem(0);

    for (var i = 0u; i < shape_dim; i++) {
        let index_input = i * stride_dim;
        sum += input[index_input + index_offset];
    }

    ASSIGN
}
