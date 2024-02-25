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
    let dim_cat = info[4u * dim + 1u];
    let dim_cat_index = info[4u * dim + 2u];

    var num_elems = 1u;
    var index_input = 0u;
    var index_output = 0u;

    for (var i: u32 = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_output = info[i + dim];
        let shape_input = info[i + 2u * dim];
        let shape_output = info[i + 3u * dim];

        let num_block_output = id / stride_input % shape_input;
        index_input += num_block_output * stride_input;
        num_elems *= shape_input;

        if i - 1u == dim_cat {
            index_output += (num_block_output + dim_cat_index) * stride_output;
        } else {
            index_output += num_block_output * stride_output;
        }
    }

    if id < num_elems {
        output[index_output] = input[index_input];
    }
}

