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
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let dim: u32 = info[0];
    let dim_cat = info[4u * dim + 1u];
    let dim_cat_index = info[4u * dim + 2u];

    var index_input: u32 = 0u;
    var index_output: u32 = 0u;
    var num_elem_input = 1u;

    for (var i: u32 = 1u; i <= dim; i++) {
        let stride_input = info[i];
        let stride_output = info[i + dim];
        let shape_input = info[i + 2u * dim];
        let shape_output = info[i + 3u * dim];

        num_elem_input *= shape_input;

        let num_block_output = global_id.x / stride_output % shape_output;
        index_input += num_block_output * stride_input;

        if i - 1u == dim_cat {
            index_output += (num_block_output + dim_cat_index) * stride_output;
        } else {
            index_output += num_block_output * stride_output;
        }
    }

    if num_elem_input < global_id.x {
        output[index_output] = input[index_input];
    }
}

