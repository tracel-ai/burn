@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>; // Unfolded result

@group(0)
@binding(2)
var<storage, read> info: array<u32, 32>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let linear_id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    let input_shape_1 = info[2];
    let input_shape_2 = info[3];
    let input_shape_3 = info[4];

    let kernel_size_0 = info[9]; 
    let kernel_size_1 = info[10];  
    let stride_0 = info[11];
    let stride_1 = info[12];
    let padding_0 = info[13];
    let padding_1 = info[14];
    let dilation_0 = info[15];
    let dilation_1 = info[16];

    // Determine the current position to process
    let b = linear_id / (input_shape_1 * input_shape_2 * input_shape_3);
    let c = (linear_id % (input_shape_1 * input_shape_2 * input_shape_3)) / (input_shape_2 * input_shape_3);
    let h = (linear_id % (input_shape_2 * input_shape_3)) / input_shape_3;
    let w = linear_id % input_shape_3;

    // Initialize index for the output tensor
    var output_idx = 0u;

    // Iterate over channels and local patch
    for (var ic = c; ic < input_shape_1; ic++) {
        for (var kh = 0u; kh < kernel_size_0; kh++) {
            for (var kw = 0u; kw < kernel_size_1; kw++) {
                let ih = h * stride_0 + kh * dilation_0 - padding_0;
                let iw = w * stride_1 + kw * dilation_1 - padding_1;

                // Boundary check
                if ih >= 0u && ih < input_shape_2 && iw >= 0u && iw < input_shape_3 {
                    let index_input = b * input_shape_1 * input_shape_2 * input_shape_3 + ic * input_shape_2 * input_shape_3 + ih * input_shape_3 + iw;

                    // Place the value from input into the correct position in the unfolded output tensor
                    output[output_idx] = input[index_input];
                } else {
                    // Set to zero, we're in a padded region
                    output[output_idx] = 0.0;
                }
                output_idx++;
            }
        }
    }
}
