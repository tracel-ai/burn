@group(0)
@binding(0)
var<storage, read> x: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32, 24>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    let input_stride_0 = info[0];
    let input_stride_1 = info[1];
    let input_stride_2 = info[2];
    let input_stride_3 = info[3];
    let input_shape_0 = info[4];
    let input_shape_1 = info[5];
    let input_shape_2 = info[6];
    let input_shape_3 = info[7];

    let output_stride_0 = info[8];
    let output_stride_1 = info[9];
    let output_stride_2 = info[10];
    let output_stride_3 = info[11];
    let output_shape_0 = info[12];
    let output_shape_1 = info[13];
    let output_shape_2 = info[14];
    let output_shape_3 = info[15];

    let kernel_size_0 = info[16];
    let kernel_size_1 = info[17];
    let pool_stride_0 = info[18];
    let pool_stride_1 = info[19];
    let padding_0 = info[20];
    let padding_1 = info[21];
    let dilation_0 = info[22];
    let dilation_1 = info[23];

    let b = id / output_stride_0 % output_shape_0;
    let c = id / output_stride_1 % output_shape_1;
    let oh = id / output_stride_2 % output_shape_2;
    let ow = id / output_stride_3 % output_shape_3;

    var max_val = -32767.0;

    for (var kh = 0u; kh < kernel_size_0; kh++) {
        let ih = oh * pool_stride_0 + kh * dilation_0;

        // Padding
        if ih < padding_0 || ih >= input_shape_2 + padding_0 {
            continue;
        }

        for (var kw = 0u; kw < kernel_size_1; kw++) {
            let iw = ow * pool_stride_1 + kw * dilation_1;

            // Padding
            if iw < padding_1 || iw >= input_shape_3 + padding_1 {
                continue;
            }

            // Correct indexes for padding
            let ih_pad = ih - padding_0;
            let iw_pad = iw - padding_1;

            let index_input = b * input_stride_0 + c * input_stride_1 + ih_pad * input_stride_2 + iw_pad * input_stride_3;
            let val = x[index_input];
            max_val = max(max_val, val);
        }
    }

    output[id] = max_val;
}
