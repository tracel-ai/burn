@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> weight: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> bias: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(4)
var<storage, read> info: array<u32, 32>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    let input_stride_0 = info[1];
    let input_stride_1 = info[2];
    let input_stride_2 = info[3];
    let input_stride_3 = info[4];
    let output_stride_0 = info[5];
    let output_stride_1 = info[6];
    let output_stride_2 = info[7];
    let output_stride_3 = info[8];
    let weight_stride_0 = info[9];
    let weight_stride_1 = info[10];
    let weight_stride_2 = info[11];
    let weight_stride_3 = info[12];

    let input_shape_0 = info[13];
    let input_shape_1 = info[14];
    let input_shape_2 = info[15];
    let input_shape_3 = info[16];
    let output_shape_0 = info[17];
    let output_shape_1 = info[18];
    let output_shape_2 = info[19];
    let output_shape_3 = info[20];
    let weight_shape_0 = info[21];
    let weight_shape_1 = info[22];
    let weight_shape_2 = info[23];
    let weight_shape_3 = info[24];

    let stride_0 = info[25];
    let stride_1 = info[26];
    let padding_0 = info[27];
    let padding_1 = info[28];
    let dilation_0 = info[29];
    let dilation_1 = info[30];
    let groups = info[31];

    let in_channels = weight_shape_1;
    let kernel_size_0 = weight_shape_2;
    let kernel_size_1 = weight_shape_3;

    let b = id / output_stride_0 % output_shape_0;
    let oc = id / output_stride_1 % output_shape_1;
    let oh = id / output_stride_2 % output_shape_2;
    let ow = id / output_stride_3 % output_shape_3;
    let g = (weight_shape_0 + oc) % groups;

    var sum = bias[oc];

    let ic_start = in_channels * g;
    let ic_end = in_channels * (g + 1u);

    for (var ic = ic_start; ic < ic_end; ic++) {
        for (var kh = 0u; kh < kernel_size_0; kh++) {
            for (var kw = 0u; kw < kernel_size_1; kw++) {
                let ih = oh * stride_0 + kh * dilation_0;
                let iw = ow * stride_1 + kw * dilation_1;

                // Padding
                if ih >= padding_0 && ih < input_shape_2 + padding_0 && iw >= padding_1 && iw < input_shape_3 + padding_1 {
                    // Correct for padding
                    let ih_pad = ih - padding_0;
                    let iw_pad = iw - padding_1;

                    let weight_ic = ic - (g * in_channels);
                    let index_input = b * input_stride_0 + ic * input_stride_1 + ih_pad * input_stride_2 + iw_pad * input_stride_3;
                    let index_weight = oc * weight_stride_0 + weight_ic * weight_stride_1 + kh * weight_stride_2 + kw * weight_stride_3;

                    sum += input[index_input] * weight[index_weight];
                }
            }
        }
    }

    output[id] = sum;
}
