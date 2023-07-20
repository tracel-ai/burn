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

    let in_channels = weight_shape_0;
    let kernel_size_0 = weight_shape_2;
    let kernel_size_1 = weight_shape_3;

    let b = id / output_stride_0 % output_shape_0;
    let oc_out = id / output_stride_1 % output_shape_1;
    let oh = id / output_stride_2 % output_shape_2;
    let ow = id / output_stride_3 % output_shape_3;

    let k = oc_out / weight_shape_1;
    let g = k % groups;
    let oc = oc_out - (weight_shape_1 * g);

    var sum = bias[oc_out];

    let ic_start = g * (in_channels / groups);
    let ic_end = ic_start + in_channels / groups;

    // The maximum number of overlapping filters that may content the current index.
    let kms_0 = i32(kernel_size_0 * dilation_0) - i32(stride_0);
    let kms_1 = i32(kernel_size_1 * dilation_1) - i32(stride_1);

    let ih_start_tmp = (i32(oh + padding_0) - kms_0) / i32(stride_0);
    let iw_start_tmp = (i32(ow + padding_1) - kms_1) / i32(stride_1);

    let ih_start = u32(max(ih_start_tmp, 0));
    let iw_start = u32(max(iw_start_tmp, 0));

    let ih_end = min(u32(max(kms_0 + ih_start_tmp + 1, 0)), input_shape_2);
    let iw_end = min(u32(max(kms_1 + iw_start_tmp + 1, 0)), input_shape_3);

    for (var ic = ic_start; ic < ic_end; ic++) {
        for (var ih = ih_start; ih < ih_end; ih++) {
            for (var iw = iw_start; iw < iw_end; iw++) {
                for (var kh = 0u; kh < kernel_size_0; kh++) {
                    for (var kw = 0u; kw < kernel_size_1; kw++) {
                        let oh_tmp = ih * stride_0 + kh * dilation_0;
                        let ow_tmp = iw * stride_1 + kw * dilation_1;

                        if oh_tmp >= padding_0 && ow_tmp >= padding_1 {
                            let oh_tmp_pad = oh_tmp - padding_0;
                            let ow_tmp_pad = ow_tmp - padding_1;

                            if oh_tmp_pad == oh && ow_tmp_pad == ow {
                                let index_input = b * input_stride_0 + ic * input_stride_1 + ih * input_stride_2 + iw * input_stride_3;
                                let index_weight = ic * weight_stride_0 + oc * weight_stride_1 + kh * weight_stride_2 + kw * weight_stride_3;

                                sum += input[index_input] * weight[index_weight];
                            }
                        }
                    }
                }
            }
        }
    }

    output[id] = sum;
}
