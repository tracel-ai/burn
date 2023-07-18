@group(0)
@binding(0)
var<storage, read> grad: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32, 22>;

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

    let grad_stride_0 = info[8];
    let grad_stride_1 = info[9];
    let grad_stride_2 = info[10];
    let grad_stride_3 = info[11];
    let grad_shape_0 = info[12];
    let grad_shape_1 = info[13];
    let grad_shape_2 = info[14];
    let grad_shape_3 = info[15];

    let kernel_size_0 = info[16];
    let kernel_size_1 = info[17];
    let pool_stride_0 = info[18];
    let pool_stride_1 = info[19];
    let padding_0 = info[20];
    let padding_1 = info[21];

    let b = id / input_stride_0 % input_shape_0;
    let c = id / input_stride_1 % input_shape_1;
    let ih = id / input_stride_2 % input_shape_2;
    let iw = id / input_stride_3 % input_shape_3;

    // The maximum number of overlapping filters that may content the current index.
    let kms_0 = i32(kernel_size_0) - i32(pool_stride_0);
    let kms_1 = i32(kernel_size_1) - i32(pool_stride_1);

    let oh_start_tmp = (i32(ih + padding_0) - kms_0) / i32(pool_stride_0);
    let ow_start_tmp = (i32(iw + padding_1) - kms_1) / i32(pool_stride_1);

    let oh_start = u32(max(oh_start_tmp, 0));
    let ow_start = u32(max(ow_start_tmp, 0));

    let oh_end = u32(max(kms_0, 0)) + oh_start;
    let ow_end = u32(max(kms_1, 0)) + ow_start;

    var grad_acc = 0.0;
    // We iterate over each potentially resulting overlapping filters and check
    // if their max index is the current one.
    for (var oh = oh_start; oh <= oh_end; oh++) {
        for (var ow = ow_start; ow <= ow_end; ow++) {

            // We check for every kernel size combination if the current oh/ow have contributed to the output.
            var contributed_h = false;
            var contributed_w = false;
            for (var kh = 0u; kh < kernel_size_0; kh++) {
                for (var kw = 0u; kw < kernel_size_1; kw++) {
                    let ih_tmp = oh * pool_stride_0 + kh - padding_0;
                    let iw_tmp = ow * pool_stride_1 + kw - padding_1;

                    if ih_tmp == ih {
                        contributed_h = true;
                    }
                    if iw_tmp == iw {
                        contributed_w = true;
                    }

                }
            }

            // If no contribution or outside of output shape, skip.
            if !contributed_h || oh >= grad_shape_2  || !contributed_w || ow >= grad_shape_3 {
                continue;
            }

            let index = b * grad_stride_0 + c * grad_stride_1 + oh * grad_stride_2 + ow * grad_stride_3;
            grad_acc += grad[index] ;
        }
    }

    output[id] = grad_acc / {{ elem }}(kernel_size_0 * kernel_size_1);
}
