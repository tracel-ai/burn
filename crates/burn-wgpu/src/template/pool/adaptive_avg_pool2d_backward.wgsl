@group(0)
@binding(0)
var<storage, read> grad: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32, 16>;

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

    let b = id / input_stride_0 % input_shape_0;
    let c = id / input_stride_1 % input_shape_1;
    let ih = id / input_stride_2 % input_shape_2;
    let iw = id / input_stride_3 % input_shape_3;

    let oh_start = start_index(ih, input_shape_2, grad_shape_2);
    let oh_end = end_index(ih, input_shape_2, grad_shape_2);

    let ow_start = start_index(iw, input_shape_3, grad_shape_3);
    let ow_end = end_index(iw, input_shape_3, grad_shape_3);

    var grad_acc = 0.0;

    for (var oh = oh_start; oh < oh_end; oh++) {
        for (var ow = ow_start; ow < ow_end; ow++) {
            let ih_start = start_index(oh, grad_shape_2, input_shape_2);
            let ih_end = end_index(oh, grad_shape_2, input_shape_2);

            let iw_start = start_index(ow, grad_shape_3, input_shape_3);
            let iw_end = end_index(ow, grad_shape_3, input_shape_3);

            let contributed_h = ih >= ih_start && ih < ih_end;
            let contributed_w = iw >= iw_start && iw < iw_end;

            // If no contribution skip
            if !contributed_h || !contributed_w {
                continue;
            }

            let index = b * grad_stride_0 + c * grad_stride_1 + oh * grad_stride_2 + ow * grad_stride_3;
            let count = {{ elem }}((ih_end - ih_start) * (iw_end - iw_start));

            grad_acc += grad[index] / count;
        }
    }

    output[id] = grad_acc;
}

fn start_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
    return u32(floor((f32(output_size_index) * f32(input_size)) / f32(output_size)));
}

fn end_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
    let index = u32(ceil((f32(output_size_index + 1u) * f32(input_size)) / f32(output_size)));

    return min(index, input_size);
}
