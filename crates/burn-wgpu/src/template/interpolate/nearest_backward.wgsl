@group(0)
@binding(0)
var<storage, read> grad: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32, 17>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    let output_stride_0 = info[1];
    let output_stride_1 = info[2];
    let output_stride_2 = info[3];
    let output_stride_3 = info[4];
    let grad_stride_0 = info[5];
    let grad_stride_1 = info[6];
    let grad_stride_2 = info[7];
    let grad_stride_3 = info[8];

    let output_shape_0 = info[9];
    let output_shape_1 = info[10];
    let output_shape_2 = info[11];
    let output_shape_3 = info[12];
    let grad_shape_0 = info[13];
    let grad_shape_1 = info[14];
    let grad_shape_2 = info[15];
    let grad_shape_3 = info[16];

    let b = (id / output_stride_0) % output_shape_0;
    let c = (id / output_stride_1) % output_shape_1;
    let oh = (id / output_stride_2) % output_shape_2;
    let ow = (id / output_stride_3) % output_shape_3;

    let gh_start = start_index(oh, grad_shape_2, output_shape_2);
    let gh_end = end_index(oh, grad_shape_2, output_shape_2);

    let gw_start = start_index(ow, grad_shape_3, output_shape_3);
    let gw_end = end_index(ow, grad_shape_3, output_shape_3);

    var grad_acc = 0.0;

    for (var gh = gh_start; gh < gh_end; gh++) {
        for (var gw = gw_start; gw < gw_end; gw++) {
            let index = b * grad_stride_0 + c * grad_stride_1 + gh * grad_stride_2 + gw * grad_stride_3;
            grad_acc += grad[index];
        }
    }

    output[id] = grad_acc;
}

fn start_index(input_index: u32, output_size: u32, input_size: u32) -> u32 {
    return u32(ceil(f32(input_index) * (f32(output_size) / f32(input_size))));
}

fn end_index(input_index: u32, output_size: u32, input_size: u32) -> u32 {
    let index = u32(ceil(f32(input_index + 1u) * (f32(output_size) / f32(input_size))));
    return min(index, output_size);
}
