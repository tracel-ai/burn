@group(0)
@binding(0)
var<storage, read> x: array<{{ elem }}>;

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

    let output_stride_0 = info[8];
    let output_stride_1 = info[9];
    let output_stride_2 = info[10];
    let output_stride_3 = info[11];
    let output_shape_0 = info[12];
    let output_shape_1 = info[13];
    let output_shape_2 = info[14];
    let output_shape_3 = info[15];

    let b = id / output_stride_0 % output_shape_0;
    let c = id / output_stride_1 % output_shape_1;
    let oh = id / output_stride_2 % output_shape_2;
    let ow = id / output_stride_3 % output_shape_3;

    let ih_start = start_index(oh, output_shape_2, input_shape_2);
    let ih_end = end_index(oh, output_shape_2, input_shape_2);

    let iw_start = start_index(ow, output_shape_3, input_shape_3);
    let iw_end = end_index(ow, output_shape_3, input_shape_3);

    var sum = 0.0;

    for (var ih = ih_start; ih < ih_end; ih++) {
        for (var iw = iw_start; iw < iw_end; iw++) {
            let index_input = b * input_stride_0 + c * input_stride_1 + ih * input_stride_2 + iw * input_stride_3;
            sum += x[index_input];
        }
    }

    let count = {{ elem }}((ih_end - ih_start) * (iw_end - iw_start));
    output[id] = sum / count;
}

fn start_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
    return u32(floor((f32(output_size_index) * f32(input_size)) / f32(output_size)));
}

fn end_index(output_size_index: u32, output_size: u32, input_size: u32) -> u32 {
    let index = u32(ceil((f32(output_size_index + 1u) * f32(input_size)) / f32(output_size)));

    return min(index, input_size);
}

