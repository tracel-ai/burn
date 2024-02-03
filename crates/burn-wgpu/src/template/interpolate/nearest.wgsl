@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

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

    let input_stride_0 = info[1];
    let input_stride_1 = info[2];
    let input_stride_2 = info[3];
    let input_stride_3 = info[4];
    let output_stride_0 = info[5];
    let output_stride_1 = info[6];
    let output_stride_2 = info[7];
    let output_stride_3 = info[8];

    let input_shape_0 = info[9];
    let input_shape_1 = info[10];
    let input_shape_2 = info[11];
    let input_shape_3 = info[12];
    let output_shape_0 = info[13];
    let output_shape_1 = info[14];
    let output_shape_2 = info[15];
    let output_shape_3 = info[16];

    let b = id / output_stride_0 % output_shape_0;
    let c = id / output_stride_1 % output_shape_1;
    let h = id / output_stride_2 % output_shape_2;
    let w = id / output_stride_3 % output_shape_3;

    let y = f32(h) * f32(input_shape_2) / f32(output_shape_2);
    let x = f32(w) * f32(input_shape_3) / f32(output_shape_3);

    let xu = u32(floor(x));
    let yu = u32(floor(y));

    let val = input[b * input_stride_0 + c * input_stride_1 + yu * input_stride_2 + xu * input_stride_3];
    output[id] = val;
}
