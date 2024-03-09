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

    let y_frac = f32(h) * f32(input_shape_2 - 1u) / f32(output_shape_2 - 1u);
    let y0 = floor(y_frac);
    let y1 = ceil(y_frac);
    let yw = y_frac - y0;

    let x_frac = f32(w) * f32(input_shape_3 - 1u) / f32(output_shape_3 - 1u);
    let x0 = floor(x_frac);
    let x1 = ceil(x_frac);
    let xw = x_frac - x0;

    let x0u = u32(x0);
    let x1u = u32(x1);
    let y0u = u32(y0);
    let y1u = u32(y1);

    let p_a = input[b * input_stride_0 + c * input_stride_1 + y0u * input_stride_2 + x0u * input_stride_3];
    let p_b = input[b * input_stride_0 + c * input_stride_1 + y0u * input_stride_2 + x1u * input_stride_3];
    let p_c = input[b * input_stride_0 + c * input_stride_1 + y1u * input_stride_2 + x0u * input_stride_3];
    let p_d = input[b * input_stride_0 + c * input_stride_1 + y1u * input_stride_2 + x1u * input_stride_3];

    let pa = p_a * (1.0 - xw) * (1.0 - yw);
    let pb = p_b * xw * (1.0 - yw);
    let pc = p_c * (1.0 - xw) * yw;
    let pd = p_d * xw * yw;

    output[id] = pa + pb + pc + pd;
}
