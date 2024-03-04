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

fn cubic_convolution1(x: f32, a: f32) -> f32 {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

fn cubic_convolution2(x: f32, a: f32) -> f32 {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

fn cubic_interp1d(x0: f32, x1: f32, x2: f32, x3: f32, t: f32) -> f32 {
    let coeffs0 = cubic_convolution2(t + 1.0, -0.75);
    let coeffs1 = cubic_convolution1(t, -0.75);
    let coeffs2 = cubic_convolution1(1.0 - t, -0.75);
    let coeffs3 = cubic_convolution2(2.0 - t, -0.75);
    return x0 * coeffs0 + x1 * coeffs1 + x2 * coeffs2 + x3 * coeffs3;
}

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

    let input_height = f32(input_shape_2 - 1u);
    let y_frac = f32(h) * input_height / f32(output_shape_2 - 1u);
    let y_in = floor(y_frac);
    let yw = y_frac - y_in;

    let y0 = u32(max(y_in - 1.0, 0.0));
    let y1 = u32(y_in);
    let y2 = u32(min(y_in + 1.0, input_height));
    let y3 = u32(min(y_in + 2.0, input_height));

    let input_width = f32(input_shape_3 - 1u);
    let x_frac = f32(w) * input_width / f32(output_shape_3 - 1u);
    let x_in = floor(x_frac);
    let xw = x_frac - x_in;

    let x0 = u32(max(x_in - 1.0, 0.0));
    let x1 = u32(x_in);
    let x2 = u32(min(x_in + 1.0, input_width));
    let x3 = u32(min(x_in + 2.0, input_width));

    let coefficients0 = cubic_interp1d(
        input[b * input_stride_0 + c * input_stride_1 + y0 * input_stride_2 + x0 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y0 * input_stride_2 + x1 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y0 * input_stride_2 + x2 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y0 * input_stride_2 + x3 * input_stride_3],
        xw,
    );
    let coefficients1 = cubic_interp1d(
        input[b * input_stride_0 + c * input_stride_1 + y1 * input_stride_2 + x0 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y1 * input_stride_2 + x1 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y1 * input_stride_2 + x2 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y1 * input_stride_2 + x3 * input_stride_3],
        xw,
    );
    let coefficients2 = cubic_interp1d(
        input[b * input_stride_0 + c * input_stride_1 + y2 * input_stride_2 + x0 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y2 * input_stride_2 + x1 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y2 * input_stride_2 + x2 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y2 * input_stride_2 + x3 * input_stride_3],
        xw,
    );
    let coefficients3 = cubic_interp1d(
        input[b * input_stride_0 + c * input_stride_1 + y3 * input_stride_2 + x0 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y3 * input_stride_2 + x1 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y3 * input_stride_2 + x2 * input_stride_3],
        input[b * input_stride_0 + c * input_stride_1 + y3 * input_stride_2 + x3 * input_stride_3],
        xw,
    );

    let val = cubic_interp1d(coefficients0, coefficients1, coefficients2, coefficients3, yw);
    output[id] = val;
}
