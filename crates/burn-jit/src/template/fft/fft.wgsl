@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32, 32>;

///////////////
// Constants //
///////////////

// const BLOCK_SIZE: u32 = {{ workgroup_size_x }}u;
const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

// Shape = [NUM_ROWS, NUM_COLS] = [num_y_values, num_x_values]
//  Note 1D FFT done along the X direction, operation cloned along Y.
// const NUM_ROWS: u32 = {{ num_rows }}u;
// const NUM_COLS: u32 = {{ num_cols }}u;
// const NUM_FFT_ITERS: u32 = {{ num_fft_iters }}u;
// const FFT_ITER: u32 = {{ fft_iter }}u;

////////////////////////////////////
// Single parameterised FFT stage //
////////////////////////////////////

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    let input_stride_0 = info[1];
    let input_stride_1 = info[2];
    let input_stride_2 = info[3];
    let output_stride_0 = info[4];
    let output_stride_1 = info[5];
    let output_stride_2 = info[6];

    let input_shape_0 = info[7];
    let input_shape_1 = info[8];
    let input_shape_2 = info[9];
    let output_shape_0 = info[10];
    let output_shape_1 = info[11];
    let output_shape_2 = info[12];

    let num_fft_iters = info[13];
    let fft_iter = info[14];

    // FFT is done over X dimension, parallelised over Y dimension. It's always
    //  a 1D transform, but many 1D transforms are done in parallel.
    let oy = id / input_stride_0 % input_shape_0;
    let ox = id / input_stride_1 % input_shape_1;
    let oc = id / input_stride_2 % input_shape_2;

    // let foo = input[id];
    // output[id] = 42.0;
    // return;

    // Only ever 0 or 1 (real or imaginary). Arbitrarily choose the real index
    //  to do the complex calculations.
    if (oc >= 1u) {
        return;
    }


    let NUM_FFT_ITERS = num_fft_iters;
    let FFT_ITER = fft_iter;
    let LAST_ITERATION: bool = (FFT_ITER + 1u) == NUM_FFT_ITERS;

    let NUM_ROWS = input_shape_0;
    let NUM_COLS = input_shape_1;

    let DFT_SIZE: u32 = 2u << FFT_ITER;
    let NUM_DFTS: u32 = NUM_COLS >> (FFT_ITER + 1u);
    let EVEN_MASK: u32 = NUM_DFTS ^ 0xFFFFu;

    let PI: f32 = 3.1415926535898;
    let TWIDDLE_ANGLE_BASE: f32 = - 2. * PI / f32(DFT_SIZE);


    // let x: u32 = workgroup_id.x * BLOCK_SIZE + (local_idx / BLOCK_SIZE);
    // let y: u32 = workgroup_id.y * BLOCK_SIZE + (local_idx % BLOCK_SIZE);

    let x = ox;
    let y = oy;

    // Returns if outside the output dimension
    if x >= NUM_COLS || y >= NUM_ROWS {
        return;
    }

    ///////////////////////////////////////
    // Position-dependant FFT Parameters //
    ///////////////////////////////////////

    let x_even: u32 = x & EVEN_MASK;
    let x_odd: u32 = x_even + NUM_DFTS;
    let exponent: u32 = reverse_bits(x >> (NUM_FFT_ITERS - FFT_ITER), FFT_ITER);
    let negative_instead_of_plus: bool = (x & NUM_DFTS) > 0u;

    /////////////
    // Indices //
    /////////////
    
    // Indices for this stage. Note that index y are just constant - 
    //  we just duplicate it across all rows.
    let i_even: u32 = y * NUM_COLS + x_even;
    let i_odd: u32 = y * NUM_COLS + x_odd;
    var i_out: u32 = y * NUM_COLS + x;

    // Running the FFT algorithm like this results in a bit-reversed ordered
    //  output. i.e. the element 000, 001, ..., 110, 111 are now sorted if
    //  they were actually 000, 100, ..., 011, 111. On the last step, undo this
    //  mapping. 

    if LAST_ITERATION {
        let remapped_x = reverse_bits(x, NUM_FFT_ITERS);
        i_out = y * NUM_COLS + remapped_x;
    }

    ///////////////////////////////////////////
    // Calculate weights ("Twiddle Factors") //
    ///////////////////////////////////////////

    var pm1: f32 = 1.;
    if(negative_instead_of_plus) {
        pm1 = -1.;
    }

    let twiddle_angle: f32 = TWIDDLE_ANGLE_BASE * f32(exponent);
    let wt_re: f32 = pm1 * cos(twiddle_angle);
    let wt_im: f32 = pm1 * sin(twiddle_angle);

    // Real part of (a + bj)(c + dj) = ab + bd(j*j) = ab - bd
    // f_hat_re[i_out] = f_re[i_even] + wt_re * f_re[i_odd] - wt_im * f_im[i_odd];
    output[id] = input[i_even] + wt_re * input[i_odd] - wt_im * input[i_odd + 1u];

    // Imaginary part of (a + bj)(c + dj) = ad(j) + bc(j) = (ad + bd)j
    // f_hat_im[i_out] = f_im[i_even] + wt_re * f_im[i_odd] + wt_im * f_re[i_odd];
    output[id + 1u] = input[i_even + 1u] + wt_re * input[i_odd + 1u] + wt_im * input[i_odd];
    
    // output[id] = f32(output_stride_2);
    // output[id + 1u] = f32(input_stride_2);

}

fn reverse_bits(x_: u32, num_bits: u32) -> u32 {
    // TODO there is actually a reverse bits function built in to WGSL

    var result = 0u;
    var x = x_;

    for(var i = 0u; i < num_bits; i = i + 1u) {
        // Self-explanatory >:I
        //  ...
        //  just kidding.
        //  Think about it - we're pulling bits from left to right.
        //  One gets bigger whilst the other gets smaller.
        result <<= 1u;
        result |= x & 1u;
        x >>= 1u;
    }
    return result;
}