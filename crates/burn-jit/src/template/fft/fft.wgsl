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
// const num_fft_iters: u32 = {{ num_fft_iters }}u;
// const fft_iter: u32 = {{ fft_iter }}u;

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
    let is_final_iter: bool = (fft_iter + 1u) == num_fft_iters;

    // FFT is done over X dimension, parallelised over Y dimension. It's always
    //  a 1D transform, but many 1D transforms are done in parallel.
    let oy = id / output_stride_0 % output_shape_0;
    let ox = id / output_stride_1 % output_shape_1;
    let oc = id / output_stride_2 % output_shape_2;

    let iy = oy;
    
    // Only ever 0 or 1 (real or imaginary). Arbitrarily choose the real index
    //  to do both complex calculations.
    if (oc >= 1u) {
        return;
    }


    let NUM_ROWS = input_shape_0;
    let NUM_COLS = input_shape_1;

    let DFT_SIZE: u32 = 2u << fft_iter;
    let NUM_DFTS: u32 = NUM_COLS >> (fft_iter + 1u);
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
    let exponent: u32 = reverse_bits(x >> (num_fft_iters - fft_iter), fft_iter);
    let negative_instead_of_plus: bool = (x & NUM_DFTS) > 0u;

    /////////////
    // Indices //
    /////////////
    
    // Indices for this stage. Note that index y are just constant - 
    //  we just duplicate it across all rows.
    // let i_even: u32 = y * NUM_COLS + x_even;
    // let i_odd: u32 = y * NUM_COLS + x_odd;
    // var i_out: u32 = y * NUM_COLS + x;

    let i_even_re: u32 = oy * input_stride_0 + x_even * input_stride_1 + 0u * input_stride_2;
    let i_even_im: u32 = oy * input_stride_0 + x_even * input_stride_1 + 1u * input_stride_2;
    
    let i_odd_re: u32 = oy * input_stride_0 + x_odd * input_stride_1 + 0u * input_stride_2;
    let i_odd_im: u32 = oy * input_stride_0 + x_odd * input_stride_1 + 1u * input_stride_2;

    // Running the FFT algorithm like this results in a bit-reversed ordered
    //  output. i.e. the element 000, 001, ..., 110, 111 are now sorted if
    //  they were actually 000, 100, ..., 011, 111. On the last step, undo this
    //  mapping. 

    var i_out_re = 0u;
    var i_out_im = 0u;
    
    if is_final_iter {
        let remapped_x = reverse_bits(x, num_fft_iters);
        i_out_re = oy * output_stride_0 + remapped_x * output_stride_1 + 0u * output_stride_2;
        i_out_im = oy * output_stride_0 + remapped_x * output_stride_1 + 1u * output_stride_2;
    } else {
        i_out_re = oy * output_stride_0 + x * output_stride_1 + 0u * output_stride_2;
        i_out_im = oy * output_stride_0 + x * output_stride_1 + 1u * output_stride_2;
    }

    ///////////////////////////////////////////
    // Calculate weights ("Twiddle Factors") //
    ///////////////////////////////////////////

    var pm1: f32 = 1.;
    if(negative_instead_of_plus) {
        pm1 = -1.;
    }

    
    // Here we compute the main computation step for each index.
    //  See the last two equations of:
    //  https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm#The_radix-2_DIT_case
    // X_k = E_k + w_k * O_k
    //  Where w_k is the +/- exp(-2 pi i k / N) term. Note the plus / minus 
    //  is included in the value of the weight.
    
    let twiddle_angle: f32 = TWIDDLE_ANGLE_BASE * f32(exponent);
    let w_k_re: f32 = pm1 * cos(twiddle_angle);
    let w_k_im: f32 = pm1 * sin(twiddle_angle);

    let e_k_re = input[i_even_re];
    let e_k_im = input[i_even_im];
    let o_k_re = input[i_odd_re];
    let o_k_im = input[i_odd_im];

    // Note the following:
    // Real part of (a + bj)(c + dj) = ac + bd(j*j) = ac - bd
    // Imaginary part of (a + bj)(c + dj) = ad(j) + bc(j) = (ad + bc)j
    // These are used for w_k * O_k; E_k real and imaginary parts are just added.
    output[i_out_re] = e_k_re + w_k_re * o_k_re - w_k_im * o_k_im;
    output[i_out_im] = e_k_im + w_k_re * o_k_im + w_k_im * o_k_re;
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