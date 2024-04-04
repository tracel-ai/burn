@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32, 32>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;
const PI: f32 = 3.141592653589793115997963468544185161590576171875;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    //////////////////////////////////////////////////
    // Single FFT stage at a single (x, y) position //
    //////////////////////////////////////////////////

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

    // Number of independent FFTs at this stage (starts at x_width/2, halves each time)
    let num_transforms: u32 = input_shape_1 >> (fft_iter + 1u);

    // Binary mask for extracting the index of E_k
    let even_mask: u32 = num_transforms ^ 0xFFFFu;

    // Returns if outside the output dimension
    if oy >= output_shape_0 || ox >= output_shape_1 {
        return;
    }

    // Position-dependent FFT Parameters
    let ix_even: u32 = ox & even_mask;
    let ix_odd: u32 = ix_even + num_transforms;
    let exponent: u32 = reverse_bits(ox >> (num_fft_iters - fft_iter), fft_iter);
    let negative_instead_of_plus: bool = (ox & num_transforms) > 0u;

    // Indices
    let i_even_re: u32 = iy * input_stride_0 + ix_even * input_stride_1 + 0u * input_stride_2;
    let i_even_im: u32 = iy * input_stride_0 + ix_even * input_stride_1 + 1u * input_stride_2;
    
    let i_odd_re: u32 = iy * input_stride_0 + ix_odd * input_stride_1 + 0u * input_stride_2;
    let i_odd_im: u32 = iy * input_stride_0 + ix_odd * input_stride_1 + 1u * input_stride_2;

    // Running the FFT algorithm like this results in a bit-reversed ordered
    //  output. i.e. the element 000, 001, ..., 110, 111 are now sorted if
    //  they were actually 000, 100, ..., 011, 111. On the last step, undo this
    //  mapping, by choosing ox differently.

    var ox_r = 0u;
    if is_final_iter {
        ox_r = reverse_bits(ox, num_fft_iters);
    } else {
        ox_r = ox;
    }
    
    let i_out_re: u32 = oy * output_stride_0 + ox_r * output_stride_1 + 0u * output_stride_2;
    let i_out_im: u32 = oy * output_stride_0 + ox_r * output_stride_1 + 1u * output_stride_2;

    // Here we compute the main computation step for each index.
    //  See the last two equations of:
    //  https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm#The_radix-2_DIT_case
    // X_k = E_k + w_k * O_k
    //  Where w_k is the +/- exp(-2 pi i k / n) term. Note the plus / minus 
    //  is included in the value of the weight.

    var pm1: f32 = 1.;
    if(negative_instead_of_plus) {
        pm1 = -1.;
    }

    // Width of the FFT at this stage (starts at 2, doubles each time)
    let n: u32 = 2u << fft_iter;
    let w_k_theta: f32 = - 2. * PI * f32(exponent) / f32(n);
    let w_k_re: f32 = pm1 * cos(w_k_theta);
    let w_k_im: f32 = pm1 * sin(w_k_theta);

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
    // For input:
    // 00000000000000000000000000011011
    //  num_bits = 1 gives:
    // 00000000000000000000000000000001
    //  num_bits = 2:
    // 00000000000000000000000000000011
    //  num_bits = 3:
    // 00000000000000000000000000000110
    //  num_bits = 4:
    // 00000000000000000000000000001101
    //  etc...
    return reverseBits(x_) >> (32u - min(32u, num_bits));
}