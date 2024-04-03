use burn_tensor::ElementConversion;
use ndarray::Ix3;

use crate::{iter_range_par, run_par, FloatNdArrayElement, NdArrayTensor, UnsafeSharedRef};

struct SingleIterParams {
    /// Nth iteration of the Fast Fourier transform
    iteration: usize,
    remaining_iterations: usize,
    /// Mask used as "mask & sample_id" to extract the required bits
    sign_mask: usize,
    /// Mask used as "mask & sample_id" to extract the required bits
    even_mask: usize,
    /// Argument (theta) of complex `z = r e^{i\theta}` such that z^n = 1.
    nth_root_of_unity: f64,
}

impl SingleIterParams {
    fn new(iteration: usize, required_iterations: usize) -> Self {
        let remaining_iterations = required_iterations - iteration;

        // Some of these expressions look a little crazy - much easier to see how they
        //  work if inspecting these numbers as binary strings (most have only 1 bit).
        SingleIterParams {
            iteration,
            remaining_iterations,
            sign_mask: (1 << (remaining_iterations - 1)),
            even_mask: !(1 << (remaining_iterations - 1)),
            nth_root_of_unity: -2. * std::f64::consts::PI / ((2 << iteration) as f64),
        }
    }
}

pub(crate) fn fft1d<E: FloatNdArrayElement>(input: NdArrayTensor<E, 3>) -> NdArrayTensor<E, 3> {
    let [batch_size, num_samples, complex] = input.shape().dims;

    // Require complex input - an extra dimension is used that is always size 2, for complex.
    if complex != 2 {
        panic!(
            "The last dimension must have length 2 (real, imaginary). For real inputs, consider
            adding an extra dimension, with the imaginary part filled with zeros."
        )
    }

    // This FFT implementation is Cooley-Tukey, only supports power of 2 along transform dim.
    if num_samples == 0 || (num_samples & (num_samples - 1)) != 0 {
        panic!(
            "The dimension that the FFT is computed over must be a power of 2, got {}",
            num_samples
        );
    }

    let mut input = input
        .array
        .into_owned()
        .into_dimensionality::<Ix3>()
        .unwrap();
    let mut output = ndarray::Array3::<E>::zeros((batch_size, num_samples, 2));

    let input_unsafe_ref = UnsafeSharedRef::new(&mut input);
    let output_unsafe_ref = UnsafeSharedRef::new(&mut output);

    let num_fft_iters: usize = f32::log2(num_samples as f32) as usize;

    for fft_iter in 0..num_fft_iters {
        let params = SingleIterParams::new(fft_iter, num_fft_iters);

        let is_last_iter = (fft_iter + 1) == num_fft_iters;
        let even_iter = fft_iter % 2 == 0;

        unsafe {
            run_par!(|| {
                iter_range_par!(0, num_samples).for_each(|sample_id| {
                    let k_even = sample_id & params.even_mask;
                    let k_odd = k_even + (num_samples >> (1 + params.iteration));
                    let twiddle: (E, E) = get_twiddle_factor(&params, sample_id);

                    // Alternate between the array being written to and from.
                    let (input_ref, output_ref) = match even_iter {
                        true => (input_unsafe_ref.get(), output_unsafe_ref.get()),
                        false => (output_unsafe_ref.get(), input_unsafe_ref.get()),
                    };

                    for b in 0..batch_size {
                        /*
                            Where E_k are even-indexed inputs and O_k are odd-indexed inputs:
                                X_k = E_k + twiddle_factor * O_k

                            Taking real and imaginary parts:
                                Re(X_k) = Re(E_k) + Re(twiddle_factor) * Re(O_k) - Im(twiddle_factor) * Im(O_k)
                                Im(X_k) = Im( as f32E_k) + Re(twiddle_factor) * Im(O_k) + Im(twiddle_factor) * Re(O_k)

                            See https://en.wikipedia.org/wiki/Cooley-Tukey_FFT_algorithm#The_radix-2_DIT_case
                        */

                        let e_k = (input_ref[(b, k_even, 0)], input_ref[(b, k_even, 1)]);
                        let o_k = (input_ref[(b, k_odd, 0)], input_ref[(b, k_odd, 1)]);

                        let x_k = (
                            e_k.0 + twiddle.0 * o_k.0 - twiddle.1 * o_k.1,
                            e_k.1 + twiddle.0 * o_k.1 + twiddle.1 * o_k.0,
                        );

                        let out_id = match is_last_iter {
                            false => sample_id,
                            true => reverse_bits(sample_id, num_fft_iters),
                        };

                        output_ref[(b, out_id, 0)] = x_k.0;
                        output_ref[(b, out_id, 1)] = x_k.1;
                    }
                });
            });
        }
    }

    let output = match num_fft_iters % 2 == 0 {
        true => input,
        false => output,
    };

    NdArrayTensor::new(output.into_dyn().into_shared())
}

fn get_twiddle_factor<E: burn_tensor::Element>(
    params: &SingleIterParams,
    sample_id: usize,
) -> (E, E) {
    // Indices are bit reversed at each iteration, but there's also a different
    //  number of values at each iteration. e.g., for the 3 iterations of an 8
    //  length transform the mapping from sample_id to k is as follows:
    //  0: [0, 0, 0, 0, 1, 1, 1, 1]
    //  1: [0, 0, 2, 2, 1, 1, 3, 3]
    //  2: [0, 4, 2, 6, 1, 7, 3, 5]
    let k = reverse_bits(sample_id >> (params.remaining_iterations), params.iteration) as f64;

    /*
        `nth_root_of_unity` has the property that

            exp(-j * root) = 1.

        Raising both sides to the power of any integer k,

            exp(-j * root)^k = 1^k
            exp(-j * k * root) = 1

        The nth root is special because it satisfies this.
    */
    let complex_angle = k * params.nth_root_of_unity;
    let sign = {
        if (sample_id & params.sign_mask) > 0 {
            -1.
        } else {
            1.
        }
    };

    // Euler's formula: exp^(ix) = cos(x) + i sin(x)
    // These are both real numbers, but represent coefficients of a + i*b.
    (
        (sign * f64::cos(complex_angle)).elem::<E>(),
        (sign * f64::sin(complex_angle)).elem::<E>(),
    )
}

fn reverse_bits(n: usize, no_of_bits: usize) -> usize {
    let mut result = 0;
    let mut n = n;

    for _ in 0..no_of_bits {
        result <<= 1;
        result |= n & 1;
        n >>= 1;
    }
    result
}
