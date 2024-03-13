use burn_tensor::ElementConversion;
use ndarray::{Array3, Ix3};
#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{
    iter_range_par, run_par, FloatNdArrayElement, NdArray, NdArrayTensor, UnsafeSharedRef,
};

struct SingleIterParams {
    /// Nth iteration of the Fast Fourier transform
    iteration: usize,
    remaining_iterations: usize,
    transform_width: usize,
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

        let samples = usize::pow(2, required_iterations as u32);
        let num_dfts = samples >> (1 + iteration);
        SingleIterParams {
            iteration,
            remaining_iterations,
            transform_width: 2 << iteration,
            sign_mask: 1 << (remaining_iterations - 1),
            even_mask: num_dfts ^ 0xFFFF,
            nth_root_of_unity: -2. * std::f64::consts::PI / ((2 << iteration) as f64),
        }
    }
}

pub(crate) fn fft1d<E: FloatNdArrayElement>(x: NdArrayTensor<E, 3>) -> NdArrayTensor<E, 3> {
    let [batch_size, samples, complex] = x.shape().dims;

    // Require complex input - an extra dimension is used that is always size 2, for complex.
    if complex != 2 {
        panic!(
            "The last dimension must have length 2 (real, imaginary). For real inputs, consider
             using Tensor::cat(vec![x, Tensor::zeros_like(x)], 3)."
        )
    }

    // This FFT implementation is Cooley-Tukey, only supports power of 2 along transform dim.
    if samples == 0 || (samples & (samples - 1)) != 0 {
        panic!(
            "The dimension that the FFT is computed over must be a power of 2, got {}",
            samples
        );
    }

    // The arrays used to compute FFT. This FFT is an iterative algorithm that
    //  writes back and forth between two buffers.
    let mut x1 = x.array.into_owned().into_dimensionality::<Ix3>().unwrap();
    let mut x2 = ndarray::Array3::<E>::zeros((batch_size, samples, 2));
    let mut output = Array3::zeros((batch_size, samples, 2));

    let x1_unsafe_ref = UnsafeSharedRef::new(&mut x1);
    let x2_unsafe_ref = UnsafeSharedRef::new(&mut x2);
    let out_shared_ref = UnsafeSharedRef::new(&mut output);

    let required_iterations: usize = f32::log2(samples as f32) as usize;
    for iteration in 0..required_iterations {
        let params = SingleIterParams::new(iteration, required_iterations);

        // TODO parallelise
        // run_par!(|| {
        //     iter_range_par!(0, out_element_num).for_each(|id| {

        for sample_id in 0..samples {
            let k_even = sample_id & params.even_mask;

            // TODO improve sample odd
            let k_odd = k_even + (samples >> (1 + params.iteration));
            let twiddle: (E, E) = get_twiddle_factor(&params, sample_id);
            
            // TODO parallel here
            for b in 0..batch_size {
                /*
                   Where E_k are even-indexed inputs and O_k are odd-indexed inputs:
                       X_k = E_k + twiddle_factor * O_k

                   Taking real and imaginary parts:
                       Re(X_k) = Re(E_k) + Re(twiddle_factor) * Re(O_k) - Im(twiddle_factor) * Im(O_k)
                       Im(X_k) = Im(E_k) + Re(twiddle_factor) * Im(O_k) + Im(twiddle_factor) * Re(O_k)

                   See https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#The_radix-2_DIT_case
                */

                unsafe {
                    // Alternate between the array being written to and from.
                    let (x_ref, x_hat_ref) = match params.iteration % 2 {
                        0 => (x1_unsafe_ref.get(), x2_unsafe_ref.get()),
                        1 => (x2_unsafe_ref.get(), x1_unsafe_ref.get()),
                        _ => unreachable!(),
                    };

                    let e_k = (x_ref[(b, k_even, 0)], x_ref[(b, k_even, 1)]);
                    let o_k = (x_ref[(b, k_odd, 0)], x_ref[(b, k_odd, 1)]);

                    let x_k = (
                        e_k.0 + twiddle.0 * o_k.0 - twiddle.1 * o_k.1,
                        e_k.1 + twiddle.0 * o_k.1 + twiddle.1 * o_k.0,
                    );

                    // println!("{:?}", x_k);
                    println!("{:?}", twiddle);

                    x_hat_ref[(b, sample_id, 0)] = x_k.0;
                    x_hat_ref[(b, sample_id, 1)] = x_k.1;
                }
            }
        }
    }

    unsafe {
        let x_hat_ref = match required_iterations % 2 {
            0 => x1_unsafe_ref.get(),
            1 => x2_unsafe_ref.get(),
            _ => unreachable!(),
        };

        let output_ref = out_shared_ref.get();

        // Running the FFT algorithm like this results in a bit-reversed ordered
        //  output. i.e. the element 000, 001, ..., 110, 111 are now sorted if
        //  they were actually 000, 100, ..., 011, 111. On the last step, undo this
        //  mapping.
        for sample_id in 0..samples {
            for b in 0..batch_size {
                let sample_id_bit_rev = reverse_bits(sample_id, required_iterations);
                output_ref[(b, sample_id_bit_rev, 0)] = x_hat_ref[(b, sample_id, 0)];
                output_ref[(b, sample_id_bit_rev, 1)] = x_hat_ref[(b, sample_id, 1)];
            }
        }
    }

    NdArrayTensor::new(output.into_dyn().into_shared())
}

fn get_twiddle_factor<E: burn_tensor::Element>(
    params: &SingleIterParams,
    sample_id: usize,
) -> (E, E) {
    // Indices are bit reversed at each iteration, but there's also a different
    //  number of values at each iteration. e.g., for the 3 iterations of an 8
    //  length transform the mapping from sample_id to k is as follows:
    //  0: [0, 1, 0, 1, 0, 1, 0, 1]
    //  1: [0, 0, 2, 2, 1, 1, 3, 3]
    //  2: [0, 4, 2, 6, 1, 7, 3, 5]
    let k = reverse_bits(
        sample_id >> (params.remaining_iterations),
        params.iteration,
    ) as f64;

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
