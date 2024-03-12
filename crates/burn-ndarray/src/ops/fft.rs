use burn_tensor::ElementConversion;
use ndarray::{Array3, Ix3};
#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{
    iter_range_par, run_par, FloatNdArrayElement, NdArray, NdArrayTensor, UnsafeSharedRef,
};

pub(crate) fn fft1d<E: FloatNdArrayElement>(x: NdArrayTensor<E, 3>) -> NdArrayTensor<E, 3> {
    let [batch_size, samples, complex] = x.shape().dims;

    // The input tensor data
    let mut x = x.array.into_owned().into_dimensionality::<Ix3>().unwrap();

    // The output tensor data. In practice, we write back and forth between these
    //  two buffers and return whichever one was last written to.
    let mut x_hat = ndarray::Array3::<E>::zeros((batch_size, samples, 2));

    // Verify shapes
    if complex != 2 {
        panic!("The last dimension must have length 2. (0: real, 1: imaginary).")
    }

    let mut num_fft_iters: usize = 0;
    let mut pow_2_size = samples;
    while pow_2_size > 1 {
        if pow_2_size % 2 != 0 {
            // This FFT implementation is Cooley-Tukey, only supports power of 2.
            panic!(
                "The dimension that the FFT is computed over must be a power of 2, got {}",
                samples
            );
        }
        pow_2_size = pow_2_size / 2;
        num_fft_iters += 1;
    }

    let x_unsafe_ref = UnsafeSharedRef::new(&mut x);
    let x_hat_unsafe_ref = UnsafeSharedRef::new(&mut x_hat);

    for fft_iter in 0..num_fft_iters {
        // Constants for the Nth iteration of FFT
        // TODO move into function

        let fft_iters_remaining = num_fft_iters - fft_iter;
        let transform_width: usize = usize::pow(2, fft_iter as u32);
        // TODO make this samples / width (?)
        let num_transforms = samples >> (1 + fft_iter);

        /*
            `nth_root_of_unity` has the property that

                exp(-j * root) = 1.

            Raising both sides to the power of any integer k,

                exp(-j * root)^k = 1^k
                exp(-j * k * root) = 1

            The nth root is special because it satisfies this.
            Here, "n" is `transform_width`, which is
        */
        let nth_root_of_unity = -2. * std::f64::consts::PI / (transform_width as f64);

        let sign_mask = num_transforms;
        // TODO verify size
        let even_mask = num_transforms ^ 0xFFFF;

        // TODO parallelise
        // run_par!(|| {
        //     iter_range_par!(0, out_element_num).for_each(|id| {

        unsafe {
            let even_iteration = fft_iter % 2 == 0;
            let input_ref;
            let output_ref;
            if even_iteration {
                input_ref = x_unsafe_ref.get();
                output_ref = x_hat_unsafe_ref.get();
            } else {
                input_ref = x_hat_unsafe_ref.get();
                output_ref = x_unsafe_ref.get();
            }

            for sample_id in 0..samples {
                // Calculate FFT weights once per location, per FFT iteration.
                //  (then apply them across batch dimension).

                let sample_even = sample_id & even_mask;
                let sample_odd = sample_even + num_transforms;

                // TODO explain calculation of k.
                let k = reverse_bits(sample_id >> fft_iters_remaining, fft_iter) as f64;
                let complex_angle = k * nth_root_of_unity;

                let sign = {
                    if (sample_id & sign_mask) > 0 {
                        -1.
                    } else {
                        1.
                    }
                };

                // Euler's formula: exp^(ix) = cos(x) + i sin(x)
                // These are both real numbers, but represent coefficients of a + i*b.
                let twiddle = (
                    (sign * f64::cos(complex_angle)).elem::<E>(),
                    (sign * f64::sin(complex_angle)).elem::<E>(),
                );

                // > comment on memory efficiency from before <
                // Copy operation over columns. The matrix is indiced with nearby
                //  being closest in memory so this is by far the quickest. 10x
                //  quicker for 1024x1024 array than doing this op over rows.

                // TODO parallel here
                for b in 0..batch_size {
                    /////////////
                    // Indices //
                    /////////////

                    // Note that indices X are just constant - duplicate result across all rows.
                    // let i_out = y * num_cols + x_buffer;
                    // let i_even = sample_even * num_cols + x_buffer;
                    // let i_odd = sample_odd * num_cols + x_buffer;

                    /*
                       Where E_k are even-indexed inputs and O_k are odd-indexed inputs:
                           X_k = E_k + (sign * twiddle_factor) * O_k

                       Taking real and imaginary parts:
                           Re(X_k) = Re(E_k) + Re(twiddle_factor) * Re(O_k) - Im(twiddle_factor) * Im(O_k)
                           Im(X_k) = Im(E_k) + Re(twiddle_factor) * Im(O_k) + Im(twiddle_factor) * Re(O_k)

                       See https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm#The_radix-2_DIT_case
                    */

                    let e_k = (
                        input_ref[(b, sample_even, 0)],
                        input_ref[(b, sample_even, 1)],
                    );
                    let o_k = (input_ref[(b, sample_odd, 0)], input_ref[(b, sample_odd, 1)]);

                    let x_k = (
                        e_k.0 + twiddle.0 * o_k.0 - twiddle.1 * o_k.1,
                        e_k.1 + twiddle.0 * o_k.1 + twiddle.1 * o_k.0,
                    );

                    output_ref[(b, sample_id, 0)] = x_k.0;
                    output_ref[(b, sample_id, 1)] = x_k.1;

                    // unsafe {
                    //     let output = unsafe_shared_out.get();
                    //     output[(b, c, h, w)] = (p_a + p_b + p_c + p_d).elem();
                    // }
                }
            }
        }
    }

    let mut output = Array3::zeros((batch_size, samples, 2));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    unsafe {
        // Need to use whichever was the last buffer written to, as the output.
        let final_ref;
        if num_fft_iters % 2 == 0 {
            final_ref = x_unsafe_ref.get();
        } else {
            final_ref = x_hat_unsafe_ref.get();
        }

        let output_ref = unsafe_shared_out.get();

        // Finally perform bit reversal from buffer. TODO explain this
        for sample_id in 0..samples {
            for b in 0..batch_size {
                let sample_id_bit_rev = reverse_bits(sample_id, num_fft_iters);
                output_ref[(b, sample_id_bit_rev, 0)] = final_ref[(b, sample_id, 0)];
                output_ref[(b, sample_id_bit_rev, 1)] = final_ref[(b, sample_id, 1)];
            }
        }
    }

    NdArrayTensor::new(output.into_dyn().into_shared())
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
