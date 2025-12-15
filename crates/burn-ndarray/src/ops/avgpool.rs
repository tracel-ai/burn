use crate::{
    SharedArray, element::FloatNdArrayElement, iter_range_par, run_par, sharing::UnsafeSharedRef,
};

use burn_backend::ElementConversion;
use burn_backend::ops::conv::calculate_pool_output_size;
use ndarray::Array4;

pub(crate) fn avg_pool2d<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> SharedArray<E> {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [batch_size, channels, x_height, x_width] = x.shape().try_into().unwrap();

    let out_height = calculate_pool_output_size(
        kernel_height,
        stride_height,
        padding_height,
        1,
        x_height,
        ceil_mode,
    );
    let out_width = calculate_pool_output_size(
        kernel_width,
        stride_width,
        padding_width,
        1,
        x_width,
        ceil_mode,
    );

    // Padded input bounds (for count_include_pad calculation)
    let padded_height = x_height + 2 * padding_height;
    let padded_width = x_width + 2 * padding_width;

    let mut output = Array4::from_elem((batch_size, channels, out_height, out_width), 0.elem());
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output = unsafe_shared_out.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum_val: E = 0.elem();
                    let mut valid_count = 0usize;
                    let mut padded_count = 0usize;

                    for kh in 0..kernel_height {
                        let ih = oh * stride_height + kh;

                        for kw in 0..kernel_width {
                            let iw = ow * stride_width + kw;

                            // Check if within padded bounds (excludes ceil_mode extensions)
                            if ih < padded_height && iw < padded_width {
                                padded_count += 1;

                                // Check if within valid (non-padding) input bounds
                                if ih >= padding_height
                                    && ih < x_height + padding_height
                                    && iw >= padding_width
                                    && iw < x_width + padding_width
                                {
                                    let ih_valid = ih - padding_height;
                                    let iw_valid = iw - padding_width;
                                    sum_val += x[[b, c, ih_valid, iw_valid]];
                                    valid_count += 1;
                                }
                            }
                        }
                    }

                    // count_include_pad: count positions within padded bounds (not ceil_mode extensions)
                    // !count_include_pad: count only valid (non-padding) positions
                    let count: E = if count_include_pad {
                        (padded_count as i32).elem()
                    } else {
                        (valid_count as i32).elem()
                    };

                    output[[b, c, oh, ow]] = sum_val / count;
                }
            }
        })
    });

    output.into_dyn().into_shared()
}

pub(crate) fn avg_pool2d_backward<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    grad: SharedArray<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    _ceil_mode: bool,
) -> SharedArray<E> {
    let [kernel_height, kernel_width] = kernel_size;
    let [stride_height, stride_width] = stride;
    let [padding_height, padding_width] = padding;
    let [batch_size, channels, x_height, x_width] = x.shape().try_into().unwrap();
    let [_batch_size, _channels, out_height, out_width] = grad.shape().try_into().unwrap();

    // Padded input bounds (for count_include_pad calculation)
    let padded_height = x_height + 2 * padding_height;
    let padded_width = x_width + 2 * padding_width;

    let mut output_grad = Array4::from_elem((batch_size, channels, x_height, x_width), 0.elem());
    let unsafe_shared_grad = UnsafeSharedRef::new(&mut output_grad);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output_grad = unsafe_shared_grad.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let ih_start_kernel = oh * stride_height;
                    let iw_start_kernel = ow * stride_width;

                    let ih_end_kernel = ih_start_kernel + kernel_height;
                    let iw_end_kernel = iw_start_kernel + kernel_width;

                    // Clip to valid input bounds (for gradient distribution)
                    let ih_start = usize::max(ih_start_kernel, padding_height);
                    let iw_start = usize::max(iw_start_kernel, padding_width);
                    let ih_end = usize::min(ih_end_kernel, x_height + padding_height);
                    let iw_end = usize::min(iw_end_kernel, x_width + padding_width);

                    // Calculate count based on count_include_pad
                    let count = if count_include_pad {
                        // Count positions within padded bounds (not ceil_mode extensions)
                        let ih_start_padded = ih_start_kernel;
                        let iw_start_padded = iw_start_kernel;
                        let ih_end_padded = usize::min(ih_end_kernel, padded_height);
                        let iw_end_padded = usize::min(iw_end_kernel, padded_width);
                        (ih_end_padded - ih_start_padded) * (iw_end_padded - iw_start_padded)
                    } else {
                        // Count only valid (non-padding) positions
                        (ih_end - ih_start) * (iw_end - iw_start)
                    };

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            let ih = ih - padding_height;
                            let iw = iw - padding_width;

                            output_grad[[b, c, ih, iw]] +=
                                grad[[b, c, oh, ow]] / (count as i32).elem();
                        }
                    }
                }
            }
        })
    });

    output_grad.into_dyn().into_shared()
}
