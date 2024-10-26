use crate::{element::FloatNdArrayElement, sharing::UnsafeSharedRef, tensor::NdArrayTensor};
use burn_common::{iter_range_par, run_par};

use burn_tensor::ElementConversion;
use ndarray::Array4;

pub(crate) fn avg_pool2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> NdArrayTensor<E> {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();

    let out_height = ((x_height + 2 * padding_height - kernel_height) / stride_height) + 1;
    let out_width = ((x_width + 2 * padding_width - kernel_width) / stride_width) + 1;

    let x = x.array;

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
                    let mut count: E = 0.elem();

                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride_height + kh;
                            let iw = ow * stride_width + kw;

                            if ih >= x_height + padding_height
                                || iw >= x_width + padding_width
                                || ih < padding_height
                                || iw < padding_width
                            {
                                continue;
                            }

                            let ih = ih - padding_height;
                            let iw = iw - padding_width;

                            count += 1.elem();
                            sum_val += x[[b, c, ih, iw]];
                        }
                    }

                    if count_include_pad {
                        count = ((kernel_height * kernel_width) as i32).elem();
                    }

                    output[[b, c, oh, ow]] = sum_val / count;
                }
            }
        })
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}

pub(crate) fn avg_pool2d_backward<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    grad: NdArrayTensor<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
) -> NdArrayTensor<E> {
    let [kernel_height, kernel_width] = kernel_size;
    let [stride_height, stride_width] = stride;
    let [padding_height, padding_width] = padding;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();
    let [_batch_size, _channels, out_height, out_width] = grad.shape().dims();

    let grad = grad.array;

    let mut output_grad = Array4::from_elem((batch_size, channels, x_height, x_width), 0.elem());
    let unsafe_shared_grad = UnsafeSharedRef::new(&mut output_grad);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output_grad = unsafe_shared_grad.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let ih_start = oh * stride_height;
                    let iw_start = ow * stride_width;

                    let ih_end = ih_start + kernel_height;
                    let iw_end = iw_start + kernel_width;

                    let ih_start = usize::max(ih_start, padding_height);
                    let iw_start = usize::max(iw_start, padding_width);

                    let ih_end = usize::min(ih_end, x_height + padding_height);
                    let iw_end = usize::min(iw_end, x_width + padding_width);

                    let count = match count_include_pad {
                        true => kernel_width * kernel_height,
                        false => (ih_end - ih_start) * (iw_end - iw_start),
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

    NdArrayTensor::new(output_grad.into_dyn().into_shared())
}
