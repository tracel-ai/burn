use crate::{
    element::{FloatNdArrayElement, IntNdArrayElement},
    ops::padding::apply_padding_4d,
    sharing::UnsafeSharedRef,
    tensor::NdArrayTensor,
};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{ElementConversion, TensorMetadata};
use ndarray::Array4;

pub(crate) fn max_pool2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> NdArrayTensor<E> {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();
    let inf = (-f32::INFINITY).elem::<E>();

    let out_height = ((x_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1)
        / stride_height)
        + 1;
    let out_width = ((x_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1)
        / stride_width)
        + 1;

    let x = apply_padding_4d::<E>(x, padding, inf).array;

    let mut output = Array4::from_elem((batch_size, channels, out_height, out_width), inf);
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output = unsafe_shared_out.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = inf;

                    for kh in 0..kernel_height {
                        let ih = oh * stride_height + kh * dilation_height;

                        for kw in 0..kernel_width {
                            let iw = ow * stride_width + kw * dilation_width;

                            let val = x[[b, c, ih, iw]];

                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }

                    output[[b, c, oh, ow]] = max_val;
                }
            }
        })
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}

pub(crate) fn max_pool2d_with_indices<E: FloatNdArrayElement, I: IntNdArrayElement>(
    x: NdArrayTensor<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (NdArrayTensor<E>, NdArrayTensor<I>) {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();
    let inf = (-f32::INFINITY).elem::<E>();

    let out_height = ((x_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1)
        / stride_height)
        + 1;
    let out_width = ((x_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1)
        / stride_width)
        + 1;

    let x = apply_padding_4d::<E>(x, padding, inf).array;

    let mut output = Array4::from_elem((batch_size, channels, out_height, out_width), inf);
    let mut indices = Array4::<I>::zeros((batch_size, channels, out_height, out_width));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);
    let unsafe_shared_indices = UnsafeSharedRef::new(&mut indices);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output = unsafe_shared_out.get();
            let indices = unsafe_shared_indices.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = inf;
                    let mut index = 0;

                    for kh in 0..kernel_height {
                        let ih = oh * stride_height + kh * dilation_height;

                        for kw in 0..kernel_width {
                            let iw = ow * stride_width + kw * dilation_width;
                            let val = x[[b, c, ih, iw]];

                            if val > max_val {
                                max_val = val;

                                let ih = ih as i64 - padding_height as i64;
                                let iw = iw as i64 - padding_width as i64;

                                index = ih * x_height as i64 + iw;
                            }
                        }
                    }

                    output[[b, c, oh, ow]] = max_val;
                    indices[[b, c, oh, ow]] = index.elem();
                }
            }
        })
    });

    let output = NdArrayTensor::new(output.into_dyn().into_shared());
    let indices = NdArrayTensor::new(indices.into_dyn().into_shared());

    (output, indices)
}

pub(crate) fn max_pool2d_backward<E: FloatNdArrayElement, I: IntNdArrayElement>(
    x: NdArrayTensor<E>,
    _kernel_size: [usize; 2],
    _stride: [usize; 2],
    _padding: [usize; 2],
    _dilation: [usize; 2],
    output_grad: NdArrayTensor<E>,
    indices: NdArrayTensor<I>,
) -> NdArrayTensor<E> {
    let [_batch_size, _channels, height, width] = output_grad.shape().dims();
    let [batch_size, channels, height_x, width_x] = x.shape().dims();

    let output_grad = output_grad.array;
    let indices = indices.array;

    let mut output = Array4::zeros((batch_size, channels, height_x, width_x));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let output = unsafe_shared_out.get();

            for h in 0..height {
                for w in 0..width {
                    let index = indices[[b, c, h, w]].elem::<i64>();
                    let grad = output_grad[[b, c, h, w]];

                    let index_h = index as usize / width_x;
                    let index_w = index as usize % width_x;

                    output[[b, c, index_h, index_w]] += grad;
                }
            }
        });
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}
