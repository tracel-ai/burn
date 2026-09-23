use crate::{
    ShapeOps, SharedArray,
    element::{FloatNdArrayElement, IntNdArrayElement},
    iter_range_par,
    ops::padding::{apply_padding_4d, apply_padding_5d},
    run_par,
    sharing::UnsafeSharedRef,
};

use burn_backend::ElementConversion;
use burn_backend::ops::conv::calculate_pool_output_size;
use ndarray::{Array4, Array5};

pub(crate) fn max_pool2d<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> SharedArray<E> {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();
    let inf = (-f32::INFINITY).elem::<E>();

    let out_height = calculate_pool_output_size(
        kernel_height,
        stride_height,
        padding_height,
        dilation_height,
        x_height,
        ceil_mode,
    );
    let out_width = calculate_pool_output_size(
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        x_width,
        ceil_mode,
    );

    // Calculate extra padding needed for ceil_mode
    // The maximum input position accessed is: (out_size - 1) * stride + (kernel_size - 1) * dilation
    // This must be < input_size + 2 * total_padding
    let max_ih =
        (out_height.saturating_sub(1)) * stride_height + (kernel_height - 1) * dilation_height;
    let max_iw = (out_width.saturating_sub(1)) * stride_width + (kernel_width - 1) * dilation_width;
    let padded_height = x_height + 2 * padding_height;
    let padded_width = x_width + 2 * padding_width;
    let extra_pad_h = max_ih.saturating_sub(padded_height.saturating_sub(1));
    let extra_pad_w = max_iw.saturating_sub(padded_width.saturating_sub(1));
    let total_padding = [padding_height + extra_pad_h, padding_width + extra_pad_w];

    let x = apply_padding_4d::<E>(x, total_padding, inf);

    // Offset to account for extra padding (extra_pad is added on both sides by apply_padding_4d)
    let offset_h = extra_pad_h;
    let offset_w = extra_pad_w;

    let mut output = Array4::from_elem((batch_size, channels, out_height, out_width), inf);
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let mut output = unsafe_shared_out.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = inf;

                    for kh in 0..kernel_height {
                        let ih = offset_h + oh * stride_height + kh * dilation_height;

                        for kw in 0..kernel_width {
                            let iw = offset_w + ow * stride_width + kw * dilation_width;

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

    output.into_dyn().into_shared()
}

pub(crate) fn max_pool2d_with_indices<E: FloatNdArrayElement, I: IntNdArrayElement>(
    x: SharedArray<E>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (SharedArray<E>, SharedArray<I>) {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();
    let inf = (-f32::INFINITY).elem::<E>();

    let out_height = calculate_pool_output_size(
        kernel_height,
        stride_height,
        padding_height,
        dilation_height,
        x_height,
        ceil_mode,
    );
    let out_width = calculate_pool_output_size(
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        x_width,
        ceil_mode,
    );

    // Calculate extra padding needed for ceil_mode
    let max_ih =
        (out_height.saturating_sub(1)) * stride_height + (kernel_height - 1) * dilation_height;
    let max_iw = (out_width.saturating_sub(1)) * stride_width + (kernel_width - 1) * dilation_width;
    let padded_height = x_height + 2 * padding_height;
    let padded_width = x_width + 2 * padding_width;
    let extra_pad_h = max_ih.saturating_sub(padded_height.saturating_sub(1));
    let extra_pad_w = max_iw.saturating_sub(padded_width.saturating_sub(1));
    let total_padding = [padding_height + extra_pad_h, padding_width + extra_pad_w];

    let x = apply_padding_4d::<E>(x, total_padding, inf);

    // Offset to account for extra padding
    let offset_h = extra_pad_h;
    let offset_w = extra_pad_w;

    let mut output = Array4::from_elem((batch_size, channels, out_height, out_width), inf);
    let mut indices = Array4::<I>::zeros((batch_size, channels, out_height, out_width));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);
    let unsafe_shared_indices = UnsafeSharedRef::new(&mut indices);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let mut output = unsafe_shared_out.get();
            let mut indices = unsafe_shared_indices.get();

            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = inf;
                    let mut index = 0;

                    for kh in 0..kernel_height {
                        let ih = offset_h + oh * stride_height + kh * dilation_height;

                        for kw in 0..kernel_width {
                            let iw = offset_w + ow * stride_width + kw * dilation_width;
                            let val = x[[b, c, ih, iw]];

                            if val > max_val {
                                max_val = val;

                                // Calculate index in original (unpadded) input
                                let ih_orig = ih as i64 - (total_padding[0]) as i64;
                                let iw_orig = iw as i64 - (total_padding[1]) as i64;

                                // Clamp to valid range for index calculation
                                let ih_clamped = ih_orig.max(0).min(x_height as i64 - 1);
                                let iw_clamped = iw_orig.max(0).min(x_width as i64 - 1);

                                index = ih_clamped * x_width as i64 + iw_clamped;
                            }
                        }
                    }

                    output[[b, c, oh, ow]] = max_val;
                    indices[[b, c, oh, ow]] = index.elem();
                }
            }
        })
    });

    let output = output.into_dyn().into_shared();
    let indices = indices.into_dyn().into_shared();

    (output, indices)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn max_pool2d_backward<E: FloatNdArrayElement, I: IntNdArrayElement>(
    x: SharedArray<E>,
    _kernel_size: [usize; 2],
    _stride: [usize; 2],
    _padding: [usize; 2],
    _dilation: [usize; 2],
    _ceil_mode: bool,
    output_grad: SharedArray<E>,
    indices: SharedArray<I>,
) -> SharedArray<E> {
    let [_batch_size, _channels, height, width] = output_grad.shape().dims();
    let [batch_size, channels, height_x, width_x] = x.shape().dims();

    let output_grad = output_grad;
    let indices = indices;

    let mut output = Array4::zeros((batch_size, channels, height_x, width_x));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let b = k / channels;
            let c = k % channels;

            let mut output = unsafe_shared_out.get();

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

    output.into_dyn().into_shared()
}

pub(crate) fn max_pool3d<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> SharedArray<E> {
    let [kernel_depth, kernel_height, kernel_width] = kernel_size;
    let [padding_depth, padding_height, padding_width] = padding;
    let [stride_depth, stride_height, stride_width] = stride;
    let [dilation_depth, dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_depth, x_height, x_width] = x.shape().dims();
    let inf = (-f32::INFINITY).elem::<E>();

    let out_depth = calculate_pool_output_size(
        kernel_depth,
        stride_depth,
        padding_depth,
        dilation_depth,
        x_depth,
        ceil_mode,
    );
    let out_height = calculate_pool_output_size(
        kernel_height,
        stride_height,
        padding_height,
        dilation_height,
        x_height,
        ceil_mode,
    );
    let out_width = calculate_pool_output_size(
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        x_width,
        ceil_mode,
    );

    // Calculate extra padding needed for ceil_mode
    let max_id = (out_depth.saturating_sub(1)) * stride_depth + (kernel_depth - 1) * dilation_depth;
    let max_ih =
        (out_height.saturating_sub(1)) * stride_height + (kernel_height - 1) * dilation_height;
    let max_iw = (out_width.saturating_sub(1)) * stride_width + (kernel_width - 1) * dilation_width;
    let padded_depth = x_depth + 2 * padding_depth;
    let padded_height = x_height + 2 * padding_height;
    let padded_width = x_width + 2 * padding_width;
    let extra_pad_d = max_id.saturating_sub(padded_depth.saturating_sub(1));
    let extra_pad_h = max_ih.saturating_sub(padded_height.saturating_sub(1));
    let extra_pad_w = max_iw.saturating_sub(padded_width.saturating_sub(1));
    let total_padding = [
        padding_depth + extra_pad_d,
        padding_height + extra_pad_h,
        padding_width + extra_pad_w,
    ];

    let x = apply_padding_5d::<E>(x, total_padding, inf);

    // Offset to account for extra padding (extra_pad is added on both sides by apply_padding_5d)
    let offset_d = extra_pad_d;
    let offset_h = extra_pad_h;
    let offset_w = extra_pad_w;

    let mut output = Array5::from_elem(
        (batch_size, channels, out_depth, out_height, out_width),
        inf,
    );

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val = inf;

                        for kd in 0..kernel_depth {
                            let id = offset_d + od * stride_depth + kd * dilation_depth;

                            for kh in 0..kernel_height {
                                let ih = offset_h + oh * stride_height + kh * dilation_height;

                                for kw in 0..kernel_width {
                                    let iw = offset_w + ow * stride_width + kw * dilation_width;

                                    let val = x[[b, c, id, ih, iw]];

                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        output[[b, c, od, oh, ow]] = max_val;
                    }
                }
            }
        }
    }

    output.into_dyn().into_shared()
}

pub(crate) fn max_pool3d_with_indices<E: FloatNdArrayElement, I: IntNdArrayElement>(
    x: SharedArray<E>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    ceil_mode: bool,
) -> (SharedArray<E>, SharedArray<I>) {
    let [kernel_depth, kernel_height, kernel_width] = kernel_size;
    let [padding_depth, padding_height, padding_width] = padding;
    let [stride_depth, stride_height, stride_width] = stride;
    let [dilation_depth, dilation_height, dilation_width] = dilation;
    let [batch_size, channels, x_depth, x_height, x_width] = x.shape().dims();
    let inf = (-f32::INFINITY).elem::<E>();

    let out_depth = calculate_pool_output_size(
        kernel_depth,
        stride_depth,
        padding_depth,
        dilation_depth,
        x_depth,
        ceil_mode,
    );
    let out_height = calculate_pool_output_size(
        kernel_height,
        stride_height,
        padding_height,
        dilation_height,
        x_height,
        ceil_mode,
    );
    let out_width = calculate_pool_output_size(
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        x_width,
        ceil_mode,
    );

    // Calculate extra padding needed for ceil_mode
    let max_id = (out_depth.saturating_sub(1)) * stride_depth + (kernel_depth - 1) * dilation_depth;
    let max_ih =
        (out_height.saturating_sub(1)) * stride_height + (kernel_height - 1) * dilation_height;
    let max_iw = (out_width.saturating_sub(1)) * stride_width + (kernel_width - 1) * dilation_width;
    let padded_depth = x_depth + 2 * padding_depth;
    let padded_height = x_height + 2 * padding_height;
    let padded_width = x_width + 2 * padding_width;
    let extra_pad_d = max_id.saturating_sub(padded_depth.saturating_sub(1));
    let extra_pad_h = max_ih.saturating_sub(padded_height.saturating_sub(1));
    let extra_pad_w = max_iw.saturating_sub(padded_width.saturating_sub(1));
    let total_padding = [
        padding_depth + extra_pad_d,
        padding_height + extra_pad_h,
        padding_width + extra_pad_w,
    ];

    let x = apply_padding_5d::<E>(x, total_padding, inf);

    // Offset to account for extra padding
    let offset_d = extra_pad_d;
    let offset_h = extra_pad_h;
    let offset_w = extra_pad_w;

    let mut output = Array5::from_elem(
        (batch_size, channels, out_depth, out_height, out_width),
        inf,
    );
    let mut indices = Array5::<I>::zeros((batch_size, channels, out_depth, out_height, out_width));

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val = inf;
                        let mut index = 0i64;

                        for kd in 0..kernel_depth {
                            let id = offset_d + od * stride_depth + kd * dilation_depth;

                            for kh in 0..kernel_height {
                                let ih = offset_h + oh * stride_height + kh * dilation_height;

                                for kw in 0..kernel_width {
                                    let iw = offset_w + ow * stride_width + kw * dilation_width;
                                    let val = x[[b, c, id, ih, iw]];

                                    if val > max_val {
                                        max_val = val;

                                        // Calculate index in original (unpadded) input
                                        let id_orig = id as i64 - total_padding[0] as i64;
                                        let ih_orig = ih as i64 - total_padding[1] as i64;
                                        let iw_orig = iw as i64 - total_padding[2] as i64;

                                        // Clamp to valid range for index calculation
                                        let id_clamped = id_orig.max(0).min(x_depth as i64 - 1);
                                        let ih_clamped = ih_orig.max(0).min(x_height as i64 - 1);
                                        let iw_clamped = iw_orig.max(0).min(x_width as i64 - 1);

                                        index = id_clamped * (x_height as i64 * x_width as i64)
                                            + ih_clamped * x_width as i64
                                            + iw_clamped;
                                    }
                                }
                            }
                        }

                        output[[b, c, od, oh, ow]] = max_val;
                        indices[[b, c, od, oh, ow]] = index.elem();
                    }
                }
            }
        }
    }

    let output = output.into_dyn().into_shared();
    let indices = indices.into_dyn().into_shared();

    (output, indices)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn max_pool3d_backward<E: FloatNdArrayElement, I: IntNdArrayElement>(
    x: SharedArray<E>,
    _kernel_size: [usize; 3],
    _stride: [usize; 3],
    _padding: [usize; 3],
    _dilation: [usize; 3],
    _ceil_mode: bool,
    output_grad: SharedArray<E>,
    indices: SharedArray<I>,
) -> SharedArray<E> {
    let [_batch_size, _channels, out_depth, out_height, out_width] = output_grad.shape().dims();
    let [batch_size, channels, depth_x, height_x, width_x] = x.shape().dims();

    let mut output = Array5::zeros((batch_size, channels, depth_x, height_x, width_x));
    let slice_size = height_x * width_x;

    if slice_size == 0 || width_x == 0 {
        return output.into_dyn().into_shared();
    }

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let index = indices[[b, c, od, oh, ow]].elem::<i64>();
                        let grad = output_grad[[b, c, od, oh, ow]];

                        if index >= 0 {
                            let idx = index as usize;
                            let index_d = idx / slice_size;
                            let rem = idx % slice_size;
                            let index_h = rem / width_x;
                            let index_w = rem % width_x;

                            if index_d < depth_x && index_h < height_x && index_w < width_x {
                                output[[b, c, index_d, index_h, index_w]] += grad;
                            }
                        }
                    }
                }
            }
        }
    }

    output.into_dyn().into_shared()
}
