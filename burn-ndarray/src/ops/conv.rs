use burn_tensor::ElementConversion;
use ndarray::{Array4, Dim};

use crate::{
    element::FloatNdArrayElement, iter_par, ops::padding::apply_padding_4d, run_par,
    sharing::UnsafeSharedRef, tensor::NdArrayTensor,
};

pub(crate) fn conv2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    weight: NdArrayTensor<E, 4>,
    bias: Option<NdArrayTensor<E, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilatation: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [dilatation_height, dilatation_width] = dilatation;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [batch_size, _in_channels, in_height, in_width] = x.shape().dims;
    let [out_channels, in_channels, kernel_height, kernel_width] = weight.shape().dims;

    let out_height = (in_height + 2 * padding_height - dilatation_height * (kernel_height - 1) - 1)
        / stride_height
        + 1;

    let out_width = (in_width + 2 * padding_width - dilatation_width * (kernel_width - 1) - 1)
        / stride_width
        + 1;

    let x = apply_padding_4d(x, padding, 0i32.elem()).array;

    let mut output = Array4::zeros(Dim([batch_size, out_channels, out_height, out_width]));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_par!(0, batch_size * out_channels).for_each(|k| unsafe {
            let b = k / out_channels;
            let oc = k % out_channels;

            let output = unsafe_shared_out.get();

            for ic in 0..in_channels {
                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                let ih = oh * stride_height + kh * dilatation_height;
                                let iw = ow * stride_width + kw * dilatation_width;

                                output[[b, oc, oh, ow]] = output[[b, oc, oh, ow]]
                                    + x[[b, ic, ih, iw]] * weight.array[[oc, ic, kh, kw]];
                            }
                        }
                    }
                }
            }

            if let Some(bias) = &bias {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        output[[b, oc, oh, ow]] = output[[b, oc, oh, ow]] + bias.array[oc];
                    }
                }
            }
        });
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}

pub(crate) fn conv_transpose2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    weight: NdArrayTensor<E, 4>,
    bias: Option<NdArrayTensor<E, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
    out_padding: [usize; 2],
    dilatation: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [dilatation_height, dilatation_width] = dilatation;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [out_padding_height, out_padding_width] = out_padding;
    let [batch_size, _in_channels, in_height, in_width] = x.shape().dims;
    let [out_channels, in_channels, kernel_height, kernel_width] = weight.shape().dims;

    let out_height =
        (in_height - 1) * stride_height + kernel_height + out_padding_height - 2 * padding_height;
    let out_width =
        (in_width - 1) * stride_width + kernel_width + out_padding_width - 2 * padding_width;

    let padding_height = i32::max(
        0,
        (out_height as i32 - in_height as i32 + kernel_height as i32) * stride_height as i32 / 2,
    ) as usize;
    let padding_width = i32::max(
        0,
        (out_width as i32 - in_width as i32 + kernel_width as i32) * stride_width as i32 / 2,
    ) as usize;
    let x = apply_padding_4d(x, [padding_height, padding_width], 0i32.elem()).array;

    let mut output = Array4::zeros(Dim([batch_size, in_channels, out_height, out_width]));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    (0..batch_size * in_channels).for_each(|k| unsafe {
        let b = k / in_channels;
        let ic = k % in_channels;

        let output = unsafe_shared_out.get();

        for oc in 0..out_channels {
            for kh in 0..kernel_height {
                for kw in 0..kernel_width {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let ih = oh * stride_height + kh * dilatation_height;
                            let iw = ow * stride_width + kw * dilatation_width;

                            output[[b, ic, oh, ow]] = output[[b, ic, oh, ow]]
                                + x[[b, ic, ih, iw]]
                                    * weight.array
                                        [[oc, ic, kernel_height - kh - 1, kernel_width - kw - 1]];
                        }
                    }
                }
            }
        }

        if let Some(bias) = &bias {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    output[[b, ic, oh, ow]] = output[[b, ic, oh, ow]] + bias.array[ic];
                }
            }
        }
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}
