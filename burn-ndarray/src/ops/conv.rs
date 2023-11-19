use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions, ConvTransposeOptions},
    ElementConversion,
};
use ndarray::{s, Array3, Array4, ArrayView2, ArrayViewMut2, Axis, Dim};

use crate::{
    element::FloatNdArrayElement, iter_par, iter_range_par, ops::padding::apply_padding_4d,
    run_par, sharing::UnsafeSharedRef, tensor::NdArrayTensor,
};

#[inline(always)]
fn conv2d_mad_inner<E: FloatNdArrayElement>(
    mut output: ArrayViewMut2<E>,
    x: ArrayView2<E>,
    k: E,
    k_xy: (usize, usize),
    out_xy: (usize, usize),
    stride: (usize, usize),
    dilation: (usize, usize),
) {
    let (kh, kw) = k_xy;
    let (out_width, out_height) = out_xy;
    let (stride_width, stride_height) = stride;
    let (dilation_width, dilation_height) = dilation;

    for oh in 0..out_height {
        // Construct a sub-slice view of the input row.
        // This is done upfront so that rustc does not have to emit bounds checks
        // in the hot loop below.
        let ir = x
            .row(oh * stride_height + kh * dilation_height)
            .to_slice()
            .unwrap();

        // Ditto. Construct a sub-slice view of the output row, and explicitly specify
        // the bounds upfront as 0..out_width so that rustc can make the assumption
        // that all accesses are in-bounds in the below loop.
        let mut or = output.row_mut(oh);
        let or = &mut or.as_slice_mut().unwrap()[0..out_width];

        #[allow(clippy::needless_range_loop)]
        for ow in 0..out_width {
            let iw = (ow * stride_width) + (kw * dilation_width);
            or[ow] += ir[iw] * k;
        }
    }
}

pub(crate) fn conv2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    weight: NdArrayTensor<E, 4>,
    bias: Option<NdArrayTensor<E, 1>>,
    options: ConvOptions<2>,
) -> NdArrayTensor<E, 4> {
    let [dilation_height, dilation_width] = options.dilation;
    let [padding_height, padding_width] = options.padding;
    let [stride_height, stride_width] = options.stride;
    let [batch_size, _in_channels, in_height, in_width] = x.shape().dims;
    let [out_channels, in_channels, kernel_height, kernel_width] = weight.shape().dims;

    let out_height = calculate_conv_output_size(
        kernel_height,
        stride_height,
        padding_height,
        dilation_height,
        in_height,
    );
    let out_width = calculate_conv_output_size(
        kernel_width,
        stride_width,
        padding_width,
        dilation_width,
        in_width,
    );

    let x = apply_padding_4d(x, options.padding, 0i32.elem()).array;

    // Convert inputs from dynamic indexes to static to improve perf.
    let x = x.into_dimensionality::<ndarray::Ix4>().unwrap();
    let weights = weight.array.into_dimensionality::<ndarray::Ix4>().unwrap();

    let mut output = Array3::zeros(Dim([batch_size * out_channels, out_height, out_width]));

    run_par!(|| {
        iter_par!(output.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(
                #[inline(never)]
                |(k, mut output)| {
                    let b = k / out_channels;
                    let oc = k % out_channels;
                    let g = k % options.groups;

                    for ic in (in_channels * g)..(in_channels * (g + 1)) {
                        let weight_ic = ic - (g * in_channels);

                        let x = x.slice(s![b, ic, .., ..]);
                        let k = weights.slice(s![oc, weight_ic, .., ..]);

                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let k = k[[kh, kw]];

                                // NOTE: This function call is duplicated twice so that the compiler can perform auto-vectorization
                                // in the case that the stride/dilation is 1.
                                #[allow(clippy::if_same_then_else)]
                                if (1, 1, 1, 1)
                                    == (
                                        stride_width,
                                        stride_height,
                                        dilation_width,
                                        dilation_height,
                                    )
                                {
                                    conv2d_mad_inner(
                                        output.view_mut(),
                                        x.view(),
                                        k,
                                        (kh, kw),
                                        (out_width, out_height),
                                        (stride_width, stride_height),
                                        (dilation_width, dilation_height),
                                    );
                                } else {
                                    conv2d_mad_inner(
                                        output.view_mut(),
                                        x.view(),
                                        k,
                                        (kh, kw),
                                        (out_width, out_height),
                                        (stride_width, stride_height),
                                        (dilation_width, dilation_height),
                                    );
                                }
                            }
                        }
                    }

                    if let Some(bias) = &bias {
                        let bias = bias.array[oc];

                        for oh in 0..out_height {
                            // Get a mutable slice reference to the row we're looping over.
                            // We explicitly define the bounds to 0..out_width so that rustc can make
                            // the assumption that all accesses are in-bounds.
                            let mut or = output.row_mut(oh);
                            let or = &mut or.as_slice_mut().unwrap()[0..out_width];

                            #[allow(clippy::needless_range_loop)]
                            for ow in 0..out_width {
                                or[ow] += bias;
                            }
                        }
                    }
                },
            );
    });

    let output = output
        .into_shape([batch_size, out_channels, out_height, out_width])
        .unwrap()
        .into_dyn()
        .into_shared();

    NdArrayTensor::new(output)
}

pub(crate) fn conv_transpose2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    weight: NdArrayTensor<E, 4>,
    bias: Option<NdArrayTensor<E, 1>>,
    options: ConvTransposeOptions<2>,
) -> NdArrayTensor<E, 4> {
    let [dilation_height, dilation_width] = options.dilation;
    let [padding_height, padding_width] = options.padding;
    let [stride_height, stride_width] = options.stride;
    let [out_padding_height, out_padding_width] = options.padding_out;
    let [batch_size, _in_channels, in_height, in_width] = x.shape().dims;
    let [in_channels, out_channels, kernel_height, kernel_width] = weight.shape().dims;

    let out_height = (in_height - 1) * stride_height
        + dilation_height * (kernel_height - 1)
        + out_padding_height
        - 2 * padding_height
        + 1;
    let out_width =
        (in_width - 1) * stride_width + dilation_width * (kernel_width - 1) + out_padding_width
            - 2 * padding_width
            + 1;

    let x = x.array;
    let mut output = Array4::zeros(Dim([
        batch_size,
        out_channels * options.groups,
        out_height,
        out_width,
    ]));

    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * out_channels * options.groups).for_each(|k| unsafe {
            let b = k / (out_channels * options.groups);
            let oc = k % out_channels;
            let g = k % options.groups;

            let output = unsafe_shared_out.get();

            let oc_out = oc + (out_channels * g);
            let ic_start = g * (in_channels / options.groups);
            let ic_end = ic_start + in_channels / options.groups;

            for ic in ic_start..ic_end {
                for ih in 0..in_height {
                    for iw in 0..in_width {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let oh = ih * stride_height + kh * dilation_height;
                                let ow = iw * stride_width + kw * dilation_width;

                                if oh >= out_height + padding_height
                                    || ow >= out_width + padding_width
                                    || oh < padding_height
                                    || ow < padding_width
                                {
                                    continue;
                                }

                                let oh = oh - padding_height;
                                let ow = ow - padding_width;

                                output[[b, oc_out, oh, ow]] +=
                                    x[[b, ic, ih, iw]] * weight.array[[ic, oc, kh, kw]];
                            }
                        }
                    }
                }
            }

            if let Some(bias) = &bias {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        output[[b, oc_out, oh, ow]] += bias.array[oc_out];
                    }
                }
            }
        });
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}
