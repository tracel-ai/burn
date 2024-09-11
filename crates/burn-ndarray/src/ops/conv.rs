use burn_common::{iter_par, iter_range_par, run_par};
use burn_tensor::{
    ops::{
        conv::{calculate_conv_output_size, calculate_conv_transpose_output_size},
        ConvOptions, ConvTransposeOptions, DeformConvOptions,
    },
    ElementConversion,
};
use ndarray::{
    s, Array3, Array4, Array5, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3, Axis, Dim,
};

use crate::{
    element::{FloatNdArrayElement, QuantElement},
    ops::padding::{apply_padding_4d, apply_padding_5d},
    sharing::UnsafeSharedRef,
    tensor::NdArrayTensor,
};

#[cfg(not(feature = "std"))]
use num_traits::Float;

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
            let iw = ow * stride_width + kw * dilation_width;
            or[ow] += ir[iw] * k;
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn deform_conv2d_mad_inner<E: FloatNdArrayElement>(
    mut output: ArrayViewMut2<E>,
    x: ArrayView2<E>,
    offset: (ArrayView2<E>, ArrayView2<E>),
    weight: E,
    mask: Option<ArrayView2<E>>,
    kernel_pos: (usize, usize),
    out_dims: (usize, usize),
    stride: (usize, usize),
    dilation: (usize, usize),
) {
    let (kernel_y, kernel_x) = kernel_pos;
    let (offset_y, offset_x) = offset;
    let (out_width, out_height) = out_dims;
    let (stride_width, stride_height) = stride;
    let (dilation_width, dilation_height) = dilation;

    for out_y in 0..out_height {
        // Construct a sub-slice view of the input row.
        // This is done upfront so that rustc does not have to emit bounds checks
        // in the hot loop below.
        let index_y = out_y * stride_height + kernel_y * dilation_height;
        let offset_y_row = offset_y.row(out_y).to_slice().unwrap();
        let offset_x_row = offset_x.row(out_y).to_slice().unwrap();

        // Ditto. Construct a sub-slice view of the output row, and explicitly specify
        // the bounds upfront as 0..out_width so that rustc can make the assumption
        // that all accesses are in-bounds in the below loop.
        let mut output_row = output.row_mut(out_y);
        let output_row = &mut output_row.as_slice_mut().unwrap()[0..out_width];

        #[allow(clippy::needless_range_loop)]
        for out_x in 0..out_width {
            let index_x = out_x * stride_width + kernel_x * dilation_width;
            let offset_y = offset_y_row[out_x];
            let offset_x = offset_x_row[out_x];

            let index_y = E::from_usize(index_y).unwrap() + offset_y;
            let index_x = E::from_usize(index_x).unwrap() + offset_x;

            let mask = mask
                .map(|mask| mask[[out_y, out_x]])
                .unwrap_or(E::from_elem(1.0));

            output_row[out_x] += bilinear_interpolate(x, index_y, index_x) * weight * mask;
        }
    }
}

fn bilinear_interpolate<E: FloatNdArrayElement>(image: ArrayView2<E>, y: E, x: E) -> E {
    let x = x.to_f64();
    let y = y.to_f64();
    let (height, width) = image.dim();
    let height = height as f64;
    let width = width as f64;

    if y <= -1.0 || y > height || x <= -1.0 || x > width {
        return E::from_elem(0.0);
    }

    let y_low = y.floor();
    let x_low = x.floor();
    let y_high = y_low + 1.0;
    let x_high = x_low + 1.0;

    let mut v1 = 0.0;
    let mut v2 = 0.0;
    let mut v3 = 0.0;
    let mut v4 = 0.0;
    if y_low >= 0.0 && x_low >= 0.0 {
        v1 = image[[y_low as usize, x_low as usize]].to_f64()
    };
    if y_low >= 0.0 && x_high <= width - 1.0 {
        v2 = image[[y_low as usize, x_high as usize]].to_f64()
    };
    if y_high <= height - 1.0 && x_low >= 0.0 {
        v3 = image[[y_high as usize, x_low as usize]].to_f64()
    };
    if y_high <= height - 1.0 && x_high <= width - 1.0 {
        v4 = image[[y_high as usize, x_high as usize]].to_f64()
    };

    let l_y = y - y_low;
    let l_x = x - x_low;
    let h_y = 1.0 - l_y;
    let h_x = 1.0 - l_x;

    let w1 = h_y * h_x;
    let w2 = h_y * l_x;
    let w3 = l_y * h_x;
    let w4 = l_y * l_x;

    E::from_elem(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)
}

#[inline(always)]
fn conv3d_mad_inner<E: FloatNdArrayElement>(
    mut output: ArrayViewMut3<E>,
    x: ArrayView3<E>,
    k: E,
    k_xyz: (usize, usize, usize),
    out_xyz: (usize, usize, usize),
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
) {
    let (kd, kh, kw) = k_xyz;
    let (out_width, out_height, out_depth) = out_xyz;
    let (stride_width, stride_height, stride_depth) = stride;
    let (dilation_width, dilation_height, dilation_depth) = dilation;

    for od in 0..out_depth {
        let id = od * stride_depth + kd * dilation_depth;

        for oh in 0..out_height {
            let ih = oh * stride_height + kh * dilation_height;

            // Construct a sub-slice view of the input row.
            // This is done upfront so that rustc does not have to emit bounds checks
            // in the hot loop below.
            let ir = x.slice(s![id, ih, ..]).to_slice().unwrap();

            // Ditto. Construct a sub-slice view of the output row, and explicitly specify
            // the bounds upfront as 0..out_width so that rustc can make the assumption
            // that all accesses are in-bounds in the below loop.
            let or = &mut output
                .slice_mut(s![od, oh, 0..out_width])
                .into_slice()
                .unwrap()[0..out_width];

            #[allow(clippy::needless_range_loop)]
            for ow in 0..out_width {
                let iw = ow * stride_width + kw * dilation_width;
                or[ow] += ir[iw] * k;
            }
        }
    }
}

pub(crate) fn conv2d<E: FloatNdArrayElement, Q: QuantElement>(
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

    let x = apply_padding_4d::<E, Q>(x, options.padding, 0i32.elem()).array;

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
        .to_shape([batch_size, out_channels, out_height, out_width])
        .unwrap()
        .into_dyn()
        .into_shared();

    NdArrayTensor::new(output)
}

pub(crate) fn deform_conv2d<E: FloatNdArrayElement, Q: QuantElement>(
    x: NdArrayTensor<E, 4>,
    offset: NdArrayTensor<E, 4>,
    weight: NdArrayTensor<E, 4>,
    mask: Option<NdArrayTensor<E, 4>>,
    bias: Option<NdArrayTensor<E, 1>>,
    options: DeformConvOptions<2>,
) -> NdArrayTensor<E, 4> {
    let [dilation_height, dilation_width] = options.dilation;
    let [padding_height, padding_width] = options.padding;
    let [stride_height, stride_width] = options.stride;
    let [batch_size, in_channels, in_height, in_width] = x.shape().dims;
    let [out_channels, in_channels_per_group, kernel_height, kernel_width] = weight.shape().dims;

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

    let x = apply_padding_4d::<E, Q>(x, options.padding, 0i32.elem()).array;
    let offsets = offset
        .array
        .into_shape([
            batch_size,
            options.offset_groups * kernel_height * kernel_width * 2,
            out_height,
            out_width,
        ])
        .unwrap();
    let mask = mask.map(|mask| {
        mask.array
            .into_shape([
                batch_size,
                options.offset_groups * kernel_height * kernel_width,
                out_height,
                out_width,
            ])
            .unwrap()
    });

    // Convert inputs from dynamic indexes to static to improve perf.
    let x = x.into_dimensionality::<ndarray::Ix4>().unwrap();
    let offsets = offsets.into_dimensionality::<ndarray::Ix4>().unwrap();
    let weights = weight.array.into_dimensionality::<ndarray::Ix4>().unwrap();
    let mask = mask.map(|mask| mask.into_dimensionality::<ndarray::Ix4>().unwrap());

    let mut output = Array3::zeros(Dim([batch_size * out_channels, out_height, out_width]));
    let channels_per_offset_group = in_channels / options.offset_groups;

    run_par!(|| {
        iter_par!(output.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(
                #[inline(never)]
                |(index, mut output)| {
                    let batch = index / out_channels;
                    let out_channel = index % out_channels;
                    let weight_group = index % options.weight_groups;

                    for in_channel in (in_channels_per_group * weight_group)
                        ..(in_channels_per_group * (weight_group + 1))
                    {
                        let weight_in_channel = in_channel - (weight_group * in_channels_per_group);
                        let offset_group = in_channel / channels_per_offset_group;

                        let x = x.slice(s![batch, in_channel, .., ..]);
                        let offset_kernel = offsets.slice(s![batch, .., .., ..]);
                        let kernel = weights.slice(s![out_channel, weight_in_channel, .., ..]);
                        let mask_kernel =
                            mask.as_ref().map(|mask| mask.slice(s![batch, .., .., ..]));

                        for kernel_y in 0..kernel_height {
                            for kernel_x in 0..kernel_width {
                                let mask_index = offset_group * (kernel_width * kernel_height)
                                    + kernel_y * kernel_width
                                    + kernel_x;
                                let offset_index = mask_index * 2;
                                let offset_y = offset_kernel.slice(s![offset_index, .., ..]);
                                let offset_x = offset_kernel.slice(s![offset_index + 1, .., ..]);
                                let mask = mask_kernel
                                    .as_ref()
                                    .map(|mask| mask.slice(s![mask_index, .., ..]));
                                let weight = kernel[[kernel_y, kernel_x]];

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
                                    deform_conv2d_mad_inner(
                                        output.view_mut(),
                                        x.view(),
                                        (offset_y.view(), offset_x.view()),
                                        weight,
                                        mask.as_ref().map(|mask| mask.view()),
                                        (kernel_y, kernel_x),
                                        (out_width, out_height),
                                        (stride_width, stride_height),
                                        (dilation_width, dilation_height),
                                    );
                                } else {
                                    deform_conv2d_mad_inner(
                                        output.view_mut(),
                                        x.view(),
                                        (offset_y.view(), offset_x.view()),
                                        weight,
                                        mask.as_ref().map(|mask| mask.view()),
                                        (kernel_y, kernel_x),
                                        (out_width, out_height),
                                        (stride_width, stride_height),
                                        (dilation_width, dilation_height),
                                    );
                                }
                            }
                        }
                    }

                    if let Some(bias) = &bias {
                        let bias = bias.array[out_channel];

                        for out_y in 0..out_height {
                            // Get a mutable slice reference to the row we're looping over.
                            // We explicitly define the bounds to 0..out_width so that rustc can make
                            // the assumption that all accesses are in-bounds.
                            let mut out_row = output.row_mut(out_y);
                            let out_row = &mut out_row.as_slice_mut().unwrap()[0..out_width];

                            #[allow(clippy::needless_range_loop)]
                            for out_x in 0..out_width {
                                out_row[out_x] += bias;
                            }
                        }
                    }
                },
            )
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

    let out_height = calculate_conv_transpose_output_size(
        kernel_height,
        stride_height,
        padding_height,
        out_padding_height,
        dilation_height,
        in_height,
    );
    let out_width = calculate_conv_transpose_output_size(
        kernel_width,
        stride_width,
        padding_width,
        out_padding_width,
        dilation_width,
        in_width,
    );

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

pub(crate) fn conv3d<E: FloatNdArrayElement, Q: QuantElement>(
    x: NdArrayTensor<E, 5>,
    weight: NdArrayTensor<E, 5>,
    bias: Option<NdArrayTensor<E, 1>>,
    options: ConvOptions<3>,
) -> NdArrayTensor<E, 5> {
    let [dilation_depth, dilation_height, dilation_width] = options.dilation;
    let [padding_depth, padding_height, padding_width] = options.padding;
    let [stride_depth, stride_height, stride_width] = options.stride;
    let [batch_size, _in_channels, in_depth, in_height, in_width] = x.shape().dims;
    let [out_channels, in_channels, kernel_depth, kernel_height, kernel_width] =
        weight.shape().dims;

    let out_depth = calculate_conv_output_size(
        kernel_depth,
        stride_depth,
        padding_depth,
        dilation_depth,
        in_depth,
    );
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

    let x = apply_padding_5d::<E, Q>(x, options.padding, 0i32.elem()).array;

    // Convert inputs from dynamic indexes to static to improve perf.
    let x = x.into_dimensionality::<ndarray::Ix5>().unwrap();
    let weights = weight.array.into_dimensionality::<ndarray::Ix5>().unwrap();

    let mut output = Array4::zeros(Dim([
        batch_size * out_channels,
        out_depth,
        out_height,
        out_width,
    ]));

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

                        let x = x.slice(s![b, ic, .., .., ..]);
                        let k = weights.slice(s![oc, weight_ic, .., .., ..]);

                        for kd in 0..kernel_depth {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let k = k[[kd, kh, kw]];

                                    // NOTE: This function call is duplicated twice so that the compiler can perform auto-vectorization
                                    // in the case that the stride/dilation is 1.
                                    #[allow(clippy::if_same_then_else)]
                                    if (1, 1, 1, 1, 1, 1)
                                        == (
                                            stride_width,
                                            stride_height,
                                            stride_depth,
                                            dilation_width,
                                            dilation_height,
                                            dilation_depth,
                                        )
                                    {
                                        conv3d_mad_inner(
                                            output.view_mut(),
                                            x.view(),
                                            k,
                                            (kd, kh, kw),
                                            (out_width, out_height, out_depth),
                                            (stride_width, stride_height, stride_depth),
                                            (dilation_width, dilation_height, dilation_depth),
                                        );
                                    } else {
                                        conv3d_mad_inner(
                                            output.view_mut(),
                                            x.view(),
                                            k,
                                            (kd, kh, kw),
                                            (out_width, out_height, out_depth),
                                            (stride_width, stride_height, stride_depth),
                                            (dilation_width, dilation_height, dilation_depth),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    if let Some(bias) = &bias {
                        let bias = bias.array[oc];

                        // Get a mutable iterator to the row we're looping over.
                        let orows = output.rows_mut();
                        for mut or in orows {
                            // We explicitly define the bounds to 0..out_width so that rustc can make
                            // the assumption that all accesses are in-bounds.
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
        .to_shape([batch_size, out_channels, out_depth, out_height, out_width])
        .unwrap()
        .into_dyn()
        .into_shared();

    NdArrayTensor::new(output)
}

pub(crate) fn conv_transpose3d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 5>,
    weight: NdArrayTensor<E, 5>,
    bias: Option<NdArrayTensor<E, 1>>,
    options: ConvTransposeOptions<3>,
) -> NdArrayTensor<E, 5> {
    let [dilation_depth, dilation_height, dilation_width] = options.dilation;
    let [padding_depth, padding_height, padding_width] = options.padding;
    let [stride_depth, stride_height, stride_width] = options.stride;
    let [out_padding_depth, out_padding_height, out_padding_width] = options.padding_out;
    let [batch_size, _in_channels, in_depth, in_height, in_width] = x.shape().dims;
    let [in_channels, out_channels, kernel_depth, kernel_height, kernel_width] =
        weight.shape().dims;

    let out_depth = calculate_conv_transpose_output_size(
        kernel_depth,
        stride_depth,
        padding_depth,
        out_padding_depth,
        dilation_depth,
        in_depth,
    );
    let out_height = calculate_conv_transpose_output_size(
        kernel_height,
        stride_height,
        padding_height,
        out_padding_height,
        dilation_height,
        in_height,
    );
    let out_width = calculate_conv_transpose_output_size(
        kernel_width,
        stride_width,
        padding_width,
        out_padding_width,
        dilation_width,
        in_width,
    );

    let x = x.array;
    let mut output = Array5::zeros(Dim([
        batch_size,
        out_channels * options.groups,
        out_depth,
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
                for id in 0..in_depth {
                    for ih in 0..in_height {
                        for iw in 0..in_width {
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let od = id * stride_depth + kd * dilation_depth;
                                        let oh = ih * stride_height + kh * dilation_height;
                                        let ow = iw * stride_width + kw * dilation_width;

                                        if od >= out_depth + padding_depth
                                            || oh >= out_height + padding_height
                                            || ow >= out_width + padding_width
                                            || od < padding_depth
                                            || oh < padding_height
                                            || ow < padding_width
                                        {
                                            continue;
                                        }

                                        let od = od - padding_depth;
                                        let oh = oh - padding_height;
                                        let ow = ow - padding_width;

                                        output[[b, oc_out, od, oh, ow]] += x[[b, ic, id, ih, iw]]
                                            * weight.array[[ic, oc, kd, kh, kw]];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(bias) = &bias {
                for od in 0..out_depth {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            output[[b, oc_out, od, oh, ow]] += bias.array[oc_out];
                        }
                    }
                }
            }
        });
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}
