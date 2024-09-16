use burn_common::{iter_par, run_par};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, DeformConvOptions},
    ElementConversion,
};
use core::ops::AddAssign;
use ndarray::{s, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayViewMut2, Axis, Dim};

use crate::{
    element::QuantElement, ops::padding::apply_padding_4d, FloatNdArrayElement, NdArrayTensor,
};

use super::matmul::matmul;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn deform_im2col_kernel<F: FloatNdArrayElement>(
    out_y: usize,
    out_x: usize,
    input: ArrayView2<F>,
    offset: ArrayView3<F>,
    mask: Option<ArrayView2<F>>,
    mut columns: ArrayViewMut2<F>,
    args: DeformConvOptions<2>,
    (kernel_h, kernel_w): (usize, usize),
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let (height, width) = input.dim();

    for kernel_y in 0..kernel_h {
        for kernel_x in 0..kernel_w {
            let mask_value = mask
                .map(|it| it[[kernel_y, kernel_x]])
                .unwrap_or_else(|| F::from_elem(1.0));

            let offset = offset.slice(s![kernel_y, kernel_x, ..]);
            let y = F::from_elem(out_y * args.stride[0] + kernel_y * args.dilation[0])
                - F::from_elem(args.padding[0])
                + offset[0];
            let x = F::from_elem(out_x * args.stride[1] + kernel_x * args.dilation[1])
                - F::from_elem(args.padding[1])
                + offset[1];

            let interpolated = bilinear_interpolate(&input, height, width, y, x);

            columns[[kernel_y, kernel_x]] = mask_value * interpolated;
        }
    }
}

fn bilinear_interpolate<F: FloatNdArrayElement>(
    input: &ArrayView2<F>,
    height: usize,
    width: usize,
    y: F,
    x: F,
) -> F {
    // To simplify code
    let y = y.to_f32();
    let x = x.to_f32();

    let mut result = F::from_elem(0.0);
    if y > -1.0 && height as f32 > y && x > -1.0 && width as f32 > x {
        let y_low = f32::floor(y);
        let x_low = f32::floor(x);
        let y_high = (y_low + 1.) as usize;
        let x_high = (x_low + 1.) as usize;

        let zero = F::from_elem(0.0);
        let v1: F = if y_low >= 0. && x_low >= 0. {
            input[[y_low as usize, x_low as usize]]
        } else {
            zero
        };
        let v2: F = if y_low >= 0. && x_high < width {
            input[[y_low as usize, x_high]]
        } else {
            zero
        };
        let v3: F = if y_high < height && x_low >= 0. {
            input[[y_high, x_low as usize]]
        } else {
            zero
        };
        let v4: F = if y_high < height && x_high < width {
            input[[y_high, x_high]]
        } else {
            zero
        };

        let l_y = y - y_low;
        let l_x = x - x_low;
        let h_y = 1.0 - l_y;
        let h_x = 1.0 - l_x;

        let w1 = F::from_elem(h_y * h_x);
        let w2 = F::from_elem(h_y * l_x);
        let w3 = F::from_elem(l_y * h_x);
        let w4 = F::from_elem(l_y * l_x);

        result = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    }
    result
}

pub(crate) fn deform_conv2d<F: FloatNdArrayElement, Q: QuantElement>(
    input: NdArrayTensor<F, 4>,
    offset: NdArrayTensor<F, 4>,
    weight: NdArrayTensor<F, 4>,
    mask: Option<NdArrayTensor<F, 4>>,
    bias: Option<NdArrayTensor<F, 1>>,
    options: DeformConvOptions<2>,
) -> NdArrayTensor<F, 4> {
    let [batch_size, _, in_height, in_width] = input.shape().dims;
    let [out_channels, _, kernel_h, kernel_w] = weight.shape().dims;
    let groups = options.weight_groups;

    let weight = weight.array.as_standard_layout();

    let out_h = calculate_conv_output_size(
        kernel_h,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );
    let out_dims = (out_h, out_w);

    let columns =
        deform_im2col::<F, Q>(input, offset, mask, options, out_dims, (kernel_h, kernel_w));

    let (col_size_0, col_size_1) = columns.dim();
    let col_size_0 = col_size_0 / groups;
    let out_c_per_group = out_channels / groups;

    let mut out = Array3::zeros(Dim([groups, out_c_per_group, col_size_1]));
    let weight = weight
        .to_shape((groups, out_c_per_group, col_size_0))
        .unwrap();
    let columns = columns.to_shape((groups, col_size_0, col_size_1)).unwrap();

    for group in 0..groups {
        let weight = weight.index_axis(Axis(0), group);
        let columns = columns.index_axis(Axis(0), group);

        let values = matmul(
            NdArrayTensor::<_, 2>::new(weight.to_owned().into_dyn().into_shared()),
            NdArrayTensor::<_, 2>::new(columns.to_owned().into_dyn().into_shared()),
        );
        out.index_axis_mut(Axis(0), group).assign(&values.array);
    }

    let mut out = out
        .into_shape_with_order((out_channels, batch_size, out_h, out_w))
        .unwrap();
    out.swap_axes(0, 1);

    if let Some(bias) = bias {
        let bias = bias.array.to_shape((1, out_channels, 1, 1)).unwrap();
        out.add_assign(&bias);
    }

    NdArrayTensor::new(out.into_dyn().into_shared())
}

pub(crate) fn deform_im2col<F: FloatNdArrayElement, Q: QuantElement>(
    input: NdArrayTensor<F, 4>,
    offset: NdArrayTensor<F, 4>,
    mask: Option<NdArrayTensor<F, 4>>,
    args: DeformConvOptions<2>,
    out_dims: (usize, usize),
    kernel_dims: (usize, usize),
) -> Array2<F> {
    let [batch_size, in_channels, _, _] = input.shape().dims;
    let (kernel_h, kernel_w) = kernel_dims;
    let (out_h, out_w) = out_dims;
    let channels_per_offset_group = in_channels / args.offset_groups;

    let x = apply_padding_4d::<F, Q>(input, args.padding, 0i32.elem()).array;
    let offsets = offset
        .array
        .to_shape([
            batch_size,
            args.offset_groups * kernel_h * kernel_w * 2,
            out_h,
            out_w,
        ])
        .unwrap();
    let mask = mask.as_ref().map(|mask| {
        mask.array
            .to_shape([
                batch_size,
                args.offset_groups,
                kernel_h,
                kernel_w,
                out_h,
                out_w,
            ])
            .unwrap()
    });

    // Convert inputs from dynamic indexes to static to improve perf.
    let input = x.into_dimensionality::<ndarray::Ix4>().unwrap();
    let offsets = offsets.into_dimensionality::<ndarray::Ix4>().unwrap();
    let mask = mask.map(|mask| mask.into_dimensionality::<ndarray::Ix6>().unwrap());
    let mut columns = Array4::zeros(Dim([
        in_channels,
        kernel_h,
        kernel_w,
        batch_size * out_h * out_w,
    ]));

    let groups = args.offset_groups;

    run_par!(|| {
        iter_par!(columns.axis_iter_mut(Axis(3)))
            .enumerate()
            .for_each(|(index, mut columns)| {
                let out_x = index % out_w;
                let out_y = (index / out_w) % out_h;
                let batch = (index / (out_w * out_h)) % batch_size;
                let offsets = offsets.slice(s![batch, .., out_y, out_x]);
                let offsets = offsets.to_shape((groups, kernel_h, kernel_w, 2)).unwrap();
                let mask = mask
                    .as_ref()
                    .map(|it| it.slice(s![batch, .., .., .., out_y, out_x]));
                columns
                    .axis_iter_mut(Axis(0))
                    .enumerate()
                    .for_each(|(in_channel, mut columns)| {
                        let group_index = in_channel / channels_per_offset_group;
                        deform_im2col_kernel(
                            out_y,
                            out_x,
                            input.slice(s![batch, in_channel, .., ..]),
                            offsets.slice(s![group_index, .., .., ..]),
                            mask.as_ref().map(|it| it.slice(s![group_index, .., ..])),
                            columns.view_mut(),
                            args.clone(),
                            kernel_dims,
                        );
                    });
            });
    });

    columns
        // Columns is created here, so we know it's contiguous
        .into_shape_with_order((
            in_channels * kernel_h * kernel_w,
            batch_size * out_h * out_w,
        ))
        .unwrap()
}
