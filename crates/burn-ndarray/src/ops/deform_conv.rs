use burn_common::{iter_par, run_par};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, DeformConvOptions},
    TensorMetadata,
};
use core::ops::AddAssign;
use ndarray::{
    s, Array2, Array4, ArrayView2, ArrayView3, ArrayView4, ArrayView6, ArrayViewMut2, Axis, Dim,
    Ix4, Zip,
};
#[cfg(not(feature = "std"))]
use num_traits::Float;

use crate::{FloatNdArrayElement, NdArrayTensor};

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

            let interpolated = bilinear_interpolate(input, height, width, y, x);

            columns[[kernel_y, kernel_x]] = mask_value * interpolated;
        }
    }
}

fn bilinear_interpolate<F: FloatNdArrayElement>(
    input: ArrayView2<F>,
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

pub(crate) fn deform_conv2d<F: FloatNdArrayElement>(
    input: NdArrayTensor<F>,
    offset: NdArrayTensor<F>,
    weight: NdArrayTensor<F>,
    mask: Option<NdArrayTensor<F>>,
    bias: Option<NdArrayTensor<F>>,
    args: DeformConvOptions<2>,
) -> NdArrayTensor<F> {
    let [batch_size, _, in_height, in_width] = input.shape().dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape().dims();
    let groups = args.weight_groups;

    let weight = weight.array.as_standard_layout();

    let out_h = calculate_conv_output_size(
        kernel_h,
        args.stride[0],
        args.padding[0],
        args.dilation[0],
        in_height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        args.stride[1],
        args.padding[1],
        args.dilation[1],
        in_width,
    );
    let out_dims = (out_h, out_w);

    let input = input.array.into_dimensionality::<Ix4>().unwrap();
    let offset = offset.array.into_dimensionality::<Ix4>().unwrap();
    let mask = mask.as_ref().map(|it| {
        it.array
            .to_shape((
                batch_size,
                args.offset_groups,
                kernel_h,
                kernel_w,
                out_h,
                out_w,
            ))
            .unwrap()
    });

    let columns = deform_im2col(
        input.view(),
        offset.view(),
        mask.as_ref().map(|it| it.view()),
        args,
        out_dims,
        (kernel_h, kernel_w),
    );

    let (col_size_0, col_size_1) = columns.dim();
    let col_size_0 = col_size_0 / groups;
    let out_c_per_group = out_channels / groups;

    let weight = weight
        .to_shape((groups, out_c_per_group, col_size_0))
        .unwrap();
    let columns = columns.to_shape((groups, col_size_0, col_size_1)).unwrap();
    let out = matmul(
        NdArrayTensor::new(weight.to_owned().into_dyn().into_shared()),
        NdArrayTensor::new(columns.to_owned().into_dyn().into_shared()),
    );

    let mut out = out
        .array
        .into_shape_with_order((out_channels, batch_size, out_h, out_w))
        .unwrap();
    out.swap_axes(0, 1);

    if let Some(bias) = bias {
        let bias = bias.array.to_shape((1, out_channels, 1, 1)).unwrap();
        out.add_assign(&bias);
    }

    NdArrayTensor::new(out.into_dyn().into_shared())
}

pub(crate) fn deform_im2col<F: FloatNdArrayElement>(
    input: ArrayView4<F>,
    offset: ArrayView4<F>,
    mask: Option<ArrayView6<F>>,
    args: DeformConvOptions<2>,
    out_dims: (usize, usize),
    kernel_dims: (usize, usize),
) -> Array2<F> {
    let (batch_size, in_channels, _, _) = input.dim();
    let (kernel_h, kernel_w) = kernel_dims;
    let (out_h, out_w) = out_dims;
    let channels_per_offset_group = in_channels / args.offset_groups;

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
                let offset = offset.slice(s![batch, .., out_y, out_x]);
                let offset = offset.to_shape((groups, kernel_h, kernel_w, 2)).unwrap();
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
                            offset.slice(s![group_index, .., .., ..]),
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

pub mod backward {
    #[cfg(target_has_atomic = "32")]
    use core::sync::atomic::Ordering;

    use atomic_float::AtomicF32;
    use ndarray::{Array1, Array5, ArrayView4, ArrayView6, Ix4};

    use super::*;

    pub(crate) type DeformConv2dBackward<F> = (
        NdArrayTensor<F>,
        NdArrayTensor<F>,
        NdArrayTensor<F>,
        Option<NdArrayTensor<F>>,
        Option<NdArrayTensor<F>>,
    );

    /// Calculate the [deformable 2D convolution](crate::ops::ModuleOps::deform_conv2d) backward pass using convolutions.
    pub(crate) fn deform_conv2d_backward<F: FloatNdArrayElement>(
        input: NdArrayTensor<F>,
        offset: NdArrayTensor<F>,
        weight: NdArrayTensor<F>,
        mask: Option<NdArrayTensor<F>>,
        bias: Option<NdArrayTensor<F>>,
        out_grad: NdArrayTensor<F>,
        args: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<F> {
        let [batch_size, out_channels, out_h, out_w] = out_grad.shape().dims();
        let [_, _, kernel_h, kernel_w] = weight.shape().dims();
        let groups = args.weight_groups;
        let out_c_per_group = out_channels / groups;
        let col_shape_1 = batch_size * out_h * out_w;
        let mut out_grad = out_grad.array.into_dimensionality::<Ix4>().unwrap();

        let gradient_bias = bias.map(|_| {
            let out_grad = out_grad
                .clone()
                .sum_axis(Axis(0))
                .sum_axis(Axis(1))
                .sum_axis(Axis(1));

            NdArrayTensor::new(out_grad.into_dyn().into_shared())
        });

        out_grad.swap_axes(0, 1);
        let out_grad = out_grad
            .to_shape((groups, out_c_per_group, col_shape_1))
            .unwrap();

        let input = input.array.into_dimensionality::<Ix4>().unwrap();
        let offset = offset.array.into_dimensionality::<Ix4>().unwrap();
        let mask = mask.map(|it| {
            it.array
                .into_shape_with_order((
                    batch_size,
                    args.offset_groups,
                    kernel_h,
                    kernel_w,
                    out_h,
                    out_w,
                ))
                .unwrap()
        });

        let (input_gradient, offset_gradient, mask_gradient) = backward_gradient_inputs(
            input.view(),
            weight,
            offset.view(),
            mask.as_ref().map(|it| it.view()),
            out_grad.view(),
            &args,
            (kernel_h, kernel_w),
        );

        let weight_grad = compute_weight_grad(
            input.view(),
            offset.view(),
            mask.as_ref().map(|it| it.view()),
            out_grad.view(),
            args,
            (kernel_h, kernel_w),
            (out_h, out_w),
        );

        (
            input_gradient,
            offset_gradient,
            weight_grad,
            mask_gradient,
            gradient_bias,
        )
    }

    fn compute_weight_grad<F: FloatNdArrayElement>(
        input: ArrayView4<F>,
        offset: ArrayView4<F>,
        mask: Option<ArrayView6<F>>,
        out_grad: ArrayView3<F>,
        options: DeformConvOptions<2>,
        kernel_dims: (usize, usize),
        out_dims: (usize, usize),
    ) -> NdArrayTensor<F> {
        let in_channels = input.dim().1;
        let (groups, out_c_per_group, _) = out_grad.dim();
        let (kernel_h, kernel_w) = kernel_dims;

        let in_c_per_group = in_channels / groups;

        let columns = deform_im2col(input, offset, mask, options, out_dims, kernel_dims);
        let (col_size_0, col_size_1) = columns.dim();
        let col_size_0 = col_size_0 / groups;

        let mut columns = columns.to_shape((groups, col_size_0, col_size_1)).unwrap();
        columns.swap_axes(1, 2);

        let grad_weight = matmul(
            NdArrayTensor::new(out_grad.to_owned().into_dyn().into_shared()),
            NdArrayTensor::new(columns.to_owned().into_dyn().into_shared()),
        );

        let grad_weight = grad_weight
            .array
            .into_shape_with_order((out_c_per_group * groups, in_c_per_group, kernel_h, kernel_w))
            .unwrap();
        NdArrayTensor::new(grad_weight.into_dyn().into_shared())
    }

    type InputGradients<F> = (NdArrayTensor<F>, NdArrayTensor<F>, Option<NdArrayTensor<F>>);

    fn backward_gradient_inputs<F: FloatNdArrayElement>(
        image: ArrayView4<F>,
        weight: NdArrayTensor<F>,
        offset: ArrayView4<F>,
        mask: Option<ArrayView6<F>>,
        out_grad: ArrayView3<F>,
        args: &DeformConvOptions<2>,
        kernel_dims: (usize, usize),
    ) -> InputGradients<F> {
        let input_shape = image.dim();
        let in_channels = input_shape.1;
        let [out_channels, in_c_per_group, kernel_h, kernel_w] = weight.shape().dims();
        let (batch_size, _, out_h, out_w) = offset.dim();

        let groups = args.weight_groups;
        let out_c_per_group = out_channels / groups;

        let col_shape_0 = in_c_per_group * kernel_h * kernel_w;

        let mut weight = weight
            .array
            .to_shape((groups, out_c_per_group, col_shape_0))
            .unwrap();
        weight.swap_axes(1, 2);
        let columns = matmul(
            NdArrayTensor::new(weight.to_owned().into_dyn().into_shared()),
            NdArrayTensor::new(out_grad.to_owned().into_dyn().into_shared()),
        );

        let columns = columns
            .array
            .to_shape((in_channels, kernel_h, kernel_w, batch_size, out_h, out_w))
            .unwrap();

        let (offset_gradient, mask_gradient) = compute_offset_and_mask_gradient(
            columns.view(),
            image.view(),
            offset,
            mask,
            args,
            kernel_dims,
        );

        let input_gradient =
            compute_input_grad(columns.view(), offset, mask, args, kernel_dims, input_shape);

        (input_gradient, offset_gradient, mask_gradient)
    }

    fn compute_offset_and_mask_gradient<F: FloatNdArrayElement>(
        columns: ArrayView6<F>,
        image: ArrayView4<F>,
        offset: ArrayView4<F>,
        mask: Option<ArrayView6<F>>,
        args: &DeformConvOptions<2>,
        kernel_dims: (usize, usize),
    ) -> (NdArrayTensor<F>, Option<NdArrayTensor<F>>) {
        let (kernel_h, kernel_w) = kernel_dims;
        let (_, in_channels, height, width) = image.dim();
        let (batch_size, offset_channels, out_h, out_w) = offset.dim();
        let offs_groups = args.offset_groups;
        let channels_per_offset_group = in_channels / args.offset_groups;

        let mut grad_offset = Array5::zeros((
            offs_groups,
            kernel_h,
            kernel_w,
            2,
            batch_size * out_h * out_w,
        ));
        let mut grad_mask =
            Array4::zeros((offs_groups, kernel_h, kernel_w, batch_size * out_h * out_w));

        grad_mask
            .axis_iter_mut(Axis(3))
            .zip(grad_offset.axis_iter_mut(Axis(4)))
            .enumerate()
            .for_each(|(index, (mut grad_mask, mut grad_offset))| {
                let out_x = index % out_w;
                let out_y = (index / out_w) % out_h;
                let batch = index / (out_w * out_h);
                let offset = offset.slice(s![batch, .., out_y, out_x]);
                let offset = offset
                    .to_shape((offs_groups, kernel_h, kernel_w, 2))
                    .unwrap();
                let mask: Option<ArrayView3<F>> = mask
                    .as_ref()
                    .map(|mask| mask.slice(s![batch, .., .., .., out_y, out_x]));
                let columns = columns.slice(s![.., .., .., batch, out_y, out_x]);
                let image = image.slice(s![batch, .., .., ..]);

                for ((group, kernel_y, kernel_x), grad_mask) in grad_mask.indexed_iter_mut() {
                    let grad_mask: &mut F = grad_mask;
                    let mut grad_offset = grad_offset.slice_mut(s![group, kernel_y, kernel_x, ..]);
                    let offset = offset.slice(s![group, kernel_y, kernel_x, ..]);
                    let mask = mask.map(|it| it[[group, kernel_y, kernel_x]]);
                    let columns = columns.slice(s![.., kernel_y, kernel_x]);
                    let group_offset = group * channels_per_offset_group;
                    let image = image.slice(s![group_offset.., .., ..]);
                    let y = F::from_elem(out_y * args.stride[0] + kernel_y * args.dilation[0])
                        - F::from_elem(args.padding[0])
                        + offset[0];
                    let x = F::from_elem(out_x * args.stride[1] + kernel_x * args.dilation[1])
                        - F::from_elem(args.padding[1])
                        + offset[1];
                    for (i, grad_offset) in grad_offset.iter_mut().enumerate() {
                        let is_y_direction = i % 2 == 0;
                        let use_mask = mask.is_some();

                        for channel in 0..channels_per_offset_group {
                            let mask = mask.unwrap_or_else(|| F::one());
                            let image = image.index_axis(Axis(0), channel);
                            let weight =
                                get_coordinate_weight(image, height, width, y, x, is_y_direction);
                            *grad_offset += mask * weight * columns[channel];
                            if use_mask && is_y_direction {
                                *grad_mask += columns[channel]
                                    * bilinear_interpolate(image, height, width, y, x);
                            }
                        }
                    }
                }
            });

        let mask_gradient = mask.map(|_| {
            let mut grad_mask = grad_mask
                .into_shape_with_order((offset_channels / 2, batch_size, out_h, out_w))
                .unwrap();
            grad_mask.swap_axes(0, 1);
            NdArrayTensor::new(grad_mask.into_dyn().into_shared())
        });
        let mut grad_offset = grad_offset
            .into_shape_with_order((offset_channels, batch_size, out_h, out_w))
            .unwrap();
        grad_offset.swap_axes(0, 1);
        let offset_gradient = NdArrayTensor::new(grad_offset.into_dyn().into_shared());
        (offset_gradient, mask_gradient)
    }

    fn get_coordinate_weight<F: FloatNdArrayElement>(
        input: ArrayView2<F>,
        height: usize,
        width: usize,
        y: F,
        x: F,
        is_y_direction: bool,
    ) -> F {
        let y = y.to_f32();
        let x = x.to_f32();

        let y_low = f32::floor(y);
        let x_low = f32::floor(x);
        let y_high = y_low + 1.;
        let x_high = x_low + 1.;

        let valid_y_low = y_low >= 0. && y_low < height as f32;
        let valid_y_high = y_high >= 0. && y_high < height as f32;
        let valid_x_low = x_low >= 0. && x_low < width as f32;
        let valid_x_high = x_high >= 0. && x_high < width as f32;

        let bottom_left = if valid_y_low && valid_x_low {
            input[[y_low as usize, x_low as usize]]
        } else {
            F::zero()
        };
        let bottom_right = if valid_y_low && valid_x_high {
            input[[y_low as usize, x_high as usize]]
        } else {
            F::zero()
        };
        let top_left = if valid_y_high && valid_x_low {
            input[[y_high as usize, x_low as usize]]
        } else {
            F::zero()
        };
        let top_right = if valid_y_high && valid_x_high {
            input[[y_high as usize, x_high as usize]]
        } else {
            F::zero()
        };

        if is_y_direction {
            let delta_x = F::from_elem(x - x_low);
            delta_x * (top_right - bottom_right) + (F::one() - delta_x) * (top_left - bottom_left)
        } else {
            let delta_y = F::from_elem(y - y_low);
            delta_y * (top_right - top_left) + (F::one() - delta_y) * (bottom_right - bottom_left)
        }
    }

    fn compute_input_grad<F: FloatNdArrayElement>(
        columns: ArrayView6<F>,
        offset: ArrayView4<F>,
        mask: Option<ArrayView6<F>>,
        args: &DeformConvOptions<2>,
        kernel_dims: (usize, usize),
        input_shape: (usize, usize, usize, usize),
    ) -> NdArrayTensor<F> {
        let (batch_size, in_channels, height, width) = input_shape;
        let (kernel_h, kernel_w) = kernel_dims;
        let offs_groups = args.offset_groups;
        let channels_per_offset_group = in_channels / offs_groups;

        let grad_in =
            Array4::from_shape_simple_fn((batch_size, in_channels, height, width), || {
                AtomicF32::new(0.0)
            });

        let compute_for_each = |(in_channel, kernel_y, kernel_x, batch, out_y, out_x), col: &F| {
            let group = in_channel / channels_per_offset_group;
            let offset = offset.slice(s![batch, .., out_y, out_x]);
            let offset = offset
                .to_shape((offs_groups, kernel_h, kernel_w, 2))
                .unwrap();
            let offset = offset.slice(s![group, kernel_y, kernel_x, ..]);
            let offset = [offset[0], offset[1]];
            let mask = mask
                .as_ref()
                .map(|it| it[[batch, group, kernel_y, kernel_x, out_y, out_x]].to_f32());
            let y = F::from_elem(out_y * args.stride[0] + kernel_y * args.dilation[0])
                - F::from_elem(args.padding[0])
                + offset[0];
            let x = F::from_elem(out_x * args.stride[1] + kernel_x * args.dilation[1])
                - F::from_elem(args.padding[1])
                + offset[1];
            let grad_in = grad_in.slice(s![batch, in_channel, .., ..]);
            deform_col2img_kernel(y.to_f32(), x.to_f32(), mask, col.to_f32(), grad_in);
        };

        // `for_each` expects a 2-tuple argument with `.into_par_iter()`, but 2 separate arguments otherwise
        #[cfg(feature = "std")]
        run_par!(|| {
            iter_par!(Zip::indexed(columns))
                .for_each(|(args0, args1)| compute_for_each(args0, args1))
        });

        #[cfg(not(feature = "std"))]
        run_par!(|| {
            iter_par!(Zip::indexed(columns)).for_each(|args0, args1| compute_for_each(args0, args1))
        });

        let grad_in: Array1<F> = grad_in
            .into_iter()
            .map(|it| F::from_elem(it.into_inner()))
            .collect();
        let grad_in = grad_in
            .into_shape_with_order((batch_size, in_channels, height, width))
            .unwrap();
        NdArrayTensor::new(grad_in.into_dyn().into_shared())
    }

    fn deform_col2img_kernel(
        y: f32,
        x: f32,
        mask: Option<f32>,
        col: f32,
        grad_input: ArrayView2<AtomicF32>,
    ) {
        let (height, width) = grad_input.dim();
        let mask_value = mask.unwrap_or(1.0);

        for dy in -1..=1 {
            for dx in -1..=1 {
                let yp = f32::floor(y) + dy as f32;
                let xp = f32::floor(x) + dx as f32;

                if yp >= 0.0
                    && yp < height as f32
                    && xp >= 0.0
                    && xp < width as f32
                    && f32::abs(y - yp) < 1.0
                    && f32::abs(x - xp) < 1.0
                {
                    let weight = (1.0 - f32::abs(y - yp)) * (1.0 - f32::abs(x - xp));

                    #[cfg_attr(not(target_has_atomic = "32"), allow(unused))]
                    let value = mask_value * weight * col;

                    #[cfg(target_has_atomic = "32")]
                    grad_input[[yp as usize, xp as usize]].fetch_add(value, Ordering::AcqRel);
                    #[cfg(not(target_has_atomic = "32"))]
                    panic!("Can't use deformable convolution backwards pass without atomics");
                }
            }
        }
    }
}
