use burn_tensor::{
    ops::{DeformConv2dBackward, DeformConvOptions, FloatTensorOps as _},
    Shape,
};
use cubecl::{calculate_cube_count_elemwise, cube, prelude::*, CubeDim, CubeLaunch};

use crate::{
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, ones_device, zeros_device},
        reshape, swap_dims,
    },
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

use super::{bilinear_interpolate, deform_im2col, index};

/// Calculate the [deformable 2D convolution](crate::ops::ModuleOps::deform_conv2d) backward pass using convolutions.
#[allow(clippy::single_range_in_vec_init)]
pub(crate) fn deform_conv2d_backward<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    offset: JitTensor<R, E>,
    weight: JitTensor<R, E>,
    mask: Option<JitTensor<R, E>>,
    bias: Option<JitTensor<R, E>>,
    out_grad: JitTensor<R, E>,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<JitBackend<R, E, I>> {
    let [_, _, out_h, out_w] = out_grad.shape.dims();
    let [_, _, kernel_h, kernel_w] = weight.shape.dims();

    let gradient_bias = bias.map(|bias| {
        let grad = JitBackend::<R, E, I>::float_sum_dim(out_grad.clone(), 0);
        let grad = JitBackend::<R, E, I>::float_sum_dim(grad, 2);
        let grad = JitBackend::<R, E, I>::float_sum_dim(grad, 3);

        reshape(grad, bias.shape)
    });

    let input = into_contiguous(input);
    let offset = into_contiguous(offset);
    let mask = mask.map(|it| into_contiguous(it));

    let (input_gradient, offset_gradient, mask_gradient) = backward_gradient_inputs::<R, E, I>(
        input.clone(),
        weight.clone(),
        offset.clone(),
        mask.clone(),
        out_grad.clone(),
        &options,
        (kernel_h, kernel_w),
    );

    let weight_grad = compute_weight_grad::<R, E, I>(
        input,
        offset,
        mask,
        out_grad,
        options,
        (kernel_h, kernel_w),
        (out_h, out_w),
    );

    DeformConv2dBackward::new(
        input_gradient,
        offset_gradient,
        weight_grad,
        mask_gradient,
        gradient_bias,
    )
}

fn compute_weight_grad<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    offset: JitTensor<R, E>,
    mask: Option<JitTensor<R, E>>,
    out_grad: JitTensor<R, E>,
    options: DeformConvOptions<2>,
    kernel_dims: (usize, usize),
    out_dims: (usize, usize),
) -> JitTensor<R, E> {
    let [_, in_channels, _, _] = input.shape.dims();
    let [_, out_channels, _, _] = out_grad.shape.dims();
    let (kernel_h, kernel_w) = kernel_dims;
    let groups = options.weight_groups;

    let in_c_per_group = in_channels / groups;
    let out_c_per_group = out_channels / groups;

    let columns = deform_im2col(input, offset, mask, options, out_dims, kernel_dims);
    let [col_size_0, col_size_1] = columns.shape.dims();
    let col_size_0 = col_size_0 / groups;

    let out_grad = swap_dims(out_grad, 0, 1);
    let out_grad = reshape(out_grad, Shape::new([groups, out_c_per_group, col_size_1]));

    let columns = reshape(columns, Shape::new([groups, col_size_0, col_size_1]));
    let columns = swap_dims(columns, 1, 2);

    let grad_weight = JitBackend::<R, E, I>::float_matmul(out_grad, columns);

    JitBackend::<R, E, I>::float_reshape(
        grad_weight,
        Shape::new([out_channels, in_c_per_group, kernel_h, kernel_w]),
    )
}

type InputGradients<R, E> = (JitTensor<R, E>, JitTensor<R, E>, Option<JitTensor<R, E>>);

fn backward_gradient_inputs<R: JitRuntime, E: FloatElement, I: IntElement>(
    image: JitTensor<R, E>,
    weight: JitTensor<R, E>,
    offset: JitTensor<R, E>,
    mask: Option<JitTensor<R, E>>,
    out_grad: JitTensor<R, E>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
) -> InputGradients<R, E> {
    let client = out_grad.client.clone();
    let device = out_grad.device.clone();

    let [out_channels, in_c_per_group, kernel_h, kernel_w] = weight.shape.dims();
    let [batch_size, _, out_h, out_w] = out_grad.shape.dims();

    let groups = options.weight_groups;
    let out_c_per_group = out_channels / groups;

    let col_shape_0 = in_c_per_group * kernel_h * kernel_w;
    let col_shape_1 = batch_size * out_h * out_w;
    let col_shape = Shape::new([groups, col_shape_0, col_shape_1]);
    let mut columns = empty_device(client, device, col_shape);

    let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_shape_0]));

    let out_grad = swap_dims(out_grad, 0, 1);
    let out_grad_shape = Shape::new([groups, out_c_per_group, col_shape_1]);
    let out_grad = reshape(out_grad, out_grad_shape);

    for group in 0..groups {
        let weight = swap_dims(index::<R, E, I>(weight.clone(), group), 0, 1);
        let out_grad = index::<R, E, I>(out_grad.clone(), group);
        let values = JitBackend::<R, E, I>::float_matmul(weight, out_grad);
        let values = reshape(values, Shape::new([1, col_shape_0, col_shape_1]));
        columns = JitBackend::<R, E, I>::float_slice_assign(
            columns,
            &[group..group + 1, 0..col_shape_0, 0..col_shape_1],
            values,
        );
    }

    let columns = reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]));

    let input_shape = image.shape.clone();
    let (offset_gradient, mask_gradient) = compute_offset_and_mask_gradient::<R, E>(
        columns.clone(),
        image,
        offset.clone(),
        mask.clone(),
        options,
        kernel_dims,
    );

    let input_gradient =
        compute_input_grad::<R, E>(columns, offset, mask, options, kernel_dims, input_shape);

    (input_gradient, offset_gradient, mask_gradient)
}

fn compute_offset_and_mask_gradient<R: JitRuntime, E: FloatElement>(
    columns: JitTensor<R, E>,
    image: JitTensor<R, E>,
    offset: JitTensor<R, E>,
    mask: Option<JitTensor<R, E>>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
) -> (JitTensor<R, E>, Option<JitTensor<R, E>>) {
    let client = offset.client.clone();
    let device = offset.device.clone();
    let (kernel_height, kernel_width) = kernel_dims;

    let use_mask = mask.is_some();

    let mask = mask.unwrap_or_else(|| {
        ones_device(
            client.clone(),
            device.clone(),
            Shape::new([
                offset.shape.dims[0],
                offset.shape.dims[1] / 2,
                offset.shape.dims[2],
                offset.shape.dims[3],
            ]),
        )
    });

    let grad_offset = empty_device(client.clone(), device.clone(), offset.shape.clone());
    let grad_mask = empty_device(client.clone(), device.clone(), mask.shape.clone());

    let num_elements_offset = offset.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elements_offset, cube_dim);

    deform_col2img_coord_kernel::launch::<E, R>(
        &image.client,
        cube_count,
        cube_dim,
        image.as_handle_ref().as_tensor_arg(1),
        offset.as_handle_ref().as_tensor_arg(1),
        mask.as_handle_ref().as_tensor_arg(1),
        columns.as_handle_ref().as_tensor_arg(1),
        grad_offset.as_handle_ref().as_tensor_arg(1),
        grad_mask.as_handle_ref().as_tensor_arg(1),
        DeformConv2dCol2ImgCoordArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(E::from_elem(options.padding[0] as f32)),
            ScalarArg::new(E::from_elem(options.padding[1] as f32)),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
        ),
        use_mask,
    );

    let mask_gradient = if use_mask { Some(grad_mask) } else { None };
    (grad_offset, mask_gradient)
}

#[derive(CubeLaunch)]
struct DeformConv2dCol2ImgCoordArgs<F: Float> {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    pad_h: F,
    pad_w: F,
    offset_groups: u32,
    kernel_height: u32,
    kernel_width: u32,
}

#[allow(clippy::collapsible_if)]
#[cube(launch)]
fn deform_col2img_coord_kernel<F: Float>(
    image: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &Tensor<F>,
    grad_offset: &mut Tensor<F>,
    grad_mask: &mut Tensor<F>,
    args: &DeformConv2dCol2ImgCoordArgs<F>,
    #[comptime] use_mask: bool,
) {
    // Position format: [batch, [offset_group, kernel_h, kernel_w, 2], out_h, out_w]
    // Alternatively : [batch, offset_channels, out_h, out_w]

    let offset_channels = offset.shape(1);
    let out_h = offset.shape(2);
    let out_w = offset.shape(3);
    let batch_size = image.shape(0);
    let in_channels = image.shape(1);
    let height = image.shape(2);
    let width = image.shape(3);
    let kernel_w = args.kernel_width;
    let kernel_h = args.kernel_height;
    let n_offset_groups = args.offset_groups;
    let _ = mask[0]; // Make sure mask isn't removed from bind group

    let mut grad_offset_val = F::new(0.0);
    let mut grad_mask_val = F::new(0.0);

    let w = ABSOLUTE_POS % out_w;
    let h = (ABSOLUTE_POS / out_w) % out_h;
    let w_w = (ABSOLUTE_POS / (out_w * out_h * 2)) % kernel_w;
    let w_h = (ABSOLUTE_POS / (out_w * out_h * 2 * kernel_w)) % kernel_h;
    let c = (ABSOLUTE_POS / (out_w * out_h)) % offset_channels;
    let b = ABSOLUTE_POS / (out_w * out_h * offset_channels);

    let offset_group = c / (kernel_h * kernel_w * 2);
    let col_step = kernel_h * kernel_w;

    let channels_per_offset_group = in_channels / args.offset_groups;

    let col_base_idx =
        offset_group * channels_per_offset_group * kernel_h * kernel_w * batch_size * out_w * out_h;
    let mut image_base_idx =
        (b * n_offset_groups + offset_group) * channels_per_offset_group * height * width;
    let offset_base_idx =
        (b * n_offset_groups + offset_group) * 2 * kernel_h * kernel_w * out_h * out_w;
    let mask_base_idx = (b * n_offset_groups + offset_group) * kernel_h * kernel_w * out_h * out_w;

    let offset_c = c - offset_group * 2 * kernel_h * kernel_w;
    let is_y_direction = offset_c % 2 == 0;

    let c_bound = channels_per_offset_group * kernel_h * kernel_w;

    for col_c in range_stepped(offset_c / 2, c_bound, col_step) {
        let col_pos = (((col_c * batch_size + b) * out_h) + h) * out_w + w;

        let out_x = col_pos % out_w;
        let out_y = (col_pos / out_w) % out_h;
        let j = (col_pos / (out_w * out_h * batch_size)) % kernel_w;
        let i = (col_pos / (out_w * out_h * batch_size * kernel_w)) % kernel_h;

        let mask_idx = i * kernel_w + j;
        let offset_idx = mask_idx * 2;

        let offset_y_idx = (offset_idx * out_h + out_y) * out_w + out_x;
        let offset_x_idx = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

        let offset_y = offset[offset_base_idx + offset_y_idx];
        let offset_x = offset[offset_base_idx + offset_x_idx];

        let mask_value = if use_mask {
            mask[mask_base_idx + (mask_idx * out_h + out_y) * out_w + out_x]
        } else {
            F::new(1.0)
        };

        let y = F::cast_from(out_y * args.stride_h + i * args.dilation_h) - args.pad_h + offset_y;
        let x = F::cast_from(out_x * args.stride_w + j * args.dilation_w) - args.pad_w + offset_x;

        let weight = get_coordinate_weight(
            image.slice(image_base_idx, image.len()),
            height,
            width,
            y,
            x,
            is_y_direction,
        );

        grad_offset_val += mask_value * weight * columns[col_base_idx + col_pos];

        if use_mask {
            if is_y_direction {
                grad_mask_val += columns[col_base_idx + col_pos]
                    * bilinear_interpolate(image, height, width, y, x, image_base_idx);
            }
        }

        image_base_idx += height * width;
    }

    grad_offset[ABSOLUTE_POS] = grad_offset_val;

    if use_mask {
        if is_y_direction {
            let idx = ((((b * n_offset_groups + offset_group) * kernel_h + w_h) * kernel_w + w_w)
                * out_h
                + h)
                * out_w
                + w;
            grad_mask[idx] = grad_mask_val
        }
    }
}

#[cube]
fn get_coordinate_weight<F: Float>(
    input: &Slice<'_, F>,
    height: u32,
    width: u32,
    y: F,
    x: F,
    is_y_direction: bool,
) -> F {
    let stride_y = width;

    let y = f32::cast_from(y);
    let x = f32::cast_from(x);

    let y_low = f32::floor(y);
    let x_low = f32::floor(x);
    let y_high = y_low + 1.;
    let x_high = x_low + 1.;

    let valid_y_low = y_low >= 0. && y_low < height as f32;
    let valid_y_high = y_high >= 0. && y_high < height as f32;
    let valid_x_low = x_low >= 0. && x_low < width as f32;
    let valid_x_high = x_high >= 0. && x_high < width as f32;

    let bottom_left = if valid_y_low && valid_x_low {
        input[y_low as u32 * stride_y + x_low as u32]
    } else {
        F::new(0.0)
    };
    let bottom_right = if valid_y_low && valid_x_high {
        input[y_low as u32 * stride_y + x_high as u32]
    } else {
        F::new(0.0)
    };
    let top_left = if valid_y_high && valid_x_low {
        input[y_high as u32 * stride_y + x_low as u32]
    } else {
        F::new(0.0)
    };
    let top_right = if valid_y_high && valid_x_high {
        input[y_high as u32 * stride_y + x_high as u32]
    } else {
        F::new(0.0)
    };

    if is_y_direction {
        let delta_x = F::cast_from(x - x_low);
        delta_x * (top_right - bottom_right) + (F::new(1.0) - delta_x) * (top_left - bottom_left)
    } else {
        let delta_y = F::cast_from(y - y_low);
        delta_y * (top_right - top_left) + (F::new(1.0) - delta_y) * (bottom_right - bottom_left)
    }
}

fn compute_input_grad<R: JitRuntime, E: FloatElement>(
    columns: JitTensor<R, E>,
    offset: JitTensor<R, E>,
    mask: Option<JitTensor<R, E>>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
    input_shape: Shape,
) -> JitTensor<R, E> {
    let client = offset.client.clone();
    let device = offset.device.clone();

    let [batch_size, in_channels, height, width] = input_shape.dims();
    let (kernel_height, kernel_width) = kernel_dims;

    let grad_in = zeros_device::<R, E>(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, in_channels, height, width]),
    );

    let use_mask = mask.is_some();
    let mask = mask
        .unwrap_or_else(|| ones_device(client.clone(), device.clone(), Shape::new([1, 1, 1, 1])));

    let num_elements = columns.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elements, cube_dim);

    deform_col2img_kernel::launch::<E, R>(
        &offset.client,
        cube_count,
        cube_dim,
        offset.as_tensor_arg(1),
        mask.as_tensor_arg(1),
        columns.as_tensor_arg(1),
        grad_in.as_tensor_arg(1),
        DeformConv2dCol2ImgArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as f32),
            ScalarArg::new(options.padding[1] as f32),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(batch_size as u32),
            ScalarArg::new(in_channels as u32),
            ScalarArg::new(height as u32),
            ScalarArg::new(width as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
        ),
        use_mask,
    );

    grad_in
}

#[derive(CubeLaunch)]
struct DeformConv2dCol2ImgArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    pad_h: f32,
    pad_w: f32,
    offset_groups: u32,
    batch_size: u32,
    in_channels: u32,
    height: u32,
    width: u32,
    kernel_height: u32,
    kernel_width: u32,
}

#[cube(launch)]
fn deform_col2img_kernel<F: Float>(
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &Tensor<F>,
    grad_input: &mut Tensor<AtomicU32>,
    args: &DeformConv2dCol2ImgArgs,
    #[comptime] use_mask: bool,
) {
    // Position format: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]
    let _ = mask[0]; // Keep mask in bind group

    let n_in_channels = args.in_channels;
    let height = args.height;
    let width = args.width;
    let out_h = offset.shape(2);
    let out_w = offset.shape(3);
    let kernel_h = args.kernel_height;
    let kernel_w = args.kernel_width;
    let n_offset_groups = args.offset_groups;
    let batch_size = args.batch_size;

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = (ABSOLUTE_POS / out_w) % out_h;
    let batch = (ABSOLUTE_POS / (out_w * out_h)) % batch_size;
    let kernel_x = (ABSOLUTE_POS / (out_w * out_h * batch_size)) % kernel_w;
    let kernel_y = (ABSOLUTE_POS / (out_w * out_h * batch_size * kernel_w)) % kernel_h;
    let in_channel = ABSOLUTE_POS / (out_w * out_h * batch_size * kernel_w * kernel_h);

    let channels_per_offset_group = n_in_channels / n_offset_groups;
    let offset_group = in_channel / channels_per_offset_group;

    let offset_base_idx =
        (batch * n_offset_groups + offset_group) * 2 * kernel_h * kernel_w * out_h * out_w;

    let mask_idx = kernel_y * kernel_w + kernel_x;
    let offset_idx = mask_idx * 2;

    let offset_y_idx = (offset_idx * out_h + out_y) * out_w + out_x;
    let offset_x_idx = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    let offset_y = f32::cast_from(offset[offset_base_idx + offset_y_idx]);
    let offset_x = f32::cast_from(offset[offset_base_idx + offset_x_idx]);

    let mask_value = if use_mask {
        let mask_base_idx =
            (batch * n_offset_groups + offset_group) * kernel_h * kernel_w * out_h * out_w;

        mask[mask_base_idx + (mask_idx * out_h + out_y) * out_w + out_x]
    } else {
        F::new(1.0)
    };

    let y =
        f32::cast_from(out_y * args.stride_h + kernel_y * args.dilation_h) - args.pad_h + offset_y;
    let x =
        f32::cast_from(out_x * args.stride_w + kernel_x * args.dilation_w) - args.pad_w + offset_x;

    for dy in -1..=1 {
        #[unroll]
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
                let gradient_pos =
                    ((batch * n_in_channels + in_channel) * height + yp as u32) * width + xp as u32;

                let weight = (1.0 - f32::abs(y - yp)) * (1.0 - f32::abs(x - xp));

                let value = mask_value * F::cast_from(weight) * columns[ABSOLUTE_POS];

                float_atomic_add::<F>(&mut grad_input[gradient_pos], value);
            }
        }
    }
}

#[cube]
fn float_atomic_add<F: Float>(ptr: &mut AtomicU32, value: F) {
    if value != F::new(0.0) {
        let mut v = AtomicU32::load(ptr);
        loop {
            let prev = v;
            let v_float = F::bitcast_from(v);
            let new = u32::bitcast_from(v_float + value);
            v = AtomicU32::compare_and_swap(ptr, v, new);
            if prev == v {
                break;
            }
        }
    }
}
