use super::{bilinear_interpolate, deform_im2col, index};
use crate::{
    CubeRuntime,
    kernel::{
        cast, into_contiguous,
        matmul::{MatmulStrategy, matmul},
        reduce::reduce_dim,
        slice_assign,
    },
    ops::{
        numeric::{empty_device_dtype, ones_client, zeros_client},
        reshape, swap_dims,
    },
    tensor::CubeTensor,
};
use burn_backend::{DType, Shape, ops::DeformConvOptions};
use cubecl::{
    CubeDim, CubeLaunch, calculate_cube_count_elemwise, cube, features::TypeUsage, prelude::*,
};
use cubek::{
    convolution::components::ConvSetupError,
    reduce::components::instructions::ReduceOperationConfig,
};
use std::marker::PhantomData;

/// Calculate the [deformable 2D convolution](crate::ops::ModuleOps::deform_conv2d) backward pass using convolutions.
#[allow(
    clippy::single_range_in_vec_init,
    clippy::type_complexity,
    clippy::too_many_arguments
)]
pub(crate) fn deform_conv2d_backward<R: CubeRuntime>(
    input: CubeTensor<R>,
    offset: CubeTensor<R>,
    weight: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    bias: Option<CubeTensor<R>>,
    out_grad: CubeTensor<R>,
    options: DeformConvOptions<2>,
) -> Result<
    (
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        Option<CubeTensor<R>>,
    ),
    ConvSetupError,
> {
    let [_, _, out_h, out_w] = out_grad.shape.dims();
    let [_, _, kernel_h, kernel_w] = weight.shape.dims();

    let gradient_bias = bias.map(|bias| {
        let grad = reduce_dim(
            out_grad.clone(),
            None,
            0,
            Default::default(),
            ReduceOperationConfig::Sum,
        )
        .unwrap();
        let grad = reduce_dim(
            grad,
            None,
            2,
            Default::default(),
            ReduceOperationConfig::Sum,
        )
        .unwrap();
        let grad = reduce_dim(
            grad,
            None,
            3,
            Default::default(),
            ReduceOperationConfig::Sum,
        )
        .unwrap();

        reshape(grad, bias.shape)
    });

    let input = into_contiguous(input);
    let offset = into_contiguous(offset);
    let mask = mask.map(|it| into_contiguous(it));

    let (input_gradient, offset_gradient, mask_gradient) = backward_gradient_inputs(
        input.clone(),
        weight.clone(),
        offset.clone(),
        mask.clone(),
        out_grad.clone(),
        &options,
        (kernel_h, kernel_w),
    )?;

    let weight_grad = compute_weight_grad(
        input,
        offset,
        mask,
        out_grad,
        options,
        (kernel_h, kernel_w),
        (out_h, out_w),
    )?;

    Ok((
        input_gradient,
        offset_gradient,
        weight_grad,
        mask_gradient,
        gradient_bias,
    ))
}

fn compute_weight_grad<R: CubeRuntime>(
    input: CubeTensor<R>,
    offset: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    out_grad: CubeTensor<R>,
    options: DeformConvOptions<2>,
    kernel_dims: (usize, usize),
    out_dims: (usize, usize),
) -> Result<CubeTensor<R>, ConvSetupError> {
    let [_, in_channels, _, _] = input.shape.dims();
    let [_, out_channels, _, _] = out_grad.shape.dims();
    let (kernel_h, kernel_w) = kernel_dims;
    let groups = options.weight_groups;
    let dtype = input.dtype;

    let in_c_per_group = in_channels / groups;
    let out_c_per_group = out_channels / groups;

    let columns = deform_im2col(input, offset, mask, options, out_dims, kernel_dims)?;
    let [col_size_0, col_size_1] = columns.shape.dims();
    let col_size_0 = col_size_0 / groups;

    let out_grad = swap_dims(out_grad, 0, 1);
    let out_grad = reshape(out_grad, Shape::new([groups, out_c_per_group, col_size_1]));

    let columns = reshape(columns, Shape::new([groups, col_size_0, col_size_1]));
    let columns = swap_dims(columns, 1, 2);

    let grad_weight = matmul(out_grad, columns, None, MatmulStrategy::default(), dtype)?;

    Ok(reshape(
        grad_weight,
        Shape::new([out_channels, in_c_per_group, kernel_h, kernel_w]),
    ))
}

type InputGradients<R> = (CubeTensor<R>, CubeTensor<R>, Option<CubeTensor<R>>);

fn backward_gradient_inputs<R: CubeRuntime>(
    image: CubeTensor<R>,
    weight: CubeTensor<R>,
    offset: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    out_grad: CubeTensor<R>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
) -> Result<InputGradients<R>, ConvSetupError> {
    let client = out_grad.client.clone();
    let device = out_grad.device.clone();

    let [out_channels, in_c_per_group, kernel_h, kernel_w] = weight.shape.dims();
    let [batch_size, _, out_h, out_w] = out_grad.shape.dims();

    let groups = options.weight_groups;
    let out_c_per_group = out_channels / groups;

    let col_shape_0 = in_c_per_group * kernel_h * kernel_w;
    let col_shape_1 = batch_size * out_h * out_w;
    let col_shape = Shape::new([groups, col_shape_0, col_shape_1]);
    let mut columns = empty_device_dtype(client, device, col_shape, weight.dtype);

    let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_shape_0]));

    let out_grad = swap_dims(out_grad, 0, 1);
    let out_grad_shape = Shape::new([groups, out_c_per_group, col_shape_1]);
    let out_grad = reshape(out_grad, out_grad_shape);

    for group in 0..groups {
        let dtype = weight.dtype;
        let weight = swap_dims(index(weight.clone(), group), 0, 1);
        let out_grad = index(out_grad.clone(), group);
        let values = matmul(weight, out_grad, None, MatmulStrategy::default(), dtype)?;
        let values = reshape(values, Shape::new([1, col_shape_0, col_shape_1]));
        columns = slice_assign(
            columns,
            &[
                burn_backend::Slice::from(group..group + 1),
                burn_backend::Slice::from(0..col_shape_0),
                burn_backend::Slice::from(0..col_shape_1),
            ],
            values,
        );
    }

    let columns = reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]));

    let input_shape = image.shape.clone();
    let (offset_gradient, mask_gradient) = compute_offset_and_mask_gradient(
        columns.clone(),
        image,
        offset.clone(),
        mask.clone(),
        options,
        kernel_dims,
    )?;

    let input_gradient =
        compute_input_grad(columns, offset, mask, options, kernel_dims, input_shape)?;

    Ok((input_gradient, offset_gradient, mask_gradient))
}

fn compute_offset_and_mask_gradient<R: CubeRuntime>(
    columns: CubeTensor<R>,
    image: CubeTensor<R>,
    offset: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
) -> Result<(CubeTensor<R>, Option<CubeTensor<R>>), ConvSetupError> {
    let client = offset.client.clone();
    let device = offset.device.clone();
    let (kernel_height, kernel_width) = kernel_dims;

    let use_mask = mask.is_some();

    let mask = mask.unwrap_or_else(|| {
        ones_client(
            client.clone(),
            device.clone(),
            Shape::new([
                offset.shape[0],
                offset.shape[1] / 2,
                offset.shape[2],
                offset.shape[3],
            ]),
            columns.dtype,
        )
    });

    let grad_offset = empty_device_dtype(
        client.clone(),
        device.clone(),
        offset.shape.clone(),
        offset.dtype,
    );
    let grad_mask = empty_device_dtype(
        client.clone(),
        device.clone(),
        mask.shape.clone(),
        mask.dtype,
    );

    let num_elements_offset = offset.shape.num_elements();
    let cube_dim = CubeDim::new(&image.client, num_elements_offset);
    let cube_count = calculate_cube_count_elemwise(&image.client, num_elements_offset, cube_dim);

    let dtype: StorageType = image.dtype.into();
    unsafe {
        deform_col2img_coord_kernel::launch_unchecked(
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
                ScalarArg::new(options.stride[0]),
                ScalarArg::new(options.stride[1]),
                ScalarArg::new(options.dilation[0]),
                ScalarArg::new(options.dilation[1]),
                InputScalar::new(options.padding[0] as f32, dtype.elem_type()),
                InputScalar::new(options.padding[1] as f32, dtype.elem_type()),
                ScalarArg::new(options.offset_groups),
                ScalarArg::new(kernel_height),
                ScalarArg::new(kernel_width),
            ),
            use_mask,
            dtype,
        )
    }?;

    let mask_gradient = if use_mask { Some(grad_mask) } else { None };
    Ok((grad_offset, mask_gradient))
}

#[derive(CubeLaunch, CubeType)]
struct DeformConv2dCol2ImgCoordArgs {
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_h: InputScalar,
    pad_w: InputScalar,
    offset_groups: usize,
    kernel_height: usize,
    kernel_width: usize,
}

#[expect(clippy::collapsible_if)]
#[cube(launch_unchecked)]
fn deform_col2img_coord_kernel<F: Float>(
    image: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &Tensor<F>,
    grad_offset: &mut Tensor<F>,
    grad_mask: &mut Tensor<F>,
    args: &DeformConv2dCol2ImgCoordArgs,
    #[comptime] use_mask: bool,
    #[define(F)] _dtype: StorageType,
) {
    // Position format: [batch, [offset_group, kernel_h, kernel_w, 2], out_h, out_w]
    // Alternatively : [batch, offset_channels, out_h, out_w]

    if ABSOLUTE_POS >= grad_offset.len() {
        terminate!();
    }

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
    let is_y_direction = offset_c.is_multiple_of(2);

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

        let y = F::cast_from(out_y * args.stride_h + i * args.dilation_h) - args.pad_h.get::<F>()
            + offset_y;
        let x = F::cast_from(out_x * args.stride_w + j * args.dilation_w) - args.pad_w.get::<F>()
            + offset_x;

        let weight = get_coordinate_weight(
            &image.slice(image_base_idx, image.len()),
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
    input: &Slice<F>,
    height: usize,
    width: usize,
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
        input[y_low as usize * stride_y + x_low as usize]
    } else {
        F::new(0.0)
    };
    let bottom_right = if valid_y_low && valid_x_high {
        input[y_low as usize * stride_y + x_high as usize]
    } else {
        F::new(0.0)
    };
    let top_left = if valid_y_high && valid_x_low {
        input[y_high as usize * stride_y + x_low as usize]
    } else {
        F::new(0.0)
    };
    let top_right = if valid_y_high && valid_x_high {
        input[y_high as usize * stride_y + x_high as usize]
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

fn compute_input_grad<R: CubeRuntime>(
    columns: CubeTensor<R>,
    offset: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
    input_shape: Shape,
) -> Result<CubeTensor<R>, LaunchError> {
    let client = offset.client.clone();
    let device = offset.device.clone();

    let supports_fadd = client
        .properties()
        .type_usage(StorageType::Atomic(
            f32::as_type_native_unchecked().elem_type(),
        ))
        .contains(TypeUsage::AtomicAdd);
    let supports_same_type = client
        .properties()
        .type_usage(StorageType::Atomic(columns.dtype.into()))
        .contains(TypeUsage::AtomicAdd);

    let [batch_size, in_channels, height, width] = input_shape.dims();
    let (kernel_height, kernel_width) = kernel_dims;

    let shape = Shape::new([batch_size, in_channels, height, width]);
    let grad_in = match supports_fadd && supports_same_type {
        // Use type as is to save a cast
        true => zeros_client(client.clone(), device.clone(), shape, columns.dtype),
        // Force `f32` to enable bitcasting as `u32`, or use intrinsic when supported
        false => zeros_client(client.clone(), device.clone(), shape, DType::F32),
    };
    let grad_arg = grad_in.as_tensor_arg(1);

    let use_mask = mask.is_some();
    let mask = mask.unwrap_or_else(|| {
        ones_client(
            client.clone(),
            device.clone(),
            Shape::new([1, 1, 1, 1]),
            columns.dtype,
        )
    });

    let num_elements = columns.shape.num_elements();
    let cube_dim = CubeDim::new(&offset.client, num_elements);
    let cube_count = calculate_cube_count_elemwise(&offset.client, num_elements, cube_dim);

    let launch = match supports_fadd {
        true => deform_col2img_kernel::launch_unchecked::<IntrinsicFloatAtomicAddFamily, R>,
        false => deform_col2img_kernel::launch_unchecked::<CASFloatAtomicAdd, R>,
    };
    let dtype = offset.dtype;
    let dtypes: [StorageType; 2] = match supports_same_type {
        true => [dtype.into(), dtype.into()],
        false => [dtype.into(), DType::F32.into()],
    };
    unsafe {
        launch(
            &offset.client,
            cube_count,
            cube_dim,
            offset.as_tensor_arg(1),
            mask.as_tensor_arg(1),
            columns.as_tensor_arg(1),
            grad_arg,
            DeformConv2dCol2ImgArgsLaunch::new(
                ScalarArg::new(options.stride[0]),
                ScalarArg::new(options.stride[1]),
                ScalarArg::new(options.dilation[0]),
                ScalarArg::new(options.dilation[1]),
                InputScalar::new(options.padding[0] as f32, dtypes[0].elem_type()),
                InputScalar::new(options.padding[1] as f32, dtypes[0].elem_type()),
                ScalarArg::new(options.offset_groups),
                ScalarArg::new(batch_size),
                ScalarArg::new(in_channels),
                ScalarArg::new(height),
                ScalarArg::new(width),
                ScalarArg::new(kernel_height),
                ScalarArg::new(kernel_width),
            ),
            use_mask,
            dtypes,
        )
    }?;

    Ok(if !supports_same_type || !supports_fadd {
        cast(grad_in, dtype)
    } else {
        grad_in
    })
}

#[derive(CubeLaunch, CubeType)]
struct DeformConv2dCol2ImgArgs {
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_h: InputScalar,
    pad_w: InputScalar,
    offset_groups: usize,
    batch_size: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    kernel_height: usize,
    kernel_width: usize,
}

#[cube(launch_unchecked)]
fn deform_col2img_kernel<F: Float, FP: Float, FAdd: FloatAtomicAddFamily>(
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &Tensor<F>,
    grad_input: &mut Tensor<Atomic<ProxyType<FAdd, FP>>>,
    args: &DeformConv2dCol2ImgArgs,
    #[comptime] use_mask: bool,
    #[define(F, FP)] _dtype: [StorageType; 2],
) {
    // Position format: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]
    if ABSOLUTE_POS >= columns.len() {
        terminate!();
    }

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

    let offset_y = offset[offset_base_idx + offset_y_idx];
    let offset_x = offset[offset_base_idx + offset_x_idx];

    let mask_value = if use_mask {
        let mask_base_idx =
            (batch * n_offset_groups + offset_group) * kernel_h * kernel_w * out_h * out_w;

        mask[mask_base_idx + (mask_idx * out_h + out_y) * out_w + out_x]
    } else {
        F::new(1.0)
    };

    let y = F::cast_from(out_y * args.stride_h + kernel_y * args.dilation_h)
        - args.pad_h.get::<F>()
        + offset_y;
    let x = F::cast_from(out_x * args.stride_w + kernel_x * args.dilation_w)
        - args.pad_w.get::<F>()
        + offset_x;

    for dy in -1..=1i32 {
        #[unroll]
        for dx in -1..=1i32 {
            let yp = F::floor(y) + F::cast_from(dy);
            let xp = F::floor(x) + F::cast_from(dx);

            if yp >= F::new(0.0)
                && yp < F::cast_from(height)
                && xp >= F::new(0.0)
                && xp < F::cast_from(width)
                && F::abs(y - yp) < F::new(1.0)
                && F::abs(x - xp) < F::new(1.0)
            {
                let gradient_pos =
                    ((batch * n_in_channels + in_channel) * height + usize::cast_from(yp)) * width
                        + usize::cast_from(xp);

                let weight = (F::new(1.0) - F::abs(y - yp)) * (F::new(1.0) - F::abs(x - xp));

                let value = mask_value * F::cast_from(weight) * columns[ABSOLUTE_POS];

                FAdd::Op::<FP>::float_atomic_add::<F>(&mut grad_input[gradient_pos], value);
            }
        }
    }
}

type ProxyType<FADF, FP> = <<FADF as FloatAtomicAddFamily>::Op<FP> as FloatAtomicAdd>::ProxyType;

#[cube]
trait FloatAtomicAddFamily: Send + Sync + 'static {
    type Op<ProxyType: Float>: FloatAtomicAdd;
}

#[cube]
trait FloatAtomicAdd: Send + Sync + 'static {
    type ProxyType: Numeric;

    fn float_atomic_add<F: Float>(ptr: &mut Atomic<Self::ProxyType>, value: F);
}

#[derive(CubeType)]
struct IntrinsicFloatAtomicAdd<F: Float> {
    #[cube(comptime)]
    _ty: PhantomData<F>,
}

#[derive(CubeType)]
struct CASFloatAtomicAdd;

struct IntrinsicFloatAtomicAddFamily;

impl FloatAtomicAddFamily for IntrinsicFloatAtomicAddFamily {
    type Op<ProxyType: Float> = IntrinsicFloatAtomicAdd<ProxyType>;
}

impl FloatAtomicAddFamily for CASFloatAtomicAdd {
    type Op<ProxyType: Float> = Self;
}

#[cube]
impl<FAdd: Float> FloatAtomicAdd for IntrinsicFloatAtomicAdd<FAdd> {
    type ProxyType = FAdd;

    fn float_atomic_add<F: Float>(ptr: &mut Atomic<FAdd>, value: F) {
        let value = FAdd::cast_from(value);
        ptr.add(value);
    }
}

#[cube]
impl FloatAtomicAdd for CASFloatAtomicAdd {
    type ProxyType = u32;

    fn float_atomic_add<F: Float>(ptr: &mut Atomic<Self::ProxyType>, value: F) {
        let value = f32::cast_from(value);
        if value != 0.0 {
            let mut v = ptr.load();
            loop {
                let prev = v;
                let v_float = f32::from_bits(v);
                let new = (v_float + value).to_bits();
                v = ptr.compare_and_swap(v, new);
                if prev == v {
                    break;
                }
            }
        }
    }
}
