use burn_tensor::{
    Shape,
    ops::{ConvOptions, conv::calculate_conv_output_size},
};
use cubecl::{
    calculate_cube_count_elemwise,
    linalg::{
        convolution::ConvLaunchError,
        tensor::{TensorHandle, into_contiguous_pitched},
    },
    prelude::*,
};
use cubecl_std::{FastDivmod, FastDivmodArgs};

use crate::{
    CubeElement, CubeRuntime, FloatElement,
    kernel::{
        AddOp,
        conv::index,
        into_contiguous, launch_binop,
        matmul::{MatmulStrategy, matmul},
    },
    ops::{
        max_line_size,
        numeric::{empty_device, empty_device_strided},
        permute, permute_nchw_to_nhwc, permute_nhwc_to_nchw, reshape, swap_dims,
    },
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct Im2ColArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: u32,
    padding_w: u32,
}

#[cube(launch_unchecked)]
fn im2col_kernel<E: Numeric>(
    image: &Tensor<Line<E>>,
    columns: &mut Tensor<Line<E>>,
    args: &Im2ColArgs,
    shape_pos: FastDivmod,
    shape_m: Sequence<FastDivmod>,
    shape_k: Sequence<FastDivmod>,
    #[comptime] elems_per_thread: u32,
    #[comptime] has_padding: bool,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let height = image.shape(1);
    let width = image.shape(2);

    let line_size = image.line_size();
    let pos = ABSOLUTE_POS * elems_per_thread * line_size;

    if pos >= columns.len() {
        terminate!();
    }

    #[unroll]
    for i in 0..elems_per_thread {
        let pos = pos + i * line_size;
        let (pos_k, pos_m) = shape_pos.div_mod(pos);

        let (rem, out_x) = shape_m.index(1).div_mod(pos_m);
        let (b, out_y) = shape_m.index(0).div_mod(rem);

        let (rem, in_c) = shape_k.index(1).div_mod(pos_k);
        let (kernel_y, kernel_x) = shape_k.index(0).div_mod(rem);

        let pos_col = (pos_m * columns.stride(0) + pos_k * columns.stride(1)) / line_size;
        let image_offs = b * image.stride(0) + in_c * image.stride(3);

        if has_padding {
            let y =
                (out_y * args.stride_h + kernel_y * args.dilation_h) as i32 - args.padding_h as i32;
            let x =
                (out_x * args.stride_w + kernel_x * args.dilation_w) as i32 - args.padding_w as i32;
            if y >= 0 && x >= 0 && y < height as i32 && x < width as i32 {
                let pos_image =
                    image_offs + y as u32 * image.stride(1) + x as u32 * image.stride(2);
                columns[pos_col] = image[pos_image / line_size];
            } else {
                columns[pos_col] = Line::empty(line_size).fill(E::from_int(0))
            };
        } else {
            let y = out_y * args.stride_h + kernel_y * args.dilation_h;
            let x = out_x * args.stride_w + kernel_x * args.dilation_w;
            let pos_image = image_offs + y * image.stride(1) + x * image.stride(2);
            columns[pos_col] = image[pos_image / line_size];
        }
    }
}

#[cfg(not(test))]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_h: usize,
    out_w: usize,
) -> Result<usize, ConvLaunchError> {
    use cubecl::linalg::matmul::kernels::MatmulAvailabilityError;

    let cube_count_per_batch = (out_h * out_w).div_ceil(cubecl::PLANE_DIM_APPROX);
    let max_cube_count = u16::MAX as usize;
    let max_simultaneous = (max_cube_count / cube_count_per_batch).min(batch_size);
    if max_simultaneous == 0 {
        return Err(MatmulAvailabilityError::CubeCountTooBig(CubeCount::Static(
            cube_count_per_batch as u32,
            1,
            1,
        ))
        .into());
    }
    Ok((0..=max_simultaneous)
        .rev()
        .find(|per_run| batch_size % per_run == 0)
        .expect("Logically not possible"))
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_h: usize,
    out_w: usize,
) -> Result<usize, ConvLaunchError> {
    Ok(1)
}

fn im2col<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    options: ConvOptions<2>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
) -> CubeTensor<R> {
    let client = input.client.clone();
    let input = into_contiguous(input);

    let [batch_size, height, width, in_channels] = input.shape.dims();

    let line_size = max_line_size(&input);
    let mut elems_per_thread = 16usize.div_ceil(line_size as usize);

    let k = in_channels * kernel_h * kernel_w;
    let m = batch_size * out_h * out_w;
    let shape_col = Shape::new([m, k]);
    let columns =
        empty_device_strided::<R, E>(input.client.clone(), input.device.clone(), shape_col);

    let num_elems = columns.shape.num_elements();
    while num_elems % elems_per_thread != 0 && elems_per_thread > 1 {
        elems_per_thread /= 2;
    }

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems / elems_per_thread / line_size as usize, cube_dim);

    let shape_pos = FastDivmodArgs::new(&client, k as u32);
    let mut shape_m = SequenceArg::new();
    let mut shape_k = SequenceArg::new();

    shape_m.push(FastDivmodArgs::new(&client, height as u32));
    shape_m.push(FastDivmodArgs::new(&client, width as u32));

    shape_k.push(FastDivmodArgs::new(&client, kernel_w as u32));
    shape_k.push(FastDivmodArgs::new(&client, in_channels as u32));

    unsafe {
        im2col_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_handle_ref().as_tensor_arg(line_size),
            columns.as_handle_ref().as_tensor_arg(line_size),
            Im2ColArgsLaunch::new(
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.padding[0] as u32),
                ScalarArg::new(options.padding[1] as u32),
            ),
            shape_pos,
            shape_m,
            shape_k,
            elems_per_thread as u32,
            options.padding != [0, 0],
        )
    };

    columns
}

/// Perform a 2D convolution using the GEMM (im2col) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_im2col<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    if let Ok(out) =
        conv2d_im2col_1x1::<R, E>(input.clone(), weight.clone(), bias.clone(), options.clone())
    {
        return Ok(out);
    }

    let [batch_size, in_channels, in_height, in_width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let groups = options.groups;
    let out_c_per_group = out_channels / groups;

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

    let input = permute_nchw_to_nhwc(input);
    let weight = permute_nchw_to_nhwc(weight);

    let batches_per_run = batches_per_run(batch_size, out_h, out_w)?;
    let shape_m = batches_per_run * out_h * out_w;
    let shape_n = out_c_per_group;
    let matmul_shape = Shape::new([groups, shape_m, shape_n]);

    let mut out = if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;
        let out_shape = Shape::new([runs, batches_per_run, out_h, out_w, out_channels]);
        let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), out_shape);
        let in_shape = Shape::new([runs, batches_per_run, in_height, in_width, in_channels]);
        let input = reshape(input, in_shape);
        let in_shape_run = Shape::new([batches_per_run, in_height, in_width, in_channels]);

        for run in 0..runs {
            let input = index::<R, E>(input.clone(), run);
            let input = reshape(input, in_shape_run.clone());
            let out_slice = index::<R, E>(out.clone(), run);
            let out_slice = reshape(out_slice, matmul_shape.clone());
            execute::<R, E>(
                input,
                weight.clone(),
                out_slice,
                options.clone(),
                out_h,
                out_w,
            )?;
        }
        reshape(out, Shape::new([batch_size, out_h, out_w, out_channels]))
    } else {
        let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), matmul_shape);
        execute::<R, E>(input, weight, out.clone(), options, out_h, out_w)?;
        let out = permute(out, &[1, 0, 2]);
        reshape(out, Shape::new([batch_size, out_h, out_w, out_channels]))
    };

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, 1, 1, out_channels]));
        out = launch_binop::<R, E, AddOp>(out, bias)
    }

    Ok(permute_nhwc_to_nchw(out))
}

pub fn conv2d_im2col_1x1<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    let client = input.client.clone();
    let device = input.device.clone();

    let [batch_size, _, height, width] = input.shape.dims();
    let [out_channels, in_c_per_grp, kernel_h, kernel_w] = weight.shape.dims();
    let groups = options.groups;
    let out_c_per_group = out_channels / groups;

    let out_h = calculate_conv_output_size(
        kernel_h,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        width,
    );

    if kernel_h != 1 || kernel_w != 1 || height != out_h || width != out_w {
        return Err(ConvLaunchError::Unknown);
    }

    let input = reshape_input::<R, E>(input, groups); // [groups, N, H, W, C]

    // Efficient permutation that takes the stride required for TMA into account
    let mut weight = if weight.strides[1] != 1 {
        // Remove kernel dims so padded dim is channels
        weight.shape.dims = vec![out_channels, in_c_per_grp]; // [(groups, N), K]
        weight.strides = vec![weight.strides[0], weight.strides[1]];
        // Pitched contiguous to skip running another kernel for TMA
        let contiguous = into_contiguous_pitched::<R, E>(&input.client, &weight.as_handle_ref());
        from_handle(&client, &device, contiguous)
    } else {
        // Already compatible, skip initial reshape
        weight
    };
    weight.shape.dims = vec![groups, out_c_per_group, in_c_per_grp]; // [groups, N, K]
    weight.strides = vec![weight.strides[0] * out_c_per_group, weight.strides[0], 1];

    // Permute to N-major, while keeping memory layout K-major. K-major for both sides is the most
    // efficient for matmul, and allows skipping a contiguous kernel
    let weight = swap_dims(weight, 1, 2); // [groups, K, N]

    let out = matmul::<R, E>(input, weight, None, MatmulStrategy::default())?;
    let mut out = permute(out, &[1, 0, 2]);
    let mut out = if groups > 1 {
        reshape(out, Shape::new([batch_size, height, width, out_channels]))
    } else {
        let stride_w = out.strides[1];
        let stride_h = stride_w * width;
        out.shape.dims = vec![batch_size, height, width, out_channels];
        out.strides = vec![stride_h * height, stride_h, stride_w, out.strides[2]];
        out
    };

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, 1, 1, out_channels]));
        out = launch_binop::<R, E, AddOp>(out, bias)
    }

    Ok(permute_nhwc_to_nchw(out))
}

/// Reshapes NCHW input to [groups, N, H, W, C]
fn reshape_input<R: CubeRuntime, E: CubeElement>(
    mut input: CubeTensor<R>,
    groups: usize,
) -> CubeTensor<R> {
    let [batch_size, in_c, height, width] = input.shape.dims();
    let in_c_per_grp = in_c / groups;

    input.shape.dims = vec![batch_size, groups, in_c_per_grp, height, width];
    input.strides = vec![
        input.strides[0],
        input.strides[1] * in_c_per_grp,
        input.strides[1],
        input.strides[2],
        input.strides[3],
    ];
    let mut input = permute(input, &[1, 0, 3, 4, 2]); // [groups, NHWC]
    if !is_spatial_contiguous(&input.shape.dims, &input.strides) {
        let contiguous = into_contiguous_pitched::<R, E>(&input.client, &input.as_handle_ref());
        input = from_handle(&input.client, &input.device, contiguous);
    }
    input.shape.dims = vec![groups, batch_size * height * width, in_c_per_grp]; // [groups, M, K]
    input
}

fn is_spatial_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let rank = shape.len();

    let mut ordered = strides.to_vec();
    ordered.sort();
    if ordered != strides {
        return false;
    }

    for i in (1..rank - 2).rev() {
        if strides[i + 1] * shape[i + 1] != strides[i] {
            return false;
        }
    }
    true
}

fn from_handle<R: CubeRuntime, E: CubeElement>(
    client: &ComputeClient<R::Server, R::Channel>,
    device: &R::Device,
    handle: TensorHandle<R, E>,
) -> CubeTensor<R> {
    CubeTensor::new(
        client.clone(),
        handle.handle,
        handle.shape.into(),
        device.clone(),
        handle.strides,
        E::dtype(),
    )
}

fn execute<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    out: CubeTensor<R>,
    options: ConvOptions<2>,
    out_h: usize,
    out_w: usize,
) -> Result<(), ConvLaunchError> {
    let [out_channels, kernel_h, kernel_w, _] = weight.shape.dims();
    let groups = options.groups;

    let columns = im2col::<R, E>(input, options.clone(), kernel_h, kernel_w, out_h, out_w);
    let [shape_m, shape_k] = columns.shape.dims();
    let shape_m = shape_m / groups;
    let shape_n = out_channels / groups;

    let columns = reshape(columns, Shape::new([groups, shape_m, shape_k]));
    let weight = reshape(weight, Shape::new([groups, shape_n, shape_k]));
    let weight = swap_dims(weight, 1, 2); // Col-major [K, N]

    matmul::<R, E>(columns, weight, Some(out), Default::default())?;

    Ok(())
}
