use burn_backend::{
    DType, Shape,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use core::iter;
use cubecl::std::{FastDivmod, FastDivmodArgs};
use cubecl::{
    calculate_cube_count_elemwise, intrinsic,
    prelude::*,
    std::tensor::{TensorHandle, into_contiguous_pitched},
};
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    kernel::{
        AddOp,
        conv::index,
        into_contiguous_aligned, launch_binop,
        matmul::{MatmulStrategy, matmul},
        utils::{merge_dims, split_dim},
    },
    ops::{
        max_line_size,
        numeric::{empty_device_dtype, empty_device_optimized_dtype},
        reshape, swap_dims,
    },
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType, Clone)]
pub(crate) struct ConvParam {
    pub stride: u32,
    pub dilation: u32,
    pub padding: i32,
}

#[cube(launch_unchecked)]
fn im2col_kernel<E: Numeric>(
    image: &Tensor<Line<E>>,
    columns: &mut Tensor<Line<E>>,
    params: Sequence<ConvParam>,
    shape_pos: FastDivmod,
    shape_in_c: FastDivmod,
    shape_out: Sequence<FastDivmod>,
    shape_kernel: Sequence<FastDivmod>,
    #[comptime] elems_per_thread: u32,
    #[comptime] has_padding: bool,
    #[define(E)] _dtype: StorageType,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let dim_c = image.rank() - 1;
    let n_spatial = comptime![shape_out.len()];

    let line_size = image.line_size();
    let pos = ABSOLUTE_POS * elems_per_thread;

    if pos >= columns.len() {
        terminate!();
    }

    let pos = pos * line_size;

    #[unroll]
    for i in 0..elems_per_thread {
        let pos = pos + i * line_size;
        let (pos_m, pos_k) = shape_pos.div_mod(pos);

        let (b, spatial_pos) = div_mod_seq(pos_m, &shape_out);

        let (rem, in_c) = shape_in_c.div_mod(pos_k);
        let (_, kernel_pos) = div_mod_seq(rem, &shape_kernel);

        let pos_col = (pos_m * columns.stride(0) + pos_k * columns.stride(1)) / line_size;

        let mut image_pos = b * image.stride(0) + in_c * image.stride(dim_c);
        let mut in_bounds = true;

        #[unroll]
        for i in 0..n_spatial {
            let i = unwrap_const(i);
            let out_pos = spatial_pos.index(i);
            let k_pos = kernel_pos.index(i);
            let params = params.index(i);
            let in_pos =
                (out_pos * params.stride + k_pos * params.dilation) as i32 - params.padding;
            image_pos += in_pos as u32 * image.stride(i + 1);

            if has_padding {
                in_bounds &= in_pos >= 0 && (in_pos as u32) < image.shape(i + 1);
            }
        }

        if in_bounds {
            columns[pos_col] = image[image_pos / line_size];
        } else {
            columns[pos_col] = Line::empty(line_size).fill(E::from_int(0))
        }
    }
}

#[cfg(not(test))]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_shape: usize,
) -> Result<usize, ConvSetupError> {
    use cubek::matmul::definition::MatmulAvailabilityError;

    let cube_count_per_batch = out_shape.div_ceil(cubecl::PLANE_DIM_APPROX);
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
        .find(|per_run| batch_size.is_multiple_of(*per_run))
        .expect("Logically not possible"))
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn batches_per_run(
    batch_size: usize,
    out_shape: usize,
) -> Result<usize, ConvSetupError> {
    Ok(1)
}

fn im2col<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    options: ConvOptions<N>,
    kernel_shape: &[usize],
    out_shape: &[usize],
) -> Result<CubeTensor<R>, LaunchError> {
    let client = input.client.clone();
    let input = into_contiguous_aligned(input);

    let rank = input.shape.num_dims();
    let batch_size = input.shape[0];
    let dim_c = rank - 1;
    let in_channels = input.shape[dim_c];

    let line_size = max_line_size(&input);
    let mut elems_per_thread = 16usize.div_ceil(line_size as usize);

    let k = in_channels * kernel_shape.iter().product::<usize>();
    let m = batch_size * out_shape.iter().product::<usize>();
    let shape_col = Shape::new([m, k]);
    let columns = empty_device_optimized_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_col,
        input.dtype,
    );

    let num_elems = columns.shape.num_elements() / line_size as usize;
    while !num_elems.is_multiple_of(elems_per_thread) && elems_per_thread > 1 {
        elems_per_thread /= 2;
    }

    let total_units = num_elems / elems_per_thread;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(total_units, cube_dim);

    let shape_pos = FastDivmodArgs::new(&client, k as u32);
    let shape_in_c = FastDivmodArgs::new(&client, in_channels as u32);
    let mut shape_out = SequenceArg::new();
    let mut shape_kernel = SequenceArg::new();

    for shape in out_shape {
        shape_out.push(FastDivmodArgs::new(&client, *shape as u32));
    }

    for shape in kernel_shape {
        shape_kernel.push(FastDivmodArgs::new(&client, *shape as u32));
    }

    let mut params = SequenceArg::new();

    for i in 0..N {
        params.push(ConvParamLaunch::new(
            ScalarArg::new(options.stride[i] as u32),
            ScalarArg::new(options.dilation[i] as u32),
            ScalarArg::new(options.padding[i] as i32),
        ))
    }

    unsafe {
        im2col_kernel::launch_unchecked(
            &input.client,
            cube_count,
            cube_dim,
            input.as_handle_ref().as_tensor_arg(line_size),
            columns.as_handle_ref().as_tensor_arg(line_size),
            params,
            shape_pos,
            shape_in_c,
            shape_out,
            shape_kernel,
            elems_per_thread as u32,
            options.padding.iter().any(|it| *it != 0),
            input.dtype.into(),
        )
    }?;

    Ok(columns)
}

/// Perform a 2D convolution using the GEMM (im2col) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_im2col<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    if let Ok(out) =
        conv_im2col_1x1::<R, N>(input.clone(), weight.clone(), bias.clone(), options.clone())
    {
        return Ok(out);
    }

    let rank = input.shape.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.shape[0];
    let in_shape = &input.shape[1..dim_c];
    let out_channels = weight.shape[0];
    let kernel_shape = &weight.shape[1..dim_c];

    let out_shape = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let out_shape_prod = out_shape.iter().product::<usize>();
    let batches_per_run = batches_per_run(batch_size, out_shape_prod)?;
    let shape_m = batches_per_run * out_shape_prod;
    let shape_n = out_channels;
    let matmul_shape = Shape::new([shape_m, shape_n]);

    let mut m_split = vec![batch_size];
    m_split.extend(out_shape.iter().copied());

    let mut out = if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;
        let shape_out = Shape::new([runs, batches_per_run, out_shape_prod, out_channels]);
        let out = empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            input.dtype,
        );
        let input = split_dim(input, 0, &[runs, batches_per_run]);

        for run in 0..runs {
            let input = index(input.clone(), run);
            let mut out_slice = index(out.clone(), run);
            out_slice.shape.dims = vec![shape_m, shape_n];
            out_slice.strides = vec![out_slice.strides[1], out_slice.strides[2]];
            execute::<R, N>(
                input,
                weight.clone(),
                &mut out_slice,
                options.clone(),
                &out_shape,
            )?;
        }
        let merged = merge_dims(out, 0, 1);
        split_dim(merged, 1, &out_shape)
    } else {
        let mut out = empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            matmul_shape,
            input.dtype,
        );
        execute::<R, N>(input, weight, &mut out, options, &out_shape)?;
        split_dim(out, 0, &m_split) // [N, H, W, C]
    };

    if let Some(bias) = bias {
        let mut bias_shape = iter::repeat_n(1, rank - 1).collect::<Vec<_>>();
        bias_shape.push(out_channels);
        let bias = reshape(bias, bias_shape.into());
        out = launch_binop::<R, AddOp>(out, bias);
    }

    Ok(out)
}

pub fn conv_im2col_1x1<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let rank = input.shape.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.shape[0];
    let in_channels = input.shape[dim_c];
    let in_shape = &input.shape[1..dim_c];
    let out_channels = weight.shape[0];
    let kernel_shape = &weight.shape[1..dim_c];

    if kernel_shape.iter().any(|s| *s != 1) {
        return Err(ConvSetupError::Unknown);
    }

    let out_shape = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let mut split_m = vec![batch_size];
    split_m.extend(out_shape.iter().copied());

    if kernel_shape.iter().any(|it| *it != 1) || in_shape != out_shape {
        return Err(ConvSetupError::Unknown);
    }

    let input = reshape_input(input); // [(NHW), C] : [M, K]
    let dtype = input.dtype;

    // Efficient permutation that takes the stride required for TMA into account
    let weight = if weight.strides[dim_c] != 1 {
        // Remove kernel dims so padded dim is channels
        weight.shape.dims = vec![out_channels, in_channels]; // [N, K]
        weight.strides = vec![weight.strides[0], weight.strides[dim_c]];
        // Pitched contiguous to skip running another kernel for TMA
        into_contiguous_aligned(weight)
    } else {
        // Already compatible, skip initial reshape
        weight.shape.dims = vec![out_channels, in_channels]; // [N, K]
        weight.strides = vec![weight.strides[0], 1];
        weight
    };

    // Permute to N-major, while keeping memory layout K-major. K-major for both sides is the most
    // efficient for matmul, and allows skipping a contiguous kernel
    let weight = swap_dims(weight, 0, 1); // [K, N]

    let out = matmul(input, weight, None, MatmulStrategy::default(), dtype)?; // [M, N]

    // Skip reshape to avoid potential `into_contiguous`. We're only splitting dims so it's safe.
    let mut out = split_dim(out, 0, &split_m); // [N, H, W, C]

    if let Some(bias) = bias {
        let mut bias_shape = iter::repeat_n(1, rank - 1).collect::<Vec<_>>();
        bias_shape.push(out_channels);
        let bias = reshape(bias, bias_shape.into());
        out = launch_binop::<R, AddOp>(out, bias);
    }

    Ok(out)
}

/// Reshapes NHWC input to [(N, H, W), C]
fn reshape_input<R: CubeRuntime>(mut input: CubeTensor<R>) -> CubeTensor<R> {
    let rank = input.shape.num_dims();
    let dim_c = rank - 1;
    let dtype = input.dtype;

    let batch_size = input.shape[0];
    let in_c: usize = input.shape[dim_c];
    let in_shape = input.shape[1..dim_c].to_vec();

    if !is_spatial_contiguous(&input.shape, &input.strides) {
        let contiguous =
            into_contiguous_pitched(&input.client, &input.as_handle_ref(), dtype.into())
                .expect("Kernel to never fail");
        input = from_handle(&input.client, &input.device, contiguous, dtype);
    }
    input.shape.dims = vec![batch_size * in_shape.iter().product::<usize>(), in_c]; // [M, K]
    input.strides = vec![input.strides[dim_c - 1], input.strides[dim_c]];
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

fn from_handle<R: CubeRuntime>(
    client: &ComputeClient<R>,
    device: &R::Device,
    handle: TensorHandle<R>,
    dtype: DType,
) -> CubeTensor<R> {
    CubeTensor::new(
        client.clone(),
        handle.handle,
        handle.shape.into(),
        device.clone(),
        handle.strides,
        dtype,
    )
}

fn execute<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    out: &mut CubeTensor<R>,
    options: ConvOptions<N>,
    out_shape: &[usize],
) -> Result<(), ConvSetupError> {
    let rank = weight.shape.num_dims();
    let dim_c = rank - 1;

    let out_channels = weight.shape[0];
    let kernel_shape = &weight.shape[1..dim_c];

    let columns = im2col::<R, N>(input, options.clone(), kernel_shape, out_shape)?;

    let [_, shape_k] = columns.shape.dims();
    let shape_n = out_channels;

    let weight = reshape(weight, Shape::new([shape_n, shape_k]));
    let weight = swap_dims(weight, 0, 1); // Col-major [K, N]

    let dtype = columns.dtype;
    matmul(
        columns,
        weight,
        Some(out.clone()),
        Default::default(),
        dtype,
    )?;

    Ok(())
}

#[cube]
pub(crate) fn div_mod_seq(pos: u32, shape: &Sequence<FastDivmod>) -> (u32, Sequence<u32>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let i = unwrap_const(i);
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}

#[allow(unused_variables)]
#[cube]
fn unwrap_const(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}
