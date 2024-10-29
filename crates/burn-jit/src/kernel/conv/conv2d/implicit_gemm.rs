use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use cmma::{Matrix, MatrixIdent, MatrixLayout};
use cubecl::{cube, prelude::*, Compiler, CubeCount, CubeDim, Feature};
use half::f16;

use crate::{
    kernel::{into_contiguous, slice},
    ops::{
        numeric::{empty_device, zeros_device},
        permute,
    },
    tensor::JitTensor,
    FloatElement, IntElement, JitRuntime,
};

/// Perform a 2D convolution using the implicit GEMM algorithm. Requires `cmma` to be available.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
#[allow(clippy::extra_unused_type_parameters)]
pub fn conv2d_implicit_gemm<R: JitRuntime, F: FloatElement, I: IntElement>(
    input: JitTensor<R, F>,
    weight: JitTensor<R, F>,
    bias: Option<JitTensor<R, F>>,
    options: ConvOptions<2>,
) -> JitTensor<R, F> {
    let [batch_size, in_channels, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let (pad_in_channels, pad_kh, pad_kw) = padded_k(in_channels, kernel_h, kernel_w);
    let padded_out_channels = out_channels.div_ceil(16) * 16;

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

    let padded_batch_size = padded_batch_size(batch_size, out_h, out_w);

    if !can_do_implicit_gemm::<R, F>(
        batch_size,
        in_channels,
        out_channels,
        [kernel_h, kernel_w],
        options.groups,
        out_h,
        out_w,
        &input.device,
    ) {
        panic!(
            "Requirements for implicit GEMM not met:
- CMMA must be available
- `groups` must be 1
        "
        );
    }

    let input = into_contiguous(permute(input, &[0, 2, 3, 1]));
    let weight = into_contiguous(permute(weight, &[0, 2, 3, 1]));

    let out_shape = Shape::new([padded_batch_size, out_h, out_w, padded_out_channels]);
    let out = empty_device(input.client.clone(), input.device.clone(), out_shape);

    // Implicit GEMM matrix size
    let gemm_m = (padded_batch_size * out_h * out_w) as u32;
    let gemm_n = padded_out_channels as u32;
    let gemm_k = (pad_in_channels * pad_kh * pad_kw) as u32;

    let slice_size = pad_kh * pad_kw * pad_in_channels;

    let (cmma_m, cmma_n, cmma_k) =
        find_cmma_size::<R, f16, F>(&input.device, gemm_m, gemm_k, gemm_n).unwrap();

    let cube_dim_x = 128;
    let cube_dim_y = Ord::min(gemm_n.div_ceil(16), 2);

    let input_tile_size = cmma_m * cmma_k;
    let weight_tile_size = cmma_k * cmma_n;
    let acc_tile_size = cmma_m * cmma_n;

    let warp_size = 32;
    let warps_per_cube = (cube_dim_y * cube_dim_x) / warp_size;

    let supported_vecs = R::supported_line_sizes();

    let input_elems_per_thread = input_tile_size / warp_size;
    let input_vectorization = find_common_vec(in_channels, input_elems_per_thread, supported_vecs);

    let weight_elems_per_thread = weight_tile_size / warp_size;
    let weight_vectorization =
        find_common_vec(in_channels, weight_elems_per_thread, supported_vecs);
    let bias_elems_per_thread = acc_tile_size / warp_size;
    let bias_vectorization = find_common_vec(out_channels, bias_elems_per_thread, supported_vecs);

    let has_bias = bias.is_some();
    let bias = bias.unwrap_or_else(|| {
        zeros_device(input.client.clone(), input.device.clone(), Shape::new([1]))
    });

    let settings = GemmSettings {
        cmma_m,
        cmma_n,
        cmma_k,
        check_m: batch_size != padded_batch_size,
        check_n: out_channels != padded_out_channels,
        check_k: (kernel_h * kernel_w * in_channels) as u32 != gemm_k,
        warp_size,
        warps_per_cube,
        cube_dim_x,
    };

    // `CUBE_DIM_X` must be a multiple of `WARP_SIZE`
    // 128x2 means we have 8 warps and a cube computes a 32x64 output tile
    let cube_dim = CubeDim {
        x: cube_dim_x,
        y: cube_dim_y,
        z: 1,
    };

    let cube_count_x = gemm_m.div_ceil(cmma_m * cube_dim_x / warp_size);
    let cube_count_y = gemm_n.div_ceil(cmma_n * cube_dim_y);

    // If div floor == div ceil then the cubes are aligned with the input dimensions
    let aligned = gemm_m / (cmma_m * cube_dim_x / warp_size) == cube_count_x
        && gemm_n / (cmma_n * cube_dim_y) == cube_count_y;

    let cube_count = CubeCount::Static(cube_count_x, cube_count_y, 1);

    implicit_gemm_kernel::launch::<F, f16, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(input_vectorization),
        weight.as_tensor_arg(weight_vectorization),
        bias.as_tensor_arg(bias_vectorization),
        out.as_tensor_arg(1),
        DimensionsLaunch::new(
            ScalarArg::new(gemm_m),
            ScalarArg::new(gemm_n),
            ScalarArg::new(gemm_k),
            ScalarArg::new(slice_size as u32),
            ScalarArg::new(pad_kw as u32),
            ScalarArg::new(pad_in_channels as u32),
            ScalarArg::new(out_h as u32),
            ScalarArg::new(out_w as u32),
        ),
        ConvArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.padding[0] as i32),
            ScalarArg::new(options.padding[1] as i32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
        ),
        settings,
        ConvSettings {
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            padding_h: options.padding[0] as i32,
            padding_w: options.padding[1] as i32,
            aligned,
            has_bias,
        },
    );

    let out = slice(out, &[0..batch_size, 0..out_h, 0..out_w, 0..out_channels]);

    // Reset to NCHW
    permute(out, &[0, 3, 1, 2])
}

fn find_common_vec(channels: usize, elems_per_thread: u32, supported_vecs: &[u8]) -> u8 {
    let channels = channels as u8;
    let elems_per_thread = elems_per_thread as u8;
    let smaller = u8::min(channels, elems_per_thread);
    (1..=smaller)
        .rev()
        .filter(|it| supported_vecs.contains(it))
        .find(|vec| channels % *vec == 0 && elems_per_thread % *vec == 0)
        .unwrap_or(1)
}

#[derive(CubeLaunch)]
struct ConvArgs {
    stride_h: u32,
    stride_w: u32,
    pad_h: i32,
    pad_w: i32,
    dilation_h: u32,
    dilation_w: u32,
}

#[derive(CubeLaunch)]
struct Dimensions {
    gemm_m: u32,
    gemm_n: u32,
    gemm_k: u32,
    slice_size: u32,

    pad_kw: u32,
    pad_channels: u32,

    out_h: u32,
    out_w: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct GemmSettings {
    cmma_m: u32,
    cmma_n: u32,
    cmma_k: u32,

    check_m: bool,
    check_n: bool,
    check_k: bool,

    warp_size: u32,
    warps_per_cube: u32,

    cube_dim_x: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ConvSettings {
    kernel_h: u32,
    kernel_w: u32,
    padding_h: i32,
    padding_w: i32,
    aligned: bool,
    has_bias: bool,
}

#[derive(Clone, Copy, CubeType)]
struct Positions {
    global_m: u32,
    global_n: u32,

    intra_warp_unit_idx: u32,
    cube_linear_warp_idx: u32,
}

#[derive(CubeType)]
struct Matrices<F: Float, FAcc: Float> {
    a: Matrix<F>,
    b: Matrix<F>,
    acc: Matrix<FAcc>,
}

#[allow(clippy::collapsible_else_if)]
#[cube(launch_unchecked, launch)]
fn implicit_gemm_kernel<F: Float, FMat: Float>(
    input: &Tensor<Line<F>>,
    weight: &Tensor<Line<F>>,
    bias: &Tensor<Line<F>>,
    out: &mut Tensor<F>,
    dims: &Dimensions,
    args: &ConvArgs,
    #[comptime] gemm_settings: GemmSettings,
    #[comptime] conv_settings: ConvSettings,
) {
    let _ = bias[0];

    let GemmSettings {
        cmma_m,
        cmma_n,
        cmma_k,
        warps_per_cube,
        ..
    } = gemm_settings;

    let cmma_out_tile_size = cmma_m * cmma_n;
    let cmma_input_tile_size = cmma_m * cmma_k;
    let cmma_filter_tile_size = cmma_k * cmma_n;

    let pos = calculate_positions(gemm_settings);

    // Shared memory tiles, currently only holds enough data for
    // each warp to have its own tile for a single MMA op (8 * 16 * 16 elements)
    // conceptually a WARPS_PER_CUBE x (CMMA_M * CMMA_K) matrix
    let mut smem_input_tile = SharedMemory::<FMat>::new(cmma_input_tile_size * warps_per_cube);
    let mut smem_weight_tile = SharedMemory::<FMat>::new(cmma_filter_tile_size * warps_per_cube);

    let input_tile_start = pos.cube_linear_warp_idx * cmma_input_tile_size;
    let weight_tile_start = pos.cube_linear_warp_idx * cmma_filter_tile_size;
    let input_tile =
        smem_input_tile.slice_mut(input_tile_start, input_tile_start + cmma_input_tile_size);
    let weight_tile =
        smem_weight_tile.slice_mut(weight_tile_start, weight_tile_start + cmma_filter_tile_size);

    let out_pos = pos.global_n + pos.global_m * dims.gemm_n;
    let out = out.slice_mut(out_pos, out_pos + cmma_out_tile_size);

    if conv_settings.aligned || pos.global_m < dims.gemm_m && pos.global_n < dims.gemm_n {
        execute_gemm(
            input,
            weight,
            bias,
            out,
            input_tile,
            weight_tile,
            dims,
            &pos,
            args,
            gemm_settings,
            conv_settings,
        );
    }
}

#[cube]
fn calculate_positions(#[comptime] gemm_settings: GemmSettings) -> Positions {
    let GemmSettings {
        cmma_m,
        cmma_n,
        warp_size,
        cube_dim_x,
        ..
    } = gemm_settings;

    // Tile using a 2D grid (over the output), each threadblock
    // is (128, 2) -> (4,2) = 8 warps -> 32x64 output
    let global_warp_m = ABSOLUTE_POS_X / warp_size;
    let global_warp_n = ABSOLUTE_POS_Y;
    let cube_warp_m = UNIT_POS_X / warp_size;
    let cube_warp_n = UNIT_POS_Y;
    let num_warps_m = cube_dim_x / warp_size;
    let intra_warp_unit_idx = UNIT_POS_X % warp_size; // Thread idx within warp (0 to 31)
    let cube_linear_warp_idx = (cube_warp_n * num_warps_m) + cube_warp_m; // Warp idx within a block (0 to WARPS_PER_BLOCK - 1)

    Positions {
        global_m: global_warp_m * cmma_m,
        global_n: global_warp_n * cmma_n,
        intra_warp_unit_idx,
        cube_linear_warp_idx,
    }
}

#[cube]
fn make_matrices<F: Float, FAcc: Float>(
    #[comptime] gemm_settings: GemmSettings,
    #[comptime] has_bias: bool,
) -> Matrices<F, FAcc> {
    let GemmSettings {
        cmma_m,
        cmma_n,
        cmma_k,
        ..
    } = gemm_settings;

    let acc = if has_bias {
        unsafe {
            Matrix::<FAcc>::uninitialized(
                MatrixIdent::Accumulator,
                cmma_m,
                cmma_n,
                cmma_k,
                MatrixLayout::Undefined,
            )
        }
    } else {
        Matrix::<FAcc>::from_value(
            MatrixIdent::Accumulator,
            cmma_m,
            cmma_n,
            cmma_k,
            MatrixLayout::Undefined,
            FAcc::new(0.0),
        )
    };

    Matrices::<F, FAcc> {
        a: unsafe {
            Matrix::<F>::uninitialized(
                MatrixIdent::A,
                cmma_m,
                cmma_n,
                cmma_k,
                MatrixLayout::RowMajor,
            )
        },
        b: unsafe {
            Matrix::<F>::uninitialized(
                MatrixIdent::B,
                cmma_m,
                cmma_n,
                cmma_k,
                MatrixLayout::ColMajor,
            )
        },
        acc,
    }
}

#[cube]
fn execute_gemm<F: Float, FMat: Float>(
    input: &Tensor<Line<F>>,
    weight: &Tensor<Line<F>>,
    bias: &Tensor<Line<F>>,
    out: &mut SliceMut<F>,
    input_tile: &mut SliceMut<FMat>,
    weight_tile: &mut SliceMut<FMat>,
    dims: &Dimensions,
    pos: &Positions,
    args: &ConvArgs,
    #[comptime] g_settings: GemmSettings,
    #[comptime] k_settings: ConvSettings,
) {
    let GemmSettings {
        cmma_m,
        cmma_n,
        cmma_k,
        warps_per_cube,
        ..
    } = g_settings;
    let has_bias = k_settings.has_bias;

    let matrices = make_matrices::<FMat, F>(g_settings, has_bias);
    if has_bias {
        let mut smem_bias = SharedMemory::new(cmma_m * cmma_n * warps_per_cube);
        load_bias_tile(bias, &mut smem_bias, pos, g_settings);
        cmma::load_with_layout(
            &matrices.acc,
            smem_bias.as_slice(),
            cmma_n,
            MatrixLayout::RowMajor,
        );
    }

    // Loop over the K-dimension
    for k in range_stepped(0, dims.gemm_k, cmma_k) {
        // Load into smem...
        // Each warp should load the 16x16 tile it's responsible for
        // i.e. each thread needs to load 8 elements of input and 8 elements of weight

        load_input_tile(
            input, args, input_tile, dims, pos, k, g_settings, k_settings,
        );

        load_weight_tile(weight, weight_tile, dims, pos, k, g_settings, k_settings);

        // Run CMMA
        cmma::load(&matrices.a, input_tile.as_slice(), cmma_k);
        cmma::load(&matrices.b, weight_tile.as_slice(), cmma_n);

        cmma::execute::<FMat, FMat, F, F>(&matrices.a, &matrices.b, &matrices.acc, &matrices.acc);
    }

    cmma::store(out, &matrices.acc, dims.gemm_n, MatrixLayout::RowMajor);
}

#[cube]
fn load_input_tile<F: Float, FMat: Float>(
    input: &Tensor<Line<F>>,
    args: &ConvArgs,
    tile: &mut SliceMut<FMat>,
    dims: &Dimensions,
    pos: &Positions,
    k: u32,
    #[comptime] gemm_settings: GemmSettings,
    #[comptime] kernel_settings: ConvSettings,
) {
    let GemmSettings {
        cmma_m,
        cmma_k,
        warp_size,
        check_m,
        check_k,
        ..
    } = gemm_settings;

    let ConvSettings {
        kernel_w,
        kernel_h,
        padding_h,
        padding_w,
        ..
    } = kernel_settings;

    let cmma_input_tile_size = cmma_m * cmma_k;
    let elems_per_thread = cmma_input_tile_size / warp_size;
    let vec = vectorization_of(input);

    let height = input.shape(1) as i32;
    let width = input.shape(2) as i32;
    let channels = dims.pad_channels;

    // Row strides in the implicit GEMM matrix
    let batch_stride = dims.out_h * dims.out_w;
    let y_stride = dims.out_w;
    let x_stride = 1;

    // Start index within a slice (0 to `kernel_size * channels - 1`) that a half warp (16 units) is responsible for
    let slice_start_idx = k % dims.slice_size;
    let start = pos.intra_warp_unit_idx * elems_per_thread;

    let rel_slice_row = start / cmma_k; // Relative row (0 - 15)
    let abs_slice_row = pos.global_m + rel_slice_row; // Row of the matrix the slice is on

    // Given the row of the matrix that the slice is in, and the index of the thread
    // within a slice, want to compute what input element to load...
    // first compute coordinates in output space (center of the kernel in MxK matrix A)
    let batch = abs_slice_row / batch_stride;

    let m_in_bounds = !check_m || batch < input.shape(0);
    let out_y = (abs_slice_row % batch_stride) / y_stride;
    let out_x = ((abs_slice_row % batch_stride) % y_stride) / x_stride;

    #[unroll]
    for m in range_stepped(0, elems_per_thread, vec) {
        let m = m + start;
        // Compute where in the slice we are starting

        // Slices are always `kernel_size * channels` elements wide so we can compute where inside a slice
        // we are and also which row the slice is in relative to the start of the CMMA matrix

        // Actual index within a slice (0 to `kernel_size * channels - 1`) that the thread is repsonsible for
        let my_slice_idx = (slice_start_idx + (m % cmma_k)) % dims.slice_size;

        let channel = my_slice_idx % channels;

        let kernel_x = (my_slice_idx / channels) % dims.pad_kw;
        let kernel_y = my_slice_idx / (channels * dims.pad_kw);

        let k_in_bounds =
            !check_k || (channel < input.shape(3) && kernel_x < kernel_w && kernel_y < kernel_h);

        let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32 - padding_h;
        let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32 - padding_w;
        let in_bounds =
            (padding_h == 0 && padding_w == 0) || (x >= 0 && x < width && y >= 0 && y < height);
        let idx = batch * input.stride(0)
            + y as u32 * input.stride(1)
            + x as u32 * input.stride(2)
            + channel;
        let value = select(
            in_bounds && m_in_bounds && k_in_bounds,
            FMat::cast_from(input[idx / vec]),
            FMat::vectorized(0.0, vec),
        );

        #[unroll]
        for i in 0..vec {
            tile[m + i] = value[i];
        }
    }
}

#[cube]
fn load_weight_tile<F: Float, FMat: Float>(
    weight: &Tensor<Line<F>>,
    tile: &mut SliceMut<FMat>,
    dims: &Dimensions,
    pos: &Positions,
    k: u32,
    #[comptime] gemm_settings: GemmSettings,
    #[comptime] kernel_settings: ConvSettings,
) {
    let GemmSettings {
        cmma_n,
        cmma_k,
        warp_size,
        check_n,
        check_k,
        ..
    } = gemm_settings;

    let ConvSettings {
        kernel_w, kernel_h, ..
    } = kernel_settings;

    let vec = vectorization_of(weight);
    let cmma_filter_tile_size = cmma_k * cmma_n;
    let elems_per_thread = cmma_filter_tile_size / warp_size;
    let start = pos.intra_warp_unit_idx * elems_per_thread;
    let abs_slice_col = pos.global_n + (start / cmma_k); // Row of the matrix the slice is on

    let n_in_bounds = !check_n || abs_slice_col < weight.shape(0);
    let col_idx = abs_slice_col * weight.stride(0);

    #[unroll]
    for n in range_stepped(0, elems_per_thread, vec) {
        let n = n + start;
        // Compute where in the slice we are starting
        let rel_slice_row = n % cmma_k; // Relative row (0 - 15)
        let abs_slice_row = k + rel_slice_row; // Row of the matrix the slice is on

        let (idx, k_in_bounds) = if check_k {
            let channel = abs_slice_row % dims.pad_channels;
            let kernel_x = abs_slice_row / dims.pad_channels % dims.pad_kw;
            let kernel_y = abs_slice_row / (dims.pad_channels * dims.pad_kw);
            let k_in_bounds = !check_k
                || (channel < weight.shape(3) && kernel_x < kernel_w && kernel_y < kernel_h);
            let idx = col_idx + kernel_y * weight.stride(1) + kernel_x * weight.stride(2) + channel;
            (idx, k_in_bounds)
        } else {
            (col_idx + abs_slice_row, true)
        };
        let value = FMat::cast_from(weight[idx / vec]);
        let value = select(k_in_bounds && n_in_bounds, value, FMat::new(0.0));

        #[unroll]
        for i in 0..vec {
            tile[n + i] = value[i];
        }
    }
}

#[cube]
fn load_bias_tile<F: Float>(
    bias: &Tensor<Line<F>>,
    tile: &mut SharedMemory<F>,
    pos: &Positions,
    #[comptime] gemm_settings: GemmSettings,
) {
    let GemmSettings {
        cmma_n,
        cmma_m,
        warp_size,
        ..
    } = gemm_settings;

    let vec = vectorization_of(bias);
    let cmma_acc_tile_size = cmma_m * cmma_n;
    let elems_per_thread = cmma_acc_tile_size / warp_size;
    let start = pos.intra_warp_unit_idx * elems_per_thread;
    let bias_tile_start = pos.cube_linear_warp_idx * cmma_acc_tile_size;

    #[unroll]
    for n in range_stepped(0, elems_per_thread, vec) {
        let n = n + start;

        let row = n % cmma_n + pos.global_n;
        let value = bias[row / vec];

        #[unroll]
        for i in 0..vec {
            tile[bias_tile_start + n + i] = value[i];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn can_do_implicit_gemm<R: JitRuntime, E: FloatElement>(
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    groups: usize,
    out_h: usize,
    out_w: usize,
    device: &R::Device,
) -> bool {
    let (in_channels, kernel_h, kernel_w) = padded_k(in_channels, kernel_size[0], kernel_size[1]);
    let batch_size = padded_batch_size(batch_size, out_h, out_w);
    let out_channels = out_channels.div_ceil(16) * 16;

    let gemm_m = batch_size * out_h * out_w;
    let gemm_n = out_channels;
    let gemm_k = in_channels * kernel_h * kernel_w;

    let size = find_cmma_size::<R, f16, E>(device, gemm_m as u32, gemm_k as u32, gemm_n as u32);

    if let Some((cmma_m, cmma_k, cmma_n)) = size {
        let warps_per_cube = 8;

        let smem_size = ((cmma_m + cmma_n) * cmma_k * warps_per_cube) as usize * size_of::<f16>();

        <R::Compiler as Compiler>::max_shared_memory_size() >= smem_size && groups == 1
    } else {
        false
    }
}

fn padded_k(in_channels: usize, kernel_h: usize, kernel_w: usize) -> (usize, usize, usize) {
    let target = 16;
    if in_channels * kernel_h * kernel_w % target == 0 {
        return (in_channels, kernel_h, kernel_w);
    }
    let kernel_h = kernel_h.next_power_of_two();
    let target = target.div_ceil(kernel_h);
    if in_channels * kernel_w % target == 0 {
        return (in_channels, kernel_h, kernel_w);
    }
    let kernel_w = kernel_w.next_power_of_two();
    let target = target.div_ceil(kernel_w);
    if in_channels % target == 0 {
        return (in_channels, kernel_h, kernel_w);
    }
    let in_channels = in_channels.div_ceil(target) * target;
    (in_channels, kernel_h, kernel_w)
}

fn padded_batch_size(batch_size: usize, out_h: usize, out_w: usize) -> usize {
    let out_size = out_h * out_w;
    let target = if out_size % 2 == 0 {
        (16usize).div_ceil(out_size)
    } else {
        16
    };
    batch_size.div_ceil(target) * target
}

fn find_cmma_size<R: JitRuntime, F: Float, FAcc: Float>(
    device: &R::JitDevice,
    gemm_m: u32,
    gemm_k: u32,
    gemm_n: u32,
) -> Option<(u32, u32, u32)> {
    supported_cmma_sizes::<R, F, FAcc>(device)
        .into_iter()
        .find(|(m, k, n)| {
            gemm_m % *m as u32 == 0 && gemm_k % *k as u32 == 0 && gemm_n % *n as u32 == 0
        })
        .map(|(m, k, n)| (m as u32, n as u32, k as u32))
}

fn supported_cmma_sizes<R: JitRuntime, F: Float, FAcc: Float>(
    device: &R::JitDevice,
) -> Vec<(u8, u8, u8)> {
    let requested_sizes = [(16, 16, 16), (32, 16, 8), (8, 16, 32)];

    requested_sizes
        .iter()
        .copied()
        .filter(|(m, k, n)| {
            R::client(device)
                .properties()
                .feature_enabled(Feature::Cmma {
                    a: F::as_elem(),
                    b: F::as_elem(),
                    c: FAcc::as_elem(),
                    m: *m,
                    k: *k,
                    n: *n,
                })
        })
        .collect()
}
