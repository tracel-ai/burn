use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions, FloatTensorOps},
    Shape,
};
use cmma::{Matrix, MatrixIdent, MatrixLayout};
use cubecl::{cube, prelude::*, Compiler, CubeCount, CubeDim, Feature};
use half::f16;

use crate::{
    kernel::{into_contiguous, slice_assign},
    ops::{
        numeric::{empty_device, zeros_device},
        permute, reshape,
    },
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

/// Perform a 2D convolution using the implicit GEMM algorithm. Requries `cmma` to be available.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_implicit_gemm<R: JitRuntime, F: FloatElement, I: IntElement>(
    input: JitTensor<R, F>,
    weight: JitTensor<R, F>,
    bias: Option<JitTensor<R, F>>,
    options: ConvOptions<2>,
) -> JitTensor<R, F> {
    let [batch_size, mut in_channels, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let padded_channels = padded_in_channels(in_channels, kernel_h, kernel_w);

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

    if !can_do_implicit_gemm(&input, &weight, &options, out_h, out_w) {
        panic!(
            "Requirements for implicit GEMM not met:
- CMMA must be available
- `batch_size * out_h * out_w` must be divisible by 16
- `out_channels` must be divisible by 16
- `in_channels * kernel_h * kernel_w` must be divisible by 16
- `groups` must be 1
        "
        );
    }

    let (input, weight) = if padded_channels != in_channels {
        let input = permute(input, &[0, 2, 3, 1]);
        let weight = permute(weight, &[0, 2, 3, 1]);

        let in_shape = Shape::new([batch_size, height, width, padded_channels]);
        let in_slice = &[0..batch_size, 0..height, 0..width, 0..in_channels];
        let new_input = zeros_device(input.client.clone(), input.device.clone(), in_shape);
        let new_input = slice_assign(new_input, in_slice, input);

        let weight_shape = Shape::new([out_channels, kernel_h, kernel_w, padded_channels]);
        let weight_slice = &[0..out_channels, 0..kernel_h, 0..kernel_w, 0..in_channels];
        let new_weight = zeros_device(weight.client.clone(), weight.device.clone(), weight_shape);
        let new_weight = slice_assign(new_weight, weight_slice, weight);

        in_channels = padded_channels;
        (new_input, new_weight)
    } else {
        // channel last is more efficient even with the extra into_contiguous kernel
        let input = into_contiguous(permute(input, &[0, 2, 3, 1]));
        let weight = into_contiguous(permute(weight, &[0, 2, 3, 1]));
        (input, weight)
    };

    let out_shape = Shape::new([batch_size, out_h, out_w, out_channels]);
    let mut out = empty_device(input.client.clone(), input.device.clone(), out_shape);

    // Implicit GEMM matrix size
    let gemm_m = (batch_size * out_h * out_w) as u32;
    let gemm_n = out_channels as u32;
    let gemm_k = (in_channels * kernel_h * kernel_w) as u32;
    let slice_size = kernel_h * kernel_w * in_channels;

    let (cmma_m, cmma_n, cmma_k) =
        find_cmma_size::<R, f16, F>(&input.device, gemm_m, gemm_k, gemm_n).unwrap();

    let cube_dim_x = 128;
    let cube_dim_y = Ord::min(gemm_n.div_ceil(16), 2);

    let input_tile_size = cmma_m * cmma_k;
    let weight_tile_size = cmma_k * cmma_n;

    let warp_size = 32;
    let warps_per_cube = (cube_dim_y * cube_dim_x) / warp_size;

    let max_vectorization = u8::MAX; // TODO: Fetch this based on backend

    let input_elems_per_thread = input_tile_size / warp_size;
    let input_vectorization = u8::min(
        find_common_vec(in_channels, input_elems_per_thread),
        max_vectorization,
    );

    let weight_elems_per_thread = weight_tile_size / warp_size;
    let weight_vectorization = u8::min(weight_elems_per_thread as u8, max_vectorization);

    let settings = GemmSettings {
        cmma_m,
        cmma_n,
        cmma_k,
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
        out.as_tensor_arg(1),
        DimensionsLaunch::new(
            ScalarArg::new(gemm_m),
            ScalarArg::new(gemm_n),
            ScalarArg::new(gemm_k),
            ScalarArg::new(slice_size as u32),
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
        KernelSettings {
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            padding_h: options.padding[0] as i32,
            padding_w: options.padding[1] as i32,
            aligned,
        },
    );

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, 1, 1, out_channels]));
        out = JitBackend::<R, F, I>::float_add(out, bias);
    }
    // Reset to NCHW
    permute(out, &[0, 3, 1, 2])
}

fn find_common_vec(channels: usize, elems_per_thread: u32) -> u8 {
    let channels = channels as u8;
    let elems_per_thread = elems_per_thread as u8;
    let smaller = u8::min(channels, elems_per_thread);
    (1..=smaller)
        .rev()
        .find(|vec| channels % *vec == 0 && elems_per_thread % *vec == 0)
        .unwrap()
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

    out_h: u32,
    out_w: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct GemmSettings {
    cmma_m: u32,
    cmma_n: u32,
    cmma_k: u32,

    warp_size: u32,
    warps_per_cube: u32,

    cube_dim_x: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct KernelSettings {
    kernel_h: u32,
    kernel_w: u32,
    padding_h: i32,
    padding_w: i32,
    aligned: bool,
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
    out: &mut Tensor<F>,
    dims: &Dimensions,
    args: &ConvArgs,
    #[comptime] gemm_settings: GemmSettings,
    #[comptime] kernel_settings: KernelSettings,
) {
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

    if kernel_settings.aligned || pos.global_m < dims.gemm_m && pos.global_n < dims.gemm_n {
        execute_gemm(
            input,
            weight,
            out,
            input_tile,
            weight_tile,
            dims,
            &pos,
            args,
            gemm_settings,
            kernel_settings,
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
) -> Matrices<F, FAcc> {
    let GemmSettings {
        cmma_m,
        cmma_n,
        cmma_k,
        ..
    } = gemm_settings;

    let matrices = unsafe {
        Matrices::<F, FAcc> {
            a: Matrix::<F>::uninitialized(
                MatrixIdent::A,
                cmma_m,
                cmma_n,
                cmma_k,
                MatrixLayout::RowMajor,
            ),
            b: Matrix::<F>::uninitialized(
                MatrixIdent::B,
                cmma_m,
                cmma_n,
                cmma_k,
                MatrixLayout::ColMajor,
            ),
            acc: Matrix::<FAcc>::uninitialized(
                MatrixIdent::Accumulator,
                cmma_m,
                cmma_n,
                cmma_k,
                MatrixLayout::Undefined,
            ),
        }
    };

    cmma::fill(&matrices.acc, FAcc::new(0.0));

    matrices
}

#[cube]
fn execute_gemm<F: Float, FMat: Float>(
    input: &Tensor<Line<F>>,
    weight: &Tensor<Line<F>>,
    out: &mut SliceMut<F>,
    input_tile: &mut SliceMut<FMat>,
    weight_tile: &mut SliceMut<FMat>,
    dims: &Dimensions,
    pos: &Positions,
    args: &ConvArgs,
    #[comptime] g_settings: GemmSettings,
    #[comptime] k_settings: KernelSettings,
) {
    let GemmSettings { cmma_n, cmma_k, .. } = g_settings;

    let matrices = make_matrices::<FMat, F>(g_settings);

    // Loop over the K-dimension
    for k in range_stepped(0, dims.gemm_k, cmma_k) {
        // Load into smem...
        // Each warp should load the 16x16 tile it's responsible for
        // i.e. each thread needs to load 8 elements of input and 8 elements of weight

        load_input_tile(
            input, args, input_tile, dims, pos, k, g_settings, k_settings,
        );

        load_weight_tile(weight, weight_tile, pos, k, g_settings);

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
    #[comptime] kernel_settings: KernelSettings,
) {
    let GemmSettings {
        cmma_m,
        cmma_k,
        warp_size,
        ..
    } = gemm_settings;

    let KernelSettings {
        kernel_w,
        padding_h,
        padding_w,
        ..
    } = kernel_settings;

    let cmma_input_tile_size = cmma_m * cmma_k;
    let elems_per_thread = cmma_input_tile_size / warp_size;
    let vec = vectorization_of(input);

    let height = input.shape(1) as i32;
    let width = input.shape(2) as i32;
    let channels = input.shape(3);

    // Row strides in the implicit GEMM matrix
    let batch_stride = dims.out_h * dims.out_w;
    let y_stride = dims.out_w;
    let x_stride = 1;

    // Start index within a slice (0 to `kernel_size * channels - 1`) that a half warp (16 units) is responsible for
    let slice_start_idx = k % dims.slice_size;
    let start = pos.intra_warp_unit_idx * elems_per_thread;

    #[unroll]
    for m in range_stepped(0, elems_per_thread, vec) {
        let m = m + start;
        // Compute where in the slice we are starting

        // Slices are always `kernel_size * channels` elements wide so we can compute where inside a slice
        // we are and also which row the slice is in relative to the start of the CMMA matrix

        let rel_slice_row = m / cmma_k; // Relative row (0 - 15)
        let abs_slice_row = pos.global_m + rel_slice_row; // Row of the matrix the slice is on

        // Actual index within a slice (0 to `kernel_size * channels - 1`) that the thread is repsonsible for
        let my_slice_idx = (slice_start_idx + (m % cmma_k)) % dims.slice_size;

        // Given the row of the matrix that the slice is in, and the index of the thread
        // within a slice, want to compute what input element to load...
        // first compute coordinates in output space (center of the kernel in MxK matrix A)
        let batch = abs_slice_row / batch_stride;
        let out_y = (abs_slice_row % batch_stride) / y_stride;
        let out_x = ((abs_slice_row % batch_stride) % y_stride) / x_stride;

        let channel = my_slice_idx % channels;

        let kernel_x = (my_slice_idx / channels) % kernel_w;
        let kernel_y = my_slice_idx / (channels * kernel_w);

        let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32 - padding_h;
        let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32 - padding_w;
        let in_bounds =
            (padding_h == 0 && padding_w == 0) || (x >= 0 && x < width && y >= 0 && y < height);
        let idx = batch * input.stride(0)
            + y as u32 * input.stride(1)
            + x as u32 * input.stride(2)
            + channel;
        let value = select(
            in_bounds,
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
    pos: &Positions,
    k: u32,
    #[comptime] gemm_settings: GemmSettings,
) {
    let GemmSettings {
        cmma_n,
        cmma_k,
        warp_size,
        ..
    } = gemm_settings;

    let vec = vectorization_of(weight);
    let cmma_filter_tile_size = cmma_k * cmma_n;
    let elems_per_thread = cmma_filter_tile_size / warp_size;
    let start = pos.intra_warp_unit_idx * elems_per_thread;

    #[unroll]
    for n in range_stepped(0, elems_per_thread, vec) {
        let n = n + start;
        // Compute where in the slice we are starting
        let rel_slice_row = n % cmma_k; // Relative row (0 - 15)
        let abs_slice_row = k + rel_slice_row; // Row of the matrix the slice is on
        let abs_slice_col = pos.global_n + (n / cmma_k); // Row of the matrix the slice is on
        let idx = abs_slice_col * weight.stride(0) + abs_slice_row;
        let value = FMat::cast_from(weight[idx / vec]);

        #[unroll]
        for i in 0..vec {
            tile[n + i] = value[i];
        }
    }
}

pub(crate) fn can_do_implicit_gemm<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R, E>,
    weight: &JitTensor<R, E>,
    options: &ConvOptions<2>,
    out_h: usize,
    out_w: usize,
) -> bool {
    let [batch_size, in_channels, _, _] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let in_channels = padded_in_channels(in_channels, kernel_h, kernel_w);

    let gemm_m = batch_size * out_h * out_w;
    let gemm_n = out_channels;
    let gemm_k = in_channels * kernel_h * kernel_w;

    let size =
        find_cmma_size::<R, f16, E>(&input.device, gemm_m as u32, gemm_k as u32, gemm_n as u32);

    if let Some((cmma_m, cmma_k, cmma_n)) = size {
        let warps_per_cube = 8;

        let smem_size = ((cmma_m + cmma_n) * cmma_k * warps_per_cube) as usize * size_of::<f16>();

        <R::Compiler as Compiler>::max_shared_memory_size() >= smem_size && options.groups == 1
    } else {
        false
    }
}

fn padded_in_channels(in_channels: usize, kernel_h: usize, kernel_w: usize) -> usize {
    let kernel_size = kernel_h * kernel_w;
    let target = if kernel_size % 2 == 0 {
        (16usize).div_ceil(kernel_size)
    } else {
        16
    };
    if in_channels % target != 0 {
        let tiles = in_channels.div_ceil(target);
        tiles * target
    } else {
        in_channels
    }
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
                .features()
                .enabled(Feature::Cmma {
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
