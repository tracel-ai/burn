use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions, FloatTensorOps},
    Shape,
};
use cmma::{Matrix, MatrixIdent, MatrixLayout};
use cubecl::{
    comptime, cube,
    ir::{Elem, FloatKind},
    prelude::*,
    Compiler, CubeCount, CubeDim, Feature,
};
use half::f16;

use crate::{
    kernel::into_contiguous,
    ops::{numeric::empty_device, permute, reshape},
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

const WARP_SIZE: u32 = 32;

const CMMA_M: u32 = 16;
const CMMA_N: u32 = 16;
const CMMA_K: u32 = 16;
const CMMA_INPUT_TILE_SIZE: u32 = CMMA_M * CMMA_K;
const CMMA_FILTER_TILE_SIZE: u32 = CMMA_K * CMMA_N;
const CMMA_OUT_TILE_SIZE: u32 = CMMA_M * CMMA_N;

const WARPS_PER_CUBE: u32 = 8;

const CUBE_DIM_X: u32 = 128;
const CUBE_DIM_Y: u32 = 2;

const _: () = assert!(CUBE_DIM_Y * CUBE_DIM_X / WARP_SIZE == WARPS_PER_CUBE);

/// Perform a 2D convolution using the implicit GEMM algorithm. Requries `cmma` to be available.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_implicit_gemm<R: JitRuntime, F: FloatElement, I: IntElement>(
    input: JitTensor<R, F, 4>,
    weight: JitTensor<R, F, 4>,
    bias: Option<JitTensor<R, F, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, F, 4> {
    let [batch_size, in_channels, height, width] = input.shape.dims;
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims;

    let input = into_contiguous(input);

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

    let out_shape = Shape::new([batch_size, out_h, out_w, out_channels]);
    let mut out = empty_device(input.client.clone(), input.device.clone(), out_shape);

    // Implicit GEMM matrix size
    let gemm_m = (batch_size * out_h * out_w) as u32;
    let gemm_n = out_channels as u32;
    let gemm_k = in_channels * kernel_h * kernel_w;
    let slice_size = kernel_h * kernel_w * in_channels;

    // `CUBE_DIM_X` must be a multiple of `WARP_SIZE`
    // 128x2 means we have 8 warps and a cube computes a 32x64 output tile
    let cube_dim = CubeDim {
        x: CUBE_DIM_X,
        y: CUBE_DIM_Y,
        z: 1,
    };

    let cube_count_x = gemm_m.div_ceil(CMMA_M * CUBE_DIM_X / WARP_SIZE);
    let cube_count_y = gemm_n.div_ceil(CMMA_N * CUBE_DIM_Y);

    let cube_count = CubeCount::Static(cube_count_x, cube_count_y, 1);

    unsafe {
        implicit_gemm_kernel::launch_unchecked::<F, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg(1),
            weight.as_tensor_arg(1),
            out.as_tensor_arg(1),
            ScalarArg::new(gemm_m),
            ScalarArg::new(gemm_n),
            ScalarArg::new(gemm_k as u32),
            ScalarArg::new(slice_size as u32),
            KernelArgsLaunch::new(
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(options.padding[0] as i32),
                ScalarArg::new(options.padding[1] as i32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
            ),
            kernel_h as u32,
            kernel_w as u32,
        )
    };

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, 1, 1, out_channels]));
        out = JitBackend::<R, F, I>::float_add(out, bias);
    }

    permute(out, [0, 3, 1, 2])
}

#[derive(CubeLaunch)]
struct KernelArgs {
    stride_h: u32,
    stride_w: u32,
    pad_h: i32,
    pad_w: i32,
    dilation_h: u32,
    dilation_w: u32,
}

#[cube(launch_unchecked)]
fn implicit_gemm_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    out: &mut Tensor<F>,
    gemm_m: u32,
    gemm_n: u32,
    gemm_k: u32,
    slice_size: u32,
    args: &KernelArgs,
    #[comptime] kernel_h: u32,
    #[comptime] kernel_w: u32,
) {
    let kernel_size = comptime! { kernel_h * kernel_w };

    let height = input.shape(2);
    let width = input.shape(3);
    let h = height as i32;
    let w = width as i32;
    let out_h = out.shape(1);
    let out_w = out.shape(2);

    // Row strides in the implicit GEMM matrix
    let batch_stride = out_h * out_w;
    let y_stride = out_w;
    let x_stride = 1;

    // Tile using a 2D grid (over the output), each threadblock
    // is (128, 2) -> (4,2) = 8 warps -> 32x64 output
    let global_warp_m = ABSOLUTE_POS_X / WARP_SIZE;
    let global_warp_n = ABSOLUTE_POS_Y;
    let cube_warp_m = UNIT_POS_X / WARP_SIZE;
    let cube_warp_n = UNIT_POS_Y;
    let num_warps_m = comptime! { CUBE_DIM_X / WARP_SIZE };
    let intra_warp_unit_idx = UNIT_POS_X % WARP_SIZE; // Thread idx within warp (0 to 31)
    let cube_linear_warp_idx = (cube_warp_n * num_warps_m) + cube_warp_m; // Warp idx within a block (0 to WARPS_PER_BLOCK - 1)

    // Shared memory tiles, currently only holds enough data for
    // each warp to have its own tile for a single MMA op (8 * 16 * 16 elements)
    // conceptually a WARPS_PER_CUBE x (CMMA_M * CMMA_K) matrix
    let mut smem_input_tile = SharedMemory::<f16>::new(CMMA_INPUT_TILE_SIZE * WARPS_PER_CUBE);
    let mut smem_weight_tile = SharedMemory::<f16>::new(CMMA_FILTER_TILE_SIZE * WARPS_PER_CUBE);

    let matrix_a = Matrix::<f16>::new(
        MatrixIdent::A,
        CMMA_M,
        CMMA_N,
        CMMA_K,
        MatrixLayout::RowMajor,
    );
    let matrix_b = Matrix::<f16>::new(
        MatrixIdent::B,
        CMMA_M,
        CMMA_N,
        CMMA_K,
        MatrixLayout::RowMajor,
    );
    let matrix_acc = Matrix::<F>::new(
        MatrixIdent::Accumulator,
        CMMA_M,
        CMMA_N,
        CMMA_K,
        MatrixLayout::Undefined,
    );
    cmma::fill(&matrix_acc, F::new(0.0));

    let input_tile_start = cube_linear_warp_idx * CMMA_INPUT_TILE_SIZE;
    let weight_tile_start = cube_linear_warp_idx * CMMA_FILTER_TILE_SIZE;
    let input_tile =
        smem_input_tile.slice_mut(input_tile_start, input_tile_start + CMMA_INPUT_TILE_SIZE);
    let weight_tile =
        smem_weight_tile.slice_mut(weight_tile_start, weight_tile_start + CMMA_FILTER_TILE_SIZE);

    let a_row = global_warp_m * CMMA_M;
    let b_col = global_warp_n * CMMA_N;

    // Loop over the K-dimension
    for k in range_stepped(0, gemm_k, CMMA_K) {
        // Load into smem...
        // Each warp should load the 16x16 tile it's responsible for
        // i.e. each thread needs to load 8 elements of input and 8 elements of weight

        // Start index within a slice (0 to `kernel_size * channels - 1`) that a half warp (16 units) is responsible for
        let slice_start_idx = k % slice_size;

        /**************************** Loading Input Tile ************************************/
        for m in range_stepped(intra_warp_unit_idx, CMMA_INPUT_TILE_SIZE, WARP_SIZE) {
            // Compute where in the slice we are starting

            // Slices are always `kernel_size * channels` elements wide so we can compute where inside a slice
            // we are and also which row the slice is in relative to the start of the CMMA matrix

            let rel_slice_row = m / CMMA_K; // Relative row (0 - 15)
            let abs_slice_row = a_row + rel_slice_row; // Row of the matrix the slice is on

            // Actual index within a slice (0 to `kernel_size * channels - 1`) that the thread is repsonsible for
            let my_slice_idx = (slice_start_idx + (m % CMMA_K)) % slice_size;

            // Given the row of the matrix that the slice is in, and the index of the thread
            // within a slice, want to compute what input element to load...
            // first compute coordinates in output space (center of the kernel in MxK matrix A)
            let batch = abs_slice_row / batch_stride;
            let out_y = (abs_slice_row % batch_stride) / y_stride;
            let out_x = ((abs_slice_row % batch_stride) % y_stride) / x_stride;

            let channel = my_slice_idx / kernel_size;

            let kernel_y = (my_slice_idx / kernel_w) % kernel_h;
            let kernel_x = my_slice_idx % kernel_w;
            let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32 - args.pad_h;
            let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32 - args.pad_w;

            if x >= 0 && x < w && y >= 0 && y < h {
                input_tile[m] = f16::cast_from(
                    input[batch * input.stride(0)
                        + y as u32 * input.stride(2)
                        + x as u32 * input.stride(3)
                        + channel * input.stride(1)],
                );
            } else {
                input_tile[m] = f16::new(0.0);
            }
        }

        /**************************** Loading Weight Tile ***********************************/
        for n in range_stepped(intra_warp_unit_idx, CMMA_FILTER_TILE_SIZE, WARP_SIZE) {
            // Compute where in the slice we are starting
            let rel_slice_row = n / CMMA_K; // Relative row (0 - 15)
            let abs_slice_row = k + rel_slice_row; // Row of the matrix the slice is on
            let abs_slice_col = b_col + (n % 16); // Row of the matrix the slice is on

            // Given the row of the matrix that the slice is in, and the index of the unit
            // within a slice, want to compute what weight element to load...
            let out_channel = abs_slice_col;
            let in_channel = abs_slice_row / kernel_size;
            let kernel_y = (abs_slice_row % kernel_size) / kernel_h;
            let kernel_x = abs_slice_row % kernel_w;

            weight_tile[n] = f16::cast_from(
                weight[out_channel * weight.stride(0)
                    + in_channel * weight.stride(1)
                    + kernel_y * weight.stride(2)
                    + kernel_x * weight.stride(3)],
            );
        }

        /**************************** Bounds Check + CMMA Op*********************************/
        if a_row < gemm_m && k < gemm_k && b_col < gemm_n {
            cmma::load(&matrix_a, input_tile.as_slice(), CMMA_K);
            cmma::load(&matrix_b, weight_tile.as_slice(), CMMA_N);

            cmma::execute::<f16, f16, F, F>(&matrix_a, &matrix_b, &matrix_acc, &matrix_acc);
        }
    }

    let c_col = global_warp_n * CMMA_N;
    let c_row = global_warp_m * CMMA_M;

    if c_row < gemm_m && c_col < gemm_n {
        let out_pos = c_col + c_row * gemm_n;
        let out = out.slice_mut(out_pos, out_pos + CMMA_OUT_TILE_SIZE);
        cmma::store(out, &matrix_acc, gemm_n, MatrixLayout::RowMajor);
    }
}

pub(crate) fn can_do_implicit_gemm<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R, E, 4>,
    weight: &JitTensor<R, E, 4>,
    options: &ConvOptions<2>,
    out_h: usize,
    out_w: usize,
) -> bool {
    let [batch_size, in_channels, _, _] = input.shape.dims;
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims;

    let gemm_m = batch_size * out_h * out_w;
    let gemm_n = out_channels;
    let gemm_k = in_channels * kernel_h * kernel_w;

    let smem_size = ((CMMA_INPUT_TILE_SIZE + CMMA_FILTER_TILE_SIZE) * WARPS_PER_CUBE) as usize
        * size_of::<f16>();

    cmma_available::<R>(&input.device)
        && <R::Compiler as Compiler>::max_shared_memory_size() >= smem_size
        && gemm_m % 16 == 0
        && gemm_n % 16 == 0
        && gemm_k % 16 == 0
        && options.groups == 1
}

fn cmma_available<R: JitRuntime>(device: &R::JitDevice) -> bool {
    R::client(device).features().enabled(Feature::Cmma {
        a: Elem::Float(FloatKind::F16),
        b: Elem::Float(FloatKind::F16),
        c: Elem::Float(FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    })
}
