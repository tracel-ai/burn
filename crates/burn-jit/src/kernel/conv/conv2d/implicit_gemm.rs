use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions, FloatTensorOps},
    Shape,
};
use cmma::{Matrix, MatrixIdent, MatrixLayout};
use cubecl::{
    cube,
    ir::{Elem, FloatKind},
    prelude::*,
    Compiler, CubeCount, CubeDim, Feature,
};
use half::f16;

use crate::{
    ops::{numeric::empty_device, reshape},
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

const WARP_SIZE: u32 = 32;

const WMMA_M: u32 = 16;
const WMMA_N: u32 = 16;
const WMMA_K: u32 = 16;
const WMMA_INPUT_TILE_SIZE: u32 = WMMA_M * WMMA_K;
const WMMA_FILTER_TILE_SIZE: u32 = WMMA_K * WMMA_N;
const WMMA_OUT_TILE_SIZE: u32 = WMMA_M * WMMA_N;

const WARPS_PER_BLOCK: u32 = 8;

const CUBE_DIM_X: u32 = 128;
const CUBE_DIM_Y: u32 = 2;

const _: () = assert!(CUBE_DIM_Y * CUBE_DIM_X / WARP_SIZE == WARPS_PER_BLOCK);

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

    if !can_do_implicit_gemm(&input, &weight, out_h, out_w) {
        panic!(
            "Requirements for implicit GEMM not met:
- CMMA must be available
- batch_size * out_h * out_w must be divisible by 16
- out_channels must be divisible by 16
- in_channels * kernel_h * kernel_w must be divisible by 16
        "
        );
    }

    let out_shape = Shape::new([batch_size, out_channels, out_h, out_w]);
    let mut out = empty_device(input.client.clone(), input.device.clone(), out_shape);

    // Implicit GEMM matrix size
    let gemm_m = (batch_size * out_h * out_w) as u32;
    let gemm_n = out_channels as u32;
    let gemm_k = in_channels * kernel_h * kernel_w;
    let slice_size = kernel_h * kernel_w * in_channels;

    let cube_dim = CubeDim { x: 128, y: 2, z: 1 };

    let cube_count_x = gemm_m.div_ceil(WMMA_M * CUBE_DIM_X / 32);
    let cube_count_y = gemm_n.div_ceil(WMMA_N * CUBE_DIM_Y);

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
        let bias = reshape(bias, Shape::new([1, out_channels, 1, 1]));
        out = JitBackend::<R, F, I>::float_add(out, bias);
    }

    out
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
    let height = input.shape(2);
    let width = input.shape(3);

    // Row strides in the implicit GEMM matrix
    let batch_stride = height * width;
    let y_stride = width;
    let x_stride = 1;

    // Tile using a 2D grid (over the output), each threadblock
    // is (128, 2) -> (4,2) = 8 warps -> 32x64 output
    let global_warp_m = ABSOLUTE_POS_X / WARP_SIZE;
    let global_warp_n = ABSOLUTE_POS_Y;
    let block_warp_m = UNIT_POS_X / WARP_SIZE;
    let block_warp_n = UNIT_POS_Y;
    let num_warps_m = CUBE_DIM_X / WARP_SIZE;
    let intra_warp_thread_idx = UNIT_POS_X % WARP_SIZE; // Thread idx within warp (0 to 31)
    let block_linear_warp_idx = (block_warp_n * num_warps_m) + block_warp_m; // Warp idx within a block (0 to WARPS_PER_BLOCK - 1)

    let mut smem_input_tile = SharedMemory::<f16>::new(WARPS_PER_BLOCK * WMMA_M * WMMA_K);
    let mut smem_weight_tile = SharedMemory::<f16>::new(WARPS_PER_BLOCK * WMMA_K * WMMA_N);

    let matrix_a = Matrix::<f16>::new(
        MatrixIdent::A,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        MatrixLayout::RowMajor,
    );
    let matrix_b = Matrix::<f16>::new(
        MatrixIdent::B,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        MatrixLayout::RowMajor,
    );
    let matrix_acc = Matrix::<F>::new(
        MatrixIdent::Accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        MatrixLayout::Undefined,
    );
    cmma::fill(&matrix_acc, F::new(0.0));

    let input_tile_start = block_linear_warp_idx * WMMA_INPUT_TILE_SIZE;
    let weight_tile_start = block_linear_warp_idx * WMMA_FILTER_TILE_SIZE;
    let input_tile =
        smem_input_tile.slice_mut(input_tile_start, input_tile_start + WMMA_INPUT_TILE_SIZE);
    let weight_tile =
        smem_weight_tile.slice_mut(weight_tile_start, weight_tile_start + WMMA_FILTER_TILE_SIZE);

    // Loop over the K-dimension
    for i in range_stepped(0, gemm_k, WMMA_K) {
        let a_row = global_warp_m * WMMA_M;
        let a_col = i;
        let b_row = i;
        let b_col = global_warp_n * WMMA_N;

        // Load into smem...
        // Each warp should load the 16x16 tile it's responsible for
        // i.e. each thread needs to load 8 elements of input and 8 elements of weight
        // TODO optimize by only loading each value once, and then copying to the correct
        // positions (since one row is a repeat of the same 9 * C values - I call this
        // group of 9 * C values a slice)

        /**************************** Loading Input Tile ************************************/
        for j in range_stepped(intra_warp_thread_idx, WMMA_INPUT_TILE_SIZE, WARP_SIZE) {
            // Compute where in the slice we are starting, e.g. the following
            // depicts slices bounded by | | symbols, and the start and end
            // of one row of the 16x16 WMMA matrix
            // row 0: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
            //                       0  1  2  3  4  5  6  7  8   9 10 11 12 13 14 15
            // row 1: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
            //                      16 17 18 19 20 21 22 23 24  25 26 27 28 29 30 31
            // row 3: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
            //                      33 34 35 36 37 38 39 40 41  42 43 44 45 46 47 48

            // Slices are always 9 * C elements wide so we can compute where inside a slice
            // we are and also which row the slice is in relative to the start of the WMMA matrix

            let rel_slice_row = j / WMMA_K; // Relative row (0 - 15)
            let abs_slice_row = a_row + rel_slice_row; // Row of the matrix the slice is on

            // Start index within a slice (0 to 9*C-1) that a half warp (16 threads) is responsible for
            // int sliceStartIdx_old = (aCol + relSliceRow * GEMM_K) % SLICE_SIZE;
            let slice_start_idx = a_col % slice_size;

            // Actual index within a slice (0 to 9*C-1) that the thread is repsonsible for
            let my_slice_idx = (slice_start_idx + (j % WMMA_K)) % slice_size;

            // Given the row of the matrix that the slice is in, and the index of the thread
            // within a slice, want to compute what input element to load...
            // first compute coordinates in output space (center of the kernel in MxK matrix A)
            let batch = abs_slice_row / batch_stride;
            let out_y = (abs_slice_row % batch_stride) / y_stride;
            let out_x = ((abs_slice_row % batch_stride) % y_stride) / x_stride;

            let kernel_y = (my_slice_idx / kernel_w) % kernel_h;
            let kernel_x = my_slice_idx % kernel_w;
            let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32 - args.pad_h;
            let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32 - args.pad_w;
            // let offset_y = kernel_y as i32 - (kernel_h as i32 / 2);
            // let offset_x = kernel_x as i32 - (kernel_w as i32 / 2);

            // Computing the coordinates of the center of the slice / kernel
            // in MxK matrix A, and using this information to figure out where
            // current thread's slice element lies. We want the following mapping:
            // 0 -> (-1, -1)
            // 1 -> (0, -1)
            // 2 -> (1, -1)
            // 3 -> (-1, 0)
            // 4 -> (0, 0)
            // 5 -> (1, 0)
            // let y = out_y as i32 + offset_y;
            // let x = out_x as i32 + offset_x;
            let channel = my_slice_idx / kernel_h * kernel_w;
            let h = height as i32;
            let w = width as i32;

            if x >= 0 && x < w && y >= 0 && y < h {
                input_tile[j] = f16::cast_from(
                    input[batch * input.stride(0)
                        + channel * input.stride(1)
                        + y as u32 * input.stride(2)
                        + x as u32 * input.stride(3)],
                );
            } else {
                input_tile[j] = f16::new(0.0);
            }
        }

        /**************************** Loading Weight Tile ***********************************/
        for j in range_stepped(intra_warp_thread_idx, WMMA_FILTER_TILE_SIZE, WARP_SIZE) {
            // Compute where in the slice we are starting, e.g. the following
            // depicts slices bounded by | | symbols, and the start and end
            // of one row of the 16x16 WMMA matrix
            // row 0: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
            //                       0  1  2  3  4  5  6  7  8   9 10 11 12 13 14 15
            // row 1: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
            //                      16 17 18 19 20 21 22 23 24  25 26 27 28 29 30 31
            // row 3: [ *  *  * | *  s  *  *  *  *  *  *  * | *  *  *  *  *  *  *  e  * |  *  *  * ]
            //                      33 34 35 36 37 38 39 40 41  42 43 44 45 46 47 48

            // Slices are always 9 * C elements wide so we can compute where inside a slice
            // we are and also which row the slice is in
            // Relative to the start of the WMMA matrix
            // NOTE: each slice is identical in the filter matrix

            let rel_slice_row = j / WMMA_K; // Relative row (0 - 15)
            let abs_slice_row = b_row + rel_slice_row; // Row of the matrix the slice is on

            // int relSliceCol = j / WMMA_N;          // Relative col (0 - 15)
            let abs_slice_col = b_col + (j % 16); // Row of the matrix the slice is on

            // Given the row of the matrix that the slice is in, and the index of the thread
            // within a slice, want to compute what weight element to load...
            let out_channel = abs_slice_col;
            let in_channel = abs_slice_row / (kernel_h * kernel_w);
            let kernel_y = (abs_slice_row % (kernel_h * kernel_w)) / kernel_h;
            let kernel_x = abs_slice_row % kernel_w;

            weight_tile[j] = f16::cast_from(
                weight[out_channel * weight.stride(0)
                    + in_channel * weight.stride(1)
                    + kernel_y * weight.stride(2)
                    + kernel_x * weight.stride(3)],
            );
        }

        /**************************** Bounds Check + WMMA Op*********************************/
        if a_row < gemm_m && a_col < gemm_k && b_row < gemm_k && b_col < gemm_n {
            cmma::load(&matrix_a, input_tile.as_slice(), WMMA_K);
            cmma::load(&matrix_b, weight_tile.as_slice(), WMMA_N);

            cmma::execute::<f16, f16, F, F>(&matrix_a, &matrix_b, &matrix_acc, &matrix_acc);
        }
    }

    let c_col = global_warp_n * WMMA_N;
    let c_row = global_warp_m * WMMA_M;

    if c_row < gemm_m && c_col < gemm_n {
        let out_pos = c_col + c_row * gemm_n;
        let out = out.slice_mut(out_pos, out_pos + WMMA_OUT_TILE_SIZE);
        cmma::store(out, &matrix_acc, gemm_n, MatrixLayout::RowMajor);
    }
}

pub(crate) fn can_do_implicit_gemm<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R, E, 4>,
    weight: &JitTensor<R, E, 4>,
    out_h: usize,
    out_w: usize,
) -> bool {
    let [batch_size, in_channels, _, _] = input.shape.dims;
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims;

    let gemm_m = batch_size * out_h * out_w;
    let gemm_n = out_channels;
    let gemm_k = in_channels * kernel_h * kernel_w;

    let smem_size = (WMMA_M + WMMA_N) * WARPS_PER_BLOCK * WMMA_K * size_of::<E>() as u32;

    cmma_available::<R>(&input.device)
        && <R::Compiler as Compiler>::max_shared_memory_size() >= smem_size as usize
        && gemm_m % 16 == 0
        && gemm_n % 16 == 0
        && gemm_k % 16 == 0
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
