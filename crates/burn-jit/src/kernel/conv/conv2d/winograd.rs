use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use cubecl::prelude::*;

use crate::{ops::numeric::empty_device, tensor::JitTensor, FloatElement, JitRuntime};

const BN: u32 = 16;
const BK: u32 = 16;
const BC: u32 = 8;

pub fn conv2d_winograd<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    filter: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 4>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let [batch_size, in_channels, height, width] = input.shape.dims;
    let [out_channels, in_ch_per_group, kernel_h, kernel_w] = filter.shape.dims;

    // Only support 3x3 kernels
    assert_eq!(kernel_h, 3);
    assert_eq!(kernel_w, 3);

    let ConvOptions {
        padding: [pad_h, pad_w],
        stride: [stride_h, stride_w],
        dilation: [dilation_h, dilation_w],
        ..
    } = options.clone();

    let out_h = calculate_conv_output_size(kernel_h, stride_h, pad_h, dilation_h, height);
    let out_w = calculate_conv_output_size(kernel_w, stride_w, pad_w, dilation_w, width);
    let out_shape = Shape::new([batch_size, out_channels, out_h, out_w]);

    let tile_size = 1 + kernel_h;
    let tiles_dim = ((((width + 2) as f64) / 2.0).ceil() - 1.0).ceil() as usize;
    let elems_dim = tiles_dim * 4;

    let num_elems = out_shape.num_elements();
}

fn winograd<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    filter_mat: JitTensor<R, E, 4>,
    out_shape: Shape<4>,
    tiles_dim: u32,
    tile_size: u32,
) -> JitTensor<R, E, 4> {
    let [batch_size, in_channels, height, width] = input.shape.dims;
    let [in_channels, _, _, out_channels] = filter_mat.shape.dims;

    let shared_in_size = 16 * BC * BN;
    let shared_filter_size = 16 * BC * BK;

    let out = empty_device(input.client.clone(), input.device.clone(), out_shape);

    let cube_dim = CubeDim { x: BN, y: 8, z: 1 };
    let cube_count = CubeCount::Static(
        (batch_size as u32).div_ceil(BN),
        tiles_dim * tiles_dim,
        (out_channels as u32).div_ceil(BK),
    );

    winograd_kernel::launch::<E::FloatPrimitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_handle_ref().as_tensor_arg(4),
        filter_mat.as_handle_ref().as_tensor_arg(4),
        out.as_handle_ref().as_tensor_arg(2),
        WinogradArgsLaunch::new(ScalarArg::new(tiles_dim), ScalarArg::new(width as u32)),
        shared_in_size.into(),
        shared_filter_size.into(),
        BN.into(),
        BK.into(),
        BC.into(),
    );

    out
}

#[derive(CubeLaunch)]
struct WinogradArgs {
    tiles_dim: UInt,
    in_width: UInt,
}

#[cube(launch)]
fn winograd_kernel<F: Float>(
    input: &Tensor<F>,
    filter_mat: &Tensor<F>,
    out: &mut Tensor<F>,
    args: &WinogradArgs,
    shared_in_size: Comptime<UInt>,
    shared_filter_size: Comptime<UInt>,
    Bn: Comptime<UInt>,
    Bk: Comptime<UInt>,
    Bc: Comptime<UInt>,
) {
    let input_smem = SharedMemory::<F>::new(Comptime::get(shared_in_size));
    let filter_smem = SharedMemory::<F>::new(Comptime::get(shared_filter_size));
    let tiles_dim = args.tiles_dim;

    let bn_ = Comptime::runtime(Bn);
    let bk_ = Comptime::runtime(Bk);
    let bc_ = Comptime::runtime(Bc);

    let mut m = UInt::new(0xffff);
    if CUBE_POS_Y / tiles_dim == 0 {
        m = m & UInt::new(0xfff0);
    }
    if CUBE_POS_Y / tiles_dim == tiles_dim - 1 {
        let mut _x = UInt::new(0x00ff);
        if args.in_width % UInt::new(2) == 0 {
            _x = UInt::new(0x0fff);
        }
        m = m & _x;
    }
    if (CUBE_POS_Y + 1) % tiles_dim == 0 {
        let mut _x0 = UInt::new(0x3333);
        if args.in_width % UInt::new(2) == 0 {
            _x0 = UInt::new(0x7777);
        }
        m = m & _x0;
    }
    if CUBE_COUNT_Y % tiles_dim == 0 {
        m = m & UInt::new(0xeeee);
    }

    let mut img_tile = Array::<F>::new(16);
    let mut filter_tile = Array::<F>::new(32);

    let input_frag_mem = Array::<F>::vectorized(UInt::new(8 * 4), UInt::new(4));
    let filter_frag_mem = Array::<F>::vectorized(UInt::new(8 * 4), UInt::new(4));

    let accumulator = Array::<F>::vectorized(UInt::new(16 * 4), UInt::new(4));

    let frag_offset = Comptime::get(Bc * Bn) * 2;
    let f_frag_offset = Comptime::get(Bc * Bk) * 2;

    let swap = F::vectorized(0.0, UInt::new(4));
}

mod fx {
    use super::{BC, BK, BN};
    use crate::{ops::numeric::empty_device, tensor::JitTensor, FloatElement, JitRuntime};
    use burn_tensor::Shape;
    use cubecl::prelude::*;

    fn fx<R: JitRuntime, E: FloatElement>(filter: JitTensor<R, E, 4>) -> JitTensor<R, E, 4> {
        let [out_channels, in_ch_per_group, kernel_h, kernel_w] = filter.shape.dims;

        let cube_dim = CubeDim { x: BN, y: BC, z: 1 };
        let cube_count = CubeCount::Static(
            (out_channels as u32).div_ceil(BK),
            (in_ch_per_group as u32).div_ceil(BC),
            1,
        );
        let out = empty_device(
            filter.client.clone(),
            filter.device.clone(),
            Shape::new([in_ch_per_group, 4, 4, out_channels]),
        );

        fx_kernel::launch::<E::FloatPrimitive, R>(
            &filter.client,
            cube_count,
            cube_dim,
            filter.as_handle_ref().as_tensor_arg(1),
            out.as_handle_ref().as_tensor_arg(1),
            FxArgsLaunch::new(
                ScalarArg::new(out_channels as u32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
            ),
            BC.into(),
            BK.into(),
            BN.into(),
        );

        out
    }

    #[derive(CubeLaunch)]
    struct FxArgs {
        out_channels: UInt,
        kernel_h: UInt,
        kernel_w: UInt,
    }

    #[cube(launch)]
    fn fx_kernel<F: Float>(
        filter: &Tensor<F>,
        out: &mut Tensor<F>,
        args: &FxArgs,
        Bc: Comptime<UInt>,
        Bk: Comptime<UInt>,
        Bn: Comptime<UInt>,
    ) {
        let bc_ = Comptime::runtime(Bc);
        let bk_ = Comptime::runtime(Bk);
        let bn_ = Comptime::runtime(Bn);

        let c_glb_offset = args.out_channels * args.kernel_h * args.kernel_w;
        let mut c_kernel = CUBE_POS_Y * bc_ * c_glb_offset
            + CUBE_POS_X * bk_
            + UNIT_POS_Y * c_glb_offset
            + UNIT_POS_X;
        let c_glb_offset_s = args.out_channels * 4 * 4;
        let mut c_kernel_s = CUBE_POS_Y * bc_ * c_glb_offset_s
            + CUBE_POS_X * bk_
            + UNIT_POS_Y * c_glb_offset_s
            + UNIT_POS_X;

        let mut gw = Array::new(21);
        let buf_offset = UInt::new(9);

        for _ in range_stepped(0, bk_, bn_, Comptime::new(false)) {
            for i in range(0, 9, Comptime::new(false)) {
                gw[i] = filter[c_kernel + i * args.out_channels];
            }

            for i0 in range(0, 4, Comptime::new(false)) {
                let aux = i0 * 3;
                for j in range(0, 3, Comptime::new(false)) {
                    gw[buf_offset + j + aux] = row(i0, &gw, j);
                }
            }

            for i1 in range(0, 4, Comptime::new(false)) {
                let aux = i1 * 3;
                let aux2 = i1 << UInt::new(2);
                for j in range(0, 4, Comptime::new(false)) {
                    out[c_kernel_s + aux2 * args.out_channels + j * args.out_channels] =
                        col(j, &gw, aux);
                }
            }

            c_kernel += CUBE_DIM_X;
            c_kernel_s += CUBE_DIM_X;
        }
    }

    #[cube]
    fn row<F: Float>(i: UInt, data: &Array<F>, j: UInt) -> F {
        let mut result = F::new(0.0);
        if i == 0 {
            result = data[j];
        } else if i == 1 {
            result = (data[j] + data[j + 6] + data[j + 3]) * 0.5;
        } else if i == 2 {
            result = (data[j] + data[j + 6] - data[j + 3]) * 0.5;
        } else if i == 3 {
            result = data[j + 6];
        }
        result
    }

    #[cube]
    fn col<F: Float>(i: UInt, data: &Array<F>, j: UInt) -> F {
        let mut result = F::new(0.0);
        if i == 0 {
            result = data[j + 9];
        } else if i == 1 {
            result = (data[j + 9] + data[j + 2 + 9] + data[j + 1 + 9]) * 0.5;
        } else if i == 2 {
            result = (data[j + 9] + data[j + 2 + 9] - data[j + 1 + 9]) * 0.5;
        } else if i == 3 {
            result = data[j + 2 + 9];
        }
        result
    }
}
