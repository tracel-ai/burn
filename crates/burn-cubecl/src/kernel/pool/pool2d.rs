use core::hash::Hash;
use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::{
            View,
            launch::ViewArg,
            layout::fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
        },
    },
};

use crate::{CubeRuntime, kernel::utils::decompose_linear, tensor::CubeTensor};

pub trait Pool2dDirectStrategyFamily: Send + Sync + 'static {
    type Indices: LaunchArg;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;
    type Pool2d<N: Numeric>: Pool2dDirectStrategy<N, Config = Self::Config, Indices = Self::Indices>;
}

pub(super) type Position = (usize, usize, usize, usize);

#[cube]
pub(crate) trait Pool2dDirectStrategy<N: Numeric>: Send + Sync + 'static {
    type Accumulator: CubeType;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    type Indices: LaunchArg;

    fn initialize(
        #[comptime] config: &Self::Config,
        #[comptime] line_size: LineSize,
    ) -> Self::Accumulator;

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: usize,
        result: Line<N>,
    );

    /// Count a position within the kernel window (for avg_pool count_include_pad).
    /// Called for each position in the kernel window with the current ih/iw coordinates.
    /// Only avg_pool uses this; max_pool implements as no-op.
    fn count_position(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        ih: u32,
        iw: u32,
    );

    fn store(
        #[comptime] config: &Self::Config,
        position: Position,
        output: &mut View<Line<N>, Position, ReadWrite>,
        output_indices: &mut Self::Indices,
        accumulator: Self::Accumulator,
    );
}

#[derive(CubeLaunch, CubeType)]
pub struct Pool2dDirectArgs {
    pub strides_0: u32,
    pub strides_1: u32,
    pub dilation_0: u32,
    pub dilation_1: u32,
    pub padding_0: u32,
    pub padding_1: u32,
}

#[cube(launch, address_type = "dynamic")]
pub fn pool2d_direct<E: Numeric, S: Pool2dDirectStrategyFamily>(
    input: &Tensor<Line<E>>,
    output: &mut View<Line<E>, Position, ReadWrite>,
    indices: &mut S::Indices,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    args: &Pool2dDirectArgs,
    #[comptime] kernel_size: (u32, u32),
    #[comptime] config: &S::Config,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS * output.line_size(), &out_shape);
    let [b, oh, ow, c] = *pos else { unreachable!() };

    let (in_stride_h, in_stride_w) = (input.stride(1), input.stride(2));
    let (in_h, in_w) = (input.shape(1) as u32, input.shape(2) as u32);

    let mut accumulator = S::Pool2d::<E>::initialize(config, input.line_size());

    let in_b_off = b * input.stride(0);
    let in_c_off = c * input.stride(3);

    let border_bottom = in_h + args.padding_0;
    let border_right = in_w + args.padding_1;

    for kh in 0..kernel_size.0 {
        let ih = oh as u32 * args.strides_0 + kh * args.dilation_0;
        let within_padding_h = ih >= args.padding_0 && ih < border_bottom;

        for kw in 0..kernel_size.1 {
            let iw = ow as u32 * args.strides_1 + kw * args.dilation_1;
            let within_padding_w = iw >= args.padding_1 && iw < border_right;

            // Let strategy handle position counting (only used by avg_pool)
            S::Pool2d::<E>::count_position(config, &mut accumulator, ih, iw);

            // Only accumulate values from valid input positions
            if within_padding_h && within_padding_w {
                let ih_pad = ih - args.padding_0;
                let iw_pad = iw - args.padding_1;

                let in_h_off = ih_pad as usize * in_stride_h;
                let in_w_off = iw_pad as usize * in_stride_w;

                let index_input = in_b_off + in_c_off + in_h_off + in_w_off;

                S::Pool2d::<E>::accumulate(
                    config,
                    &mut accumulator,
                    ih_pad as usize * in_w as usize + iw_pad as usize,
                    input[index_input / input.line_size()],
                );
            }
        }
    }

    S::Pool2d::<E>::store(config, (b, oh, ow, c), output, indices, accumulator);
}

pub(super) fn view4d<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    line_size: LineSize,
) -> ViewArg<'_, Position, R> {
    let shape = tensor.meta.shape();
    let shape = (
        ScalarArg::new(shape[0]),
        ScalarArg::new(shape[1]),
        ScalarArg::new(shape[2]),
        ScalarArg::new(shape[3]),
    );
    let handle = tensor.as_handle_ref();
    let len = handle.shape.iter().product::<usize>();
    let layout =
        FixedDimLayoutLaunch::<Position, R>::from_shape_handle_unchecked(&handle, shape, line_size);
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(handle.handle, len, line_size, handle.elem_size)
    };
    ViewArg::new::<FixedDimLayout<Position>>(buffer, layout)
}
