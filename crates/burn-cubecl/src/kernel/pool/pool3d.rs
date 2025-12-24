use core::hash::Hash;
use cubecl::prelude::*;

pub trait Pool3dDirectStrategyFamily: Send + Sync + 'static {
    type Indices: LaunchArg;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;
    type Pool3d<N: Numeric>: Pool3dDirectStrategy<N, Config = Self::Config, Indices = Self::Indices>;
}

#[cube]
pub(crate) trait Pool3dDirectStrategy<N: Numeric>: Send + Sync + 'static {
    type Accumulator: CubeType;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    type Indices: LaunchArg;

    fn initialize(
        #[comptime] config: &Self::Config,
        #[comptime] line_size: u32,
    ) -> Self::Accumulator;

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: u32,
        result: Line<N>,
    );

    /// Count a position within the kernel window (for avg_pool count_include_pad).
    /// Called for each position in the kernel window with the current id/ih/iw coordinates.
    /// Only avg_pool uses this; max_pool implements as no-op.
    fn count_position(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        id: u32,
        ih: u32,
        iw: u32,
    );

    fn store(
        #[comptime] config: &Self::Config,
        position: u32,
        output: &mut Tensor<Line<N>>,
        output_indices: &mut Self::Indices,
        accumulator: Self::Accumulator,
    );
}

#[derive(CubeLaunch, CubeType)]
pub struct Pool3dDirectArgs {
    pub strides_0: u32,
    pub strides_1: u32,
    pub strides_2: u32,
    pub dilation_0: u32,
    pub dilation_1: u32,
    pub dilation_2: u32,
    pub padding_0: u32,
    pub padding_1: u32,
    pub padding_2: u32,
}

#[cube(launch)]
pub fn pool3d_direct<E: Numeric, S: Pool3dDirectStrategyFamily>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    indices: &mut S::Indices,
    args: &Pool3dDirectArgs,
    #[comptime] kernel_size: (u32, u32, u32),
    #[comptime] config: &S::Config,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    // Output shape is [batch, out_d, out_h, out_w, channels] in NDHWC format
    let (out_d, out_h, out_w, channels) = (
        output.shape(1),
        output.shape(2),
        output.shape(3),
        output.shape(4),
    );
    let channel_lines = channels / input.line_size();
    let (in_stride_b, in_stride_d, in_stride_h, in_stride_w, in_stride_c) = (
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        input.stride(4),
    );
    let (in_d, in_h, in_w) = (input.shape(1), input.shape(2), input.shape(3));

    // Decode position: c, ow, oh, od, b
    let c = (ABSOLUTE_POS % channel_lines) * input.line_size();
    let pos = ABSOLUTE_POS / channel_lines;
    let ow = pos % out_w;
    let pos = pos / out_w;
    let oh = pos % out_h;
    let pos = pos / out_h;
    let od = pos % out_d;
    let b = pos / out_d;

    let mut accumulator = S::Pool3d::<E>::initialize(config, input.line_size());

    let in_b_off = b * in_stride_b;
    let in_c_off = c * in_stride_c;

    let border_back = in_d + args.padding_0;
    let border_bottom = in_h + args.padding_1;
    let border_right = in_w + args.padding_2;

    for kd in 0..kernel_size.0 {
        let id = od * args.strides_0 + kd * args.dilation_0;
        let within_padding_d = id >= args.padding_0 && id < border_back;

        for kh in 0..kernel_size.1 {
            let ih = oh * args.strides_1 + kh * args.dilation_1;
            let within_padding_h = ih >= args.padding_1 && ih < border_bottom;

            for kw in 0..kernel_size.2 {
                let iw = ow * args.strides_2 + kw * args.dilation_2;
                let within_padding_w = iw >= args.padding_2 && iw < border_right;

                // Let strategy handle position counting (only used by avg_pool)
                S::Pool3d::<E>::count_position(config, &mut accumulator, id, ih, iw);

                // Only accumulate values from valid input positions
                if within_padding_d && within_padding_h && within_padding_w {
                    let id_pad = id - args.padding_0;
                    let ih_pad = ih - args.padding_1;
                    let iw_pad = iw - args.padding_2;

                    let in_d_off = id_pad * in_stride_d;
                    let in_h_off = ih_pad * in_stride_h;
                    let in_w_off = iw_pad * in_stride_w;

                    let index_input = in_b_off + in_c_off + in_d_off + in_h_off + in_w_off;

                    S::Pool3d::<E>::accumulate(
                        config,
                        &mut accumulator,
                        id_pad * in_h * in_w + ih_pad * in_w + iw_pad,
                        input[index_input / input.line_size()],
                    );
                }
            }
        }
    }

    S::Pool3d::<E>::store(config, ABSOLUTE_POS, output, indices, accumulator);
}
