use core::hash::Hash;
use cubecl::prelude::*;

pub trait Pool2dDirectStrategyFamily: Send + Sync + 'static {
    type Indices: LaunchArg;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;
    type Pool2d<N: Numeric>: Pool2dDirectStrategy<N, Config = Self::Config, Indices = Self::Indices>;
}

#[cube]
pub(crate) trait Pool2dDirectStrategy<N: Numeric>: Send + Sync + 'static {
    type Accumulator: CubeType;
    type Config: CubeType + Clone + Send + Sync + core::fmt::Debug + Hash + core::cmp::Eq;

    type Indices: LaunchArg;

    fn initialize(#[comptime] config: &Self::Config) -> Self::Accumulator;

    fn accumulate(
        #[comptime] config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: u32,
        result: N,
    );

    fn store(
        #[comptime] config: &Self::Config,
        position: u32,
        output: &mut Tensor<N>,
        output_indices: &mut Self::Indices,
        accumulator: Self::Accumulator,
    );
}

#[derive(CubeLaunch)]
pub struct Pool2dDirectArgs {
    pub strides_0: u32,
    pub strides_1: u32,
    pub dilation_0: u32,
    pub dilation_1: u32,
    pub padding_0: u32,
    pub padding_1: u32,
}

#[cube(launch)]
pub fn pool2d_direct<E: Numeric, S: Pool2dDirectStrategyFamily>(
    input: &Tensor<E>,
    output: &mut Tensor<E>,
    indices: &mut S::Indices,
    args: &Pool2dDirectArgs,
    #[comptime] kernel_size: (u32, u32),
    #[comptime] config: &S::Config,
) {
    let (output_stride_0, output_stride_1, output_stride_2, output_stride_3) = (
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
    );
    let (output_shape_0, output_shape_1, output_shape_2, output_shape_3) = (
        output.shape(0),
        output.shape(1),
        output.shape(2),
        output.shape(3),
    );
    let (input_stride_0, input_stride_1, input_stride_2, input_stride_3) = (
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
    );
    let (input_shape_2, input_shape_3) = (input.shape(2), input.shape(3));

    let b = (ABSOLUTE_POS / output_stride_0) % output_shape_0;
    let c = (ABSOLUTE_POS / output_stride_1) % output_shape_1;
    let oh = (ABSOLUTE_POS / output_stride_2) % output_shape_2;
    let ow = (ABSOLUTE_POS / output_stride_3) % output_shape_3;

    let mut accumulator = S::Pool2d::<E>::initialize(config);

    let index_input_0 = b * input_stride_0;
    let index_input_1 = c * input_stride_1;

    let border_bottom = input_shape_2 + args.padding_0;
    let border_right = input_shape_3 + args.padding_1;

    for kh in 0..kernel_size.0 {
        let ih = oh * args.strides_0 + kh * args.dilation_0;
        let within_padding_h = ih >= args.padding_0 && ih < border_bottom;

        for kw in 0..kernel_size.1 {
            let iw = ow * args.strides_1 + kw * args.dilation_1;
            let within_padding_w = iw >= args.padding_1 && iw < border_right;

            if within_padding_h && within_padding_w {
                let ih_pad = ih - args.padding_0;
                let iw_pad = iw - args.padding_1;

                let index_input_2 = ih_pad * input_stride_2;
                let index_input_3 = iw_pad * input_stride_3;

                let index_input = index_input_0 + index_input_1 + index_input_2 + index_input_3;

                S::Pool2d::<E>::accumulate(
                    config,
                    &mut accumulator,
                    index_input_2 + iw_pad,
                    input[index_input],
                );
            }
        }
    }

    S::Pool2d::<E>::store(config, ABSOLUTE_POS, output, indices, accumulator);
}
