use cubecl::prelude::*;

use crate::{
    CubeRuntime,
    kernel::{NumericUnaryOp, NumericUnaryOpFamily, launch_unary_numeric},
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct Options {
    min_value: InputScalar,
    max_value: InputScalar,
}

pub(crate) fn clamp<R: CubeRuntime>(
    input: CubeTensor<R>,
    min_value: InputScalar,
    max_value: InputScalar,
) -> CubeTensor<R> {
    struct ClampOp;

    #[cube]
    impl<N: Numeric> NumericUnaryOp<N> for ClampOp {
        type Options = Options;

        fn execute(input: Line<N>, options: &Self::Options) -> Line<N> {
            let line_size = input.size();
            cubecl::prelude::clamp(
                input,
                Line::empty(line_size).fill(options.min_value.get::<N>()),
                Line::empty(line_size).fill(options.max_value.get::<N>()),
            )
        }
    }

    impl NumericUnaryOpFamily for ClampOp {
        type Options = Options;
        type Unary<N: Numeric> = Self;
    }

    launch_unary_numeric::<R, ClampOp, _>(input, |_| OptionsLaunch::new(min_value, max_value))
}
