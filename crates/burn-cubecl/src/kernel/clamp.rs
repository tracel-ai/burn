use cubecl::{prelude::*, std::scalar::InputScalar};

use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::{NumericUnaryOp, NumericUnaryOpFamily, launch_unary_numeric},
    ops::numeric::input_scalar,
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct Options {
    min_value: InputScalar,
    max_value: InputScalar,
}

pub(crate) fn clamp<R: CubeRuntime, E: CubeElement>(
    input: CubeTensor<R>,
    min_value: E,
    max_value: E,
) -> CubeTensor<R> {
    struct ClampOp;

    #[cube]
    impl<N: Numeric> NumericUnaryOp<N> for ClampOp {
        type Options = Options;

        fn execute(input: Line<N>, options: &Self::Options) -> Line<N> {
            let line_size = input.size();
            Line::clamp(
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

    let dtype = input.dtype;
    launch_unary_numeric::<R, ClampOp, _>(input, |_| {
        OptionsLaunch::new(
            input_scalar(min_value, dtype),
            input_scalar(max_value, dtype),
        )
    })
}
