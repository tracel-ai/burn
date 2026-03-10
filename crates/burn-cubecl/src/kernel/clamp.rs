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
    impl<T: Numeric, N: Size> NumericUnaryOp<T, N> for ClampOp {
        type Options = Options;

        fn execute(input: Vector<T, N>, options: &Self::Options) -> Vector<T, N> {
            cubecl::prelude::clamp(
                input,
                Vector::new(options.min_value.get::<T>()),
                Vector::new(options.max_value.get::<T>()),
            )
        }
    }

    impl NumericUnaryOpFamily for ClampOp {
        type Options = Options;
        type Unary<T: Numeric, N: Size> = Self;
    }

    launch_unary_numeric::<R, ClampOp, _>(input, |_| OptionsLaunch::new(min_value, max_value))
}
