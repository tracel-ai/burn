use cubecl::prelude::*;

use crate::{
    kernel::{NumericUnaryOp, NumericUnaryOpFamily, launch_unary_numeric},
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct Options {
    min_value: InputScalar,
    max_value: InputScalar,
}

pub(crate) fn clamp(
    input: CubeTensor,
    min_value: InputScalar,
    max_value: InputScalar,
) -> CubeTensor {
    struct ClampOp;

    #[cube]
    impl<T: Numeric, N: Size> NumericUnaryOp<T, N> for ClampOp {
        type Options = Options;

        fn execute(input: Vector<T, N>, options: &Self::Options) -> Vector<T, N> {
            // clamp lowers to max(min(x, max), min), which returns the non-NaN operand and
            // so mapped NaN to a bound. Comparisons against NaN are false, so it survives.
            let min_value = Vector::new(options.min_value.get::<T>());
            let max_value = Vector::new(options.max_value.get::<T>());

            let clamped = select(input > max_value, max_value, input);
            select(clamped < min_value, min_value, clamped)
        }
    }

    impl NumericUnaryOpFamily for ClampOp {
        type Options = Options;
        type Unary<T: Numeric, N: Size> = Self;
    }

    launch_unary_numeric::<ClampOp, _>(input, |_| OptionsLaunch::new(min_value, max_value))
}
