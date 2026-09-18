use cubecl::prelude::*;

use crate::{
    kernel::{
        FloatUnaryOp, FloatUnaryOpFamily, NumericUnaryOp, NumericUnaryOpFamily, launch_unary_float,
        launch_unary_numeric,
    },
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct Options {
    min_value: InputScalar,
    max_value: InputScalar,
}

/// Clamp a float tensor, keeping NaN inputs.
pub(crate) fn clamp_float(
    input: CubeTensor,
    min_value: InputScalar,
    max_value: InputScalar,
) -> CubeTensor {
    struct ClampOp;

    #[cube]
    impl<F: Float, N: Size> FloatUnaryOp<F, N> for ClampOp {
        type Options = Options;

        fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N> {
            // clamp lowers to max(min(x, max), min), which returns the non-NaN operand and so
            // mapped NaN to a bound. Comparisons against NaN are false, so it survives here, in
            // the same order as the clamp_min/clamp_max trait default.
            let min_value = Vector::new(options.min_value.get::<F>());
            let max_value = Vector::new(options.max_value.get::<F>());

            let clamped = select(input > max_value, max_value, input);
            select(clamped < min_value, min_value, clamped)
        }
    }

    impl FloatUnaryOpFamily for ClampOp {
        type Options = Options;
        type Unary<F: Float, N: Size> = Self;
    }

    launch_unary_float::<ClampOp, _>(input, |_| OptionsLaunch::new(min_value, max_value))
}

/// Clamp an int tensor.
pub(crate) fn clamp_int(
    input: CubeTensor,
    min_value: InputScalar,
    max_value: InputScalar,
) -> CubeTensor {
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

    launch_unary_numeric::<ClampOp, _>(input, |_| OptionsLaunch::new(min_value, max_value))
}
