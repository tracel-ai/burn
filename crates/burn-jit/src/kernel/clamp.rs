use burn_cube::prelude::*;

use crate::kernel::{launch_unary, UnaryOp};
use crate::{element::JitElement, tensor::JitTensor, JitRuntime};

pub(crate) fn clamp<R: JitRuntime, E: JitElement, const D: usize>(
    input: JitTensor<R, E, D>,
    min_value: E,
    max_value: E,
) -> JitTensor<R, E, D> {
    #[derive(CubeLaunch)]
    struct Options<C: Numeric> {
        min_value: C,
        max_value: C,
    }

    struct ClampOp;

    impl<C: Numeric> UnaryOp<C> for ClampOp {
        type Options = Options<C>;

        fn execute_expand(
            context: &mut CubeContext,
            input: C::ExpandType,
            options: OptionsExpand<C>,
        ) -> C::ExpandType {
            #[cube]
            fn execute<C: Numeric>(input: C, options: &Options<C>) -> C {
                C::clamp(input, options.max_value, options.min_value)
            }

            execute_expand(context, input, options)
        }
    }

    launch_unary::<D, R, E, ClampOp, _>(input, |_| {
        OptionsLaunch::new(ScalarArg::new(min_value), ScalarArg::new(max_value))
    })
}
