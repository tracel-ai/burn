use cubecl::prelude::*;

use crate::kernel::{launch_unary, UnaryOp};
use crate::{element::JitElement, tensor::JitTensor, JitRuntime};

#[derive(CubeLaunch)]
struct Options<C: Numeric> {
    min_value: C,
    max_value: C,
}

pub(crate) fn clamp<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    min_value: E,
    max_value: E,
) -> JitTensor<R, E> {
    struct ClampOp;

    impl<C: Numeric> UnaryOp<C> for ClampOp {
        type Options = Options<C>;

        fn __expand_execute(
            context: &mut CubeContext,
            input: C::ExpandType,
            options: OptionsExpand<C>,
        ) -> C::ExpandType {
            #[cube]
            fn execute<C: Numeric>(input: C, options: &Options<C>) -> C {
                C::clamp(input, options.min_value, options.max_value)
            }

            execute::expand(context, input, options)
        }
    }

    launch_unary::<R, E, ClampOp, _>(input, |_| {
        OptionsLaunch::new(ScalarArg::new(min_value), ScalarArg::new(max_value))
    })
}
