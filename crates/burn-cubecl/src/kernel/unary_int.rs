use crate::{
    CubeRuntime, IntElement,
    kernel::utils::{linear_view, linear_view_alias},
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait IntUnaryOpFamily: 'static + Send + Sync {
    type Options<I: Int>: LaunchArg;
    type Unary<I: Int>: IntUnaryOp<I, Options = Self::Options<I>>;
}

#[cube]
pub(crate) trait IntUnaryOp<I: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Line<I>, options: &Self::Options) -> Line<I>;
}

#[cube(launch_unchecked)]
pub(crate) fn unary_int<I: Int, O: IntUnaryOpFamily>(
    input: &LinearView<Line<I>>,
    output: &mut LinearView<Line<I>, ReadWrite>,
    options: &O::Options<I>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<I>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_int<R, E, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options<E>, R>,
    R: CubeRuntime,
    E: IntElement + Int,
    O: IntUnaryOpFamily,
{
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            unary_int::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, &line_size),
                linear_view_alias(&tensor, &line_size, 0),
                args(&()),
            );

            tensor
        } else {
            let output = empty_device::<R, E>(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
            );

            unary_int::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view(&tensor, &line_size),
                linear_view(&output, &line_size),
                args(&()),
            );
            output
        }
    }
}

pub(crate) mod unary_basic_int {

    use super::*;

    pub(crate) fn launch<R, Args, I>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
    where
        R: CubeRuntime,
        for<'a> Args: FnOnce(&'a ()) -> &'a BasicIntUnaryKind,
        I: IntElement,
    {
        launch_unary_int::<R, I, BasicIntUnary, _>(tensor, |input| {
            BasicIntUnaryOptionsLaunch::new(args(input))
        })
    }

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum BasicIntUnaryKind {
        BitwiseNot,
    }

    #[derive(CubeLaunch, CubeType)]
    struct BasicIntUnaryOptions {
        #[cube(comptime)]
        kind: BasicIntUnaryKind,
    }
    struct BasicIntUnary;

    #[cube]
    impl<I: Int> IntUnaryOp<I> for BasicIntUnary {
        type Options = BasicIntUnaryOptions;

        fn execute(input: Line<I>, options: &Self::Options) -> Line<I> {
            match comptime![options.kind] {
                BasicIntUnaryKind::BitwiseNot => Line::bitwise_not(input),
            }
        }
    }

    impl IntUnaryOpFamily for BasicIntUnary {
        type Options<I: Int> = BasicIntUnaryOptions;
        type Unary<I: Int> = Self;
    }
}
