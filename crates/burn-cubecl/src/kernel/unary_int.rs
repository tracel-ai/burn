use crate::{
    CubeRuntime,
    kernel::utils::{address_type, linear_view, linear_view_alias},
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait IntUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<I: Int>: IntUnaryOp<I, Options = Self::Options>;
}

#[cube]
pub(crate) trait IntUnaryOp<I: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Line<I>, options: &Self::Options) -> Line<I>;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn unary_int<I: Int, O: IntUnaryOpFamily>(
    input: &LinearView<Line<I>>,
    output: &mut LinearView<Line<I>, ReadWrite>,
    options: &O::Options,
    #[define(I)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<I>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_int<R, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options, R>,
    R: CubeRuntime,
    O: IntUnaryOpFamily,
{
    let line_size = max_line_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            unary_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                linear_view(&tensor, line_size),
                linear_view_alias(&tensor, line_size, 0),
                args(&()),
                tensor.dtype.into(),
            )
            .expect("Kernel to never fail");

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
                tensor.dtype,
            );

            unary_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor, output),
                linear_view(&tensor, line_size),
                linear_view(&output, line_size),
                args(&()),
                tensor.dtype.into(),
            )
            .expect("Kernel to never fail");
            output
        }
    }
}

pub(crate) mod unary_basic_int {

    use super::*;

    pub(crate) fn launch<R, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
    where
        R: CubeRuntime,
        for<'a> Args: FnOnce(&'a ()) -> BasicIntUnaryKind,
    {
        launch_unary_int::<R, BasicIntUnary, _>(tensor, |input| {
            BasicIntUnaryOptionsLaunch::new(args(input))
        })
    }

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum BasicIntUnaryKind {
        BitwiseNot,
        Sign,
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
                BasicIntUnaryKind::BitwiseNot => !input,
                BasicIntUnaryKind::Sign => {
                    let zero = Line::new(I::new(0));
                    let one = Line::new(I::new(1));
                    let minus_one = Line::new(I::new(-1));

                    let is_positive = input.greater_than(zero);
                    let is_negative = input.less_than(zero);
                    let sign = select_many(is_negative, minus_one, zero);

                    select_many(is_positive, one, sign)
                }
            }
        }
    }

    impl IntUnaryOpFamily for BasicIntUnary {
        type Options = BasicIntUnaryOptions;
        type Unary<I: Int> = Self;
    }
}
