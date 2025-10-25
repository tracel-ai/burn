use crate::{
    CubeRuntime,
    element::CubeElement,
    kernel::utils::{linear_view, linear_view_alias},
    ops::{max_line_size, numeric::empty_device},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait FloatUnaryOpFamily: 'static + Send + Sync {
    type Options<F: Float>: LaunchArg;
    type Unary<F: Float>: FloatUnaryOp<F, Options = Self::Options<F>>;
}

#[cube]
pub(crate) trait FloatUnaryOp<F: Float>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Line<F>, options: &Self::Options) -> Line<F>;
}

#[cube(launch_unchecked)]
pub(crate) fn unary_float<F: Float, O: FloatUnaryOpFamily>(
    input: &LinearView<Line<F>>,
    output: &mut LinearView<Line<F>, ReadWrite>,
    options: &O::Options<F>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<F>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_float<R, E, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options<E>, R>,
    R: CubeRuntime,
    E: CubeElement + Float,
    O: FloatUnaryOpFamily,
{
    let line_size = max_line_size(&tensor);

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            unary_float::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                linear_view(&tensor, line_size),
                linear_view_alias(&tensor, line_size, 0),
                args(&()),
            );

            tensor
        } else {
            let output = empty_device::<R, E>(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape.clone(),
            );

            unary_float::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                CubeDim::default(),
                linear_view(&tensor, line_size),
                linear_view(&output, line_size),
                args(&()),
            );
            output
        }
    }
}

/// Use comptime enum to implement all unary operations that don't have any input argument in the
/// kernel definition.
pub(crate) mod unary_basic {
    use crate::execute_with_dtype;

    use super::*;

    pub(crate) fn launch<R, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
    where
        R: CubeRuntime,
        for<'a> Args: FnOnce(&'a ()) -> BasicFloatUnaryKind,
    {
        execute_with_dtype!(
            float(tensor.dtype),
            F,
            launch_unary_float::<R, F, BasicFloatUnary, _>(tensor, |input| {
                BasicFloatUnaryOptionsLaunch::new(args(input))
            })
        )
    }

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum BasicFloatUnaryKind {
        Exp,
        Log,
        Log1p,
        Sqrt,
        Abs,
        Cos,
        Sin,
        Tanh,
        Round,
        Floor,
        Ceil,
        Trunc,
        Erf,
        Recip,
    }

    #[derive(CubeLaunch, CubeType)]
    struct BasicFloatUnaryOptions {
        #[cube(comptime)]
        kind: BasicFloatUnaryKind,
    }
    struct BasicFloatUnary;

    #[cube]
    impl<F: Float> FloatUnaryOp<F> for BasicFloatUnary {
        type Options = BasicFloatUnaryOptions;

        fn execute(input: Line<F>, options: &Self::Options) -> Line<F> {
            match comptime![options.kind] {
                BasicFloatUnaryKind::Exp => Line::exp(input),
                BasicFloatUnaryKind::Log => Line::log(input),
                BasicFloatUnaryKind::Log1p => Line::log1p(input),
                BasicFloatUnaryKind::Sqrt => Line::sqrt(input),
                BasicFloatUnaryKind::Abs => Line::abs(input),
                BasicFloatUnaryKind::Cos => Line::cos(input),
                BasicFloatUnaryKind::Sin => Line::sin(input),
                BasicFloatUnaryKind::Tanh => Line::tanh(input),
                BasicFloatUnaryKind::Round => Line::round(input),
                BasicFloatUnaryKind::Floor => Line::floor(input),
                BasicFloatUnaryKind::Ceil => Line::ceil(input),
                BasicFloatUnaryKind::Trunc => Line::trunc(input),
                BasicFloatUnaryKind::Erf => Line::erf(input),
                BasicFloatUnaryKind::Recip => Line::recip(input),
            }
        }
    }

    impl FloatUnaryOpFamily for BasicFloatUnary {
        type Options<F: Float> = BasicFloatUnaryOptions;
        type Unary<F: Float> = Self;
    }
}
