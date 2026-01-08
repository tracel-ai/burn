use crate::{
    CubeRuntime,
    kernel::utils::{linear_view, linear_view_alias},
    ops::{max_line_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait FloatUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<F: Float>: FloatUnaryOp<F, Options = Self::Options>;
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
    options: &O::Options,
    #[define(F)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<F>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_float<R, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options, R>,
    R: CubeRuntime,
    O: FloatUnaryOpFamily,
{
    let line_size = max_line_size(&tensor);

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let working_units = num_elems / line_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            unary_float::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
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

            unary_float::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
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

/// Use comptime enum to implement all unary operations that don't have any input argument in the
/// kernel definition.
pub(crate) mod unary_basic {
    use super::*;

    pub(crate) fn launch<R, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
    where
        R: CubeRuntime,
        for<'a> Args: FnOnce(&'a ()) -> BasicFloatUnaryKind,
    {
        launch_unary_float::<R, BasicFloatUnary, _>(tensor, |input| {
            BasicFloatUnaryOptionsLaunch::new(args(input))
        })
    }

    #[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum BasicFloatUnaryKind {
        Exp,
        Log,
        Log1p,
        Sqrt,
        Abs,
        ArcCos,
        ArcCosh,
        ArcSin,
        ArcSinh,
        ArcTan,
        ArcTanh,
        Cos,
        Cosh,
        Sin,
        Sinh,
        Tan,
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
                BasicFloatUnaryKind::Log => Line::ln(input),
                BasicFloatUnaryKind::Log1p => Line::log1p(input),
                BasicFloatUnaryKind::Sqrt => Line::sqrt(input),
                BasicFloatUnaryKind::Abs => Line::abs(input),
                BasicFloatUnaryKind::Cos => Line::cos(input),
                BasicFloatUnaryKind::Sin => Line::sin(input),
                BasicFloatUnaryKind::Tan => Line::tan(input),
                BasicFloatUnaryKind::Cosh => Line::cosh(input),
                BasicFloatUnaryKind::Sinh => Line::sinh(input),
                BasicFloatUnaryKind::Tanh => Line::tanh(input),
                BasicFloatUnaryKind::Round => Line::round(input),
                BasicFloatUnaryKind::Floor => Line::floor(input),
                BasicFloatUnaryKind::Ceil => Line::ceil(input),
                BasicFloatUnaryKind::Trunc => Line::trunc(input),
                BasicFloatUnaryKind::Erf => Line::erf(input),
                BasicFloatUnaryKind::Recip => Line::recip(input),
                BasicFloatUnaryKind::ArcCos => Line::acos(input),
                BasicFloatUnaryKind::ArcCosh => Line::acosh(input),
                BasicFloatUnaryKind::ArcSin => Line::asin(input),
                BasicFloatUnaryKind::ArcSinh => Line::asinh(input),
                BasicFloatUnaryKind::ArcTan => Line::atan(input),
                BasicFloatUnaryKind::ArcTanh => Line::atanh(input),
            }
        }
    }

    impl FloatUnaryOpFamily for BasicFloatUnary {
        type Options = BasicFloatUnaryOptions;
        type Unary<F: Float> = Self;
    }
}
