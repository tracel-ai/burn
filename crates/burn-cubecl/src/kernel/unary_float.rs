use crate::{element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor, CubeRuntime};
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_line_size_parallel,
};

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
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    options: &O::Options<F>,
    #[comptime] rank: Option<u32>,
    #[comptime] to_contiguous: bool,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        terminate!();
    }

    if comptime![to_contiguous] {
        let offset_input = index_offset_with_layout::<F, F>(
            input,
            output,
            offset_output,
            0,
            rank.unwrap_or_else(|| output.rank()),
            rank.is_some(),
        );

        output[offset_output] = O::Unary::<F>::execute(input[offset_input], options);
    } else {
        output[offset_output] = O::Unary::<F>::execute(input[offset_output], options);
    }
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
    let ndims = tensor.shape.num_dims();
    let line_size = tensor_line_size_parallel(
        R::line_size_elem(&E::as_elem_native_unchecked()),
        &tensor.shape.dims,
        &tensor.strides,
        ndims - 1,
    );

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems / line_size as usize, cube_dim);
    let is_contiguous = tensor.is_contiguous();

    unsafe {
        if tensor.can_mut() && tensor.is_contiguous_buffer() {
            unary_float::launch_unchecked::<E, O, R>(
                &client,
                cube_count,
                cube_dim,
                tensor.as_tensor_arg::<E>(line_size),
                TensorArg::alias(0),
                args(&()),
                None,
                false,
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
                tensor.as_tensor_arg::<E>(line_size),
                output.as_tensor_arg::<E>(line_size),
                args(&()),
                Some(ndims as u32),
                !is_contiguous,
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
        for<'a> Args: FnOnce(&'a ()) -> &'a BasicFloatUnaryKind,
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
        Erf,
        Recip,
    }

    #[derive(CubeLaunch)]
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
