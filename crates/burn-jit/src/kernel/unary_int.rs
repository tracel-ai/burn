use crate::{ops::numeric::empty_device, tensor::JitTensor, IntElement, JitRuntime};
use cubecl::{
    calculate_cube_count_elemwise, linalg::tensor::index_offset_with_layout, prelude::*,
    tensor_line_size_parallel,
};

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
    input: &Tensor<Line<I>>,
    output: &mut Tensor<Line<I>>,
    options: &O::Options<I>,
    #[comptime] rank: Option<u32>,
    #[comptime] to_contiguous: bool,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        terminate!();
    }

    if comptime![to_contiguous] {
        let offset_input = index_offset_with_layout::<I, I>(
            input,
            output,
            offset_output,
            0,
            rank.unwrap_or_else(|| output.rank()),
            rank.is_some(),
        );

        output[offset_output] = O::Unary::<I>::execute(input[offset_input], options);
    } else {
        output[offset_output] = O::Unary::<I>::execute(input[offset_output], options);
    }
}

pub(crate) fn launch_unary_int<R, E, O, Args>(tensor: JitTensor<R>, args: Args) -> JitTensor<R>
where
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options<E>, R>,
    R: JitRuntime,
    E: IntElement + Int,
    O: IntUnaryOpFamily,
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
            unary_int::launch_unchecked::<E, O, R>(
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

            unary_int::launch_unchecked::<E, O, R>(
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

pub(crate) mod unary_basic_int {

    use super::*;

    pub(crate) fn launch<R, Args, I>(tensor: JitTensor<R>, args: Args) -> JitTensor<R>
    where
        R: JitRuntime,
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

    #[derive(CubeLaunch)]
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
