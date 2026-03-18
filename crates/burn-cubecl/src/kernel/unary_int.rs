use crate::{
    CubeRuntime,
    kernel::utils::address_type,
    ops::{max_vector_size, numeric::empty_device_dtype},
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

pub(crate) trait IntUnaryOpFamily: 'static + Send + Sync {
    type Options: LaunchArg;
    type Unary<I: Int, N: Size>: IntUnaryOp<I, N, Options = Self::Options>;
}

#[cube]
pub(crate) trait IntUnaryOp<I: Scalar, N: Size>: 'static + Send + Sync {
    type Options: LaunchArg;

    fn execute(input: Vector<I, N>, options: &Self::Options) -> Vector<I, N>;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub(crate) fn unary_int<I: Int, N: Size, O: IntUnaryOpFamily>(
    input: &LinearView<Vector<I, N>>,
    output: &mut LinearView<Vector<I, N>, ReadWrite>,
    options: &O::Options,
    #[define(I)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = O::Unary::<I, N>::execute(input[ABSOLUTE_POS], options);
}

pub(crate) fn launch_unary_int<R, O, Args>(tensor: CubeTensor<R>, args: Args) -> CubeTensor<R>
where
    for<'a> Args: FnOnce(&'a ()) -> RuntimeArg<O::Options, R>,
    R: CubeRuntime,
    O: IntUnaryOpFamily,
{
    let vector_size = max_vector_size(&tensor);
    let client = tensor.client.clone();
    let num_elems = tensor.meta.num_elements();

    let working_units = num_elems / vector_size as usize;
    let cube_dim = CubeDim::new(&tensor.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, working_units, cube_dim);
    let dtype = tensor.dtype;

    unsafe {
        if tensor.can_mut() && tensor.is_nonoverlapping() {
            unary_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor),
                vector_size,
                tensor.clone().into_linear_view(),
                tensor.as_linear_view_alias(0),
                args(&()),
                dtype.into(),
            );

            tensor
        } else {
            let output = empty_device_dtype(
                tensor.client.clone(),
                tensor.device.clone(),
                tensor.shape(),
                tensor.dtype,
            );

            unary_int::launch_unchecked::<O, R>(
                &client,
                cube_count,
                cube_dim,
                address_type!(tensor, output),
                vector_size,
                tensor.into_linear_view(),
                output.clone().into_linear_view(),
                args(&()),
                dtype.into(),
            );

            output
        }
    }
}

pub(crate) mod unary_basic_int {

    use cubecl::num_traits::{One, Zero};

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
    impl<I: Int, N: Size> IntUnaryOp<I, N> for BasicIntUnary {
        type Options = BasicIntUnaryOptions;

        fn execute(input: Vector<I, N>, options: &Self::Options) -> Vector<I, N> {
            match comptime![options.kind] {
                BasicIntUnaryKind::BitwiseNot => !input,
                BasicIntUnaryKind::Sign => {
                    let zero = Vector::zero();
                    let one = Vector::one();
                    let minus_one = Vector::new(I::new(-1));

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
        type Unary<I: Int, N: Size> = Self;
    }
}
