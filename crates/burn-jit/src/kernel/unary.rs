use burn_cube::{calculate_cube_count_elemwise, prelude::*, unexpanded, SUBCUBE_DIM_APPROX};
use burn_cube::{frontend::TensorHandle, CubeCountSettings, Execution};

use crate::{element::JitElement, tensor::JitTensor, JitRuntime};

use super::{index_offset_with_layout, index_offset_with_layout_expand, Kernel};

pub(crate) trait UnaryOp<C: CubePrimitive>: 'static + Send + Sync {
    type Options: LaunchArg;

    /// Execute a unary operation.
    fn execute(_input: C, _options: &Self::Options) -> C {
        unexpanded!();
    }
    fn execute_expand(
        context: &mut CubeContext,
        input: C::ExpandType,
        options: <Self::Options as CubeType>::ExpandType,
    ) -> C::ExpandType;
}

#[cube(launch)]
pub(crate) fn unary_kernel<C: CubePrimitive, O: UnaryOp<C>>(
    input: &Tensor<C>,
    output: &mut Tensor<C>,
    options: &O::Options,
    rank: Comptime<Option<UInt>>,
    to_contiguous: Comptime<bool>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    if Comptime::get(to_contiguous) {
        let offset_input = index_offset_with_layout::<C>(
            input,
            output,
            offset_output,
            UInt::new(0),
            Comptime::unwrap_or_else(rank, || output.rank()),
            Comptime::is_some(rank),
        );

        output[offset_output] = O::execute(input[offset_input], options);
    } else {
        output[ABSOLUTE_POS] = O::execute(input[ABSOLUTE_POS], options);
    }
}

pub(crate) fn launch_unary<
    const D: usize,
    R: JitRuntime,
    E: JitElement,
    O: UnaryOp<E::Primitive>,
    F,
>(
    tensor: JitTensor<R, E, D>,
    options: F,
) -> JitTensor<R, E, D>
where
    // Magic fix for lifetime, the closure is supposed to capture everything required to create the
    // argument.
    for<'a> F: FnOnce(&'a ()) -> RuntimeArg<'a, O::Options, R>,
{
    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor = if tensor.strides[D - 1] == 1 {
        let last_dim = tensor.shape.dims[D - 1];
        if last_dim % 4 == 0 {
            4
        } else if last_dim % 2 == 0 {
            2
        } else {
            1
        }
    } else {
        1
    };
    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );
    let is_contiguous = tensor.is_contiguous();

    if tensor.can_mut() && is_contiguous {
        unary_kernel_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            TensorArg::alias(0),
            options(&()),
            None,
            false,
        );

        tensor
    } else {
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let output = JitTensor::new(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );

        unary_kernel_launch::<E::Primitive, O, R>(
            client,
            cube_count,
            CubeDim::default(),
            TensorArg::vectorized(
                vectorization_factor,
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            ),
            TensorArg::vectorized(
                vectorization_factor,
                &output.handle,
                &output.strides,
                &output.shape.dims,
            ),
            options(&()),
            Some(UInt::new(D as u32)),
            !is_contiguous,
        );
        output
    }
}

macro_rules! unary_op {
    ($name:ident, $elem:ident, $expand:expr) => {
        struct $name;

        impl<C: $elem> UnaryOp<C> for $name {
            type Options = ();

            fn execute_expand(
                context: &mut CubeContext,
                input: C::ExpandType,
                _options: <Self::Options as CubeType>::ExpandType,
            ) -> C::ExpandType {
                $expand(context, input)
            }
        }
    };
    (scalar $name:ident, $elem:ident, $expand:expr) => {
        struct $name;

        impl<C: $elem> UnaryOp<C> for $name {
            type Options = C;

            fn execute_expand(
                context: &mut CubeContext,
                input: C::ExpandType,
                scalar: C::ExpandType,
            ) -> C::ExpandType {
                $expand(context, input, scalar)
            }
        }
    };
    (float($tensor:expr) => $exp:expr) => {{
        unary_op!(Op, Float, $exp);
        launch_unary::<D, R, F, Op, _>($tensor, |_| ())
    }};
    (float($tensor:expr, $scalar:expr) => $exp:expr) => {{
        unary_op!(scalar Op, Float, $exp);
        launch_unary::<D, R, F, Op, _>($tensor, |_| ScalarArg::new($scalar))
    }};
}

pub(crate) use unary_op;

/// Creates a unary kernel.
#[macro_export]
macro_rules! unary {
    (
        operation: $ops:expr,
        runtime: $runtime:ty,
        input: $input:expr,
        elem: $elem:ty
    ) => {{
        unary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler);

        $crate::kernel::unary::<
            Ops<<$runtime as Runtime>::Compiler, $elem>,
            OpsInplace<<$runtime as Runtime>::Compiler, $elem>,
            $runtime,
            $elem,
            D
        >($input, None, true, Ops::new(), OpsInplace::new())
    }};
    (
        operation: $ops:expr,
        runtime: $runtime:ty,
        input: $input:expr; $scalar:expr,
        elem: $elem:ty
    ) => {{
        unary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler, scalar 1);

        $crate::kernel::unary::<
            Ops<<$runtime as Runtime>::Compiler, $elem>,
            OpsInplace<<$runtime as Runtime>::Compiler, $elem>,
            $runtime,
            $elem,
            D
        >($input, Some(&[$scalar]), true, Ops::new(), OpsInplace::new())
    }};

    (
        operation: $ops:expr,
        compiler: $compiler:ty
    ) => {
        #[derive(new)]
        pub struct Ops<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }
        #[derive(new)]
        pub struct OpsInplace<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }

        #[allow(clippy::redundant_closure_call)]
        fn compile<E>(
            settings: burn_cube::KernelSettings,
        ) -> burn_cube::ir::KernelDefinition
        where
            E: $crate::element::JitElement
        {

            let mut scope = burn_cube::ir::Scope::root();
            let op = $ops(&mut scope, E::cube_elem(), burn_cube::ir::Variable::AbsolutePos);
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let input = burn_cube::InputInfo::Array {
                item: burn_cube::ir::Item::new(E::cube_elem()),
                visibility: burn_cube::ir::Visibility::Read,
            };
            let out = burn_cube::OutputInfo::ArrayWrite {
                item: burn_cube::ir::Item::new(E::cube_elem()),
                local,
                position: burn_cube::ir::Variable::AbsolutePos,
            };
            let info = burn_cube::KernelExpansion {
                inputs: vec![input],
                outputs: vec![out],
                scope,
            };
            burn_cube::KernelIntegrator::new(info).integrate(settings)
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::Kernel for Ops<C, E>
        where
            C: burn_cube::Compiler,
            E: $crate::element::JitElement,
        {
            fn define(&self) -> burn_cube::ir::KernelDefinition {
                let settings = burn_cube::KernelSettings::default();
                compile::<E>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::Kernel for OpsInplace<C, E>
        where
            C: burn_cube::Compiler,
            E: $crate::element::JitElement,
        {
            fn define(&self) -> burn_cube::ir::KernelDefinition {
                let mapping = burn_cube::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = burn_cube::KernelSettings::default()
                    .inplace(vec![mapping]);
                compile::<E>(settings)
            }
        }
    };
    (
        operation: $ops:expr,
        compiler: $compiler:ty,
        scalar $num:expr
    ) => {
        #[derive(new)]
        pub struct Ops<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }
        #[derive(new)]
        pub struct OpsInplace<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }

        #[allow(clippy::redundant_closure_call)]
        fn compile<E>(
            settings: burn_cube::KernelSettings,
        ) -> burn_cube::ir::KernelDefinition
        where
            E: $crate::element::JitElement
        {

            let mut scope = burn_cube::ir::Scope::root();
            let op = $ops(&mut scope, E::cube_elem(), burn_cube::ir::Variable::AbsolutePos);
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let input = burn_cube::InputInfo::Array {
                item: burn_cube::ir::Item::new(E::cube_elem()),
                visibility: burn_cube::ir::Visibility::Read,
            };
            let scalars = burn_cube::InputInfo::Scalar {
                elem: E::cube_elem(),
                size: $num,
            };
            let out = burn_cube::OutputInfo::ArrayWrite {
                item: burn_cube::ir::Item::new(E::cube_elem()),
                local,
                position: burn_cube::ir::Variable::AbsolutePos,
            };
            let info = burn_cube::KernelExpansion {
                inputs: vec![input, scalars],
                outputs: vec![out],
                scope,
            };
            burn_cube::KernelIntegrator::new(info).integrate(settings)
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::Kernel for Ops<C, E>
        where
            C: burn_cube::Compiler,
            E: $crate::element::JitElement,
        {
            fn define(&self) -> burn_cube::ir::KernelDefinition {
                let settings = burn_cube::KernelSettings::default();
                compile::<E>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::Kernel for OpsInplace<C, E>
        where
            C: burn_cube::Compiler,
            E: $crate::element::JitElement,
        {
            fn define(&self) -> burn_cube::ir::KernelDefinition {
                let mapping = burn_cube::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = burn_cube::KernelSettings::default()
                    .inplace(vec![mapping]);
                compile::<E>(settings)
            }
        }
    };
}

/// Launch an unary operation.
pub fn unary<K, Kinplace, R: JitRuntime, E, const D: usize>(
    tensor: JitTensor<R, E, D>,
    scalars: Option<&[E]>,
    inplace_enabled: bool,
    kernel: K,
    kernel_inplace: Kinplace,
) -> JitTensor<R, E, D>
where
    K: Kernel,
    Kinplace: Kernel,
    E: JitElement,
{
    if inplace_enabled && tensor.can_mut() {
        let input_handles = &[TensorHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )];

        let launch = CubeCountSettings::Input { pos: 0 };

        match scalars {
            Some(scalars) => {
                Execution::start(kernel_inplace, tensor.client.clone())
                    .inputs(input_handles)
                    .with_scalars(scalars)
                    .execute(launch);
            }
            None => {
                Execution::start(kernel_inplace, tensor.client.clone())
                    .inputs(input_handles)
                    .execute(launch);
            }
        }

        tensor
    } else {
        let num_elems = tensor.shape.num_elements();
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let output = JitTensor::new(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );

        let input_handles = &[TensorHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )];

        let output_handles = &[TensorHandle::<R>::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )];

        let launch = CubeCountSettings::Output { pos: 0 };

        match scalars {
            Some(scalars) => {
                Execution::start(kernel, tensor.client)
                    .inputs(input_handles)
                    .outputs(output_handles)
                    .with_scalars(scalars)
                    .execute(launch);
            }
            None => {
                Execution::start(kernel, tensor.client)
                    .inputs(input_handles)
                    .outputs(output_handles)
                    .execute(launch);
            }
        }

        output
    }
}
