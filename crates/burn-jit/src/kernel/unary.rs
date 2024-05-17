use crate::{
    codegen::{EagerHandle, Execution, WorkgroupLaunch},
    element::JitElement,
    tensor::JitTensor,
    JitRuntime,
};

use super::GpuComputeShaderPhase;

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
            settings: $crate::codegen::CompilationSettings,
        ) -> burn_cube::dialect::ComputeShader
        where
            E: $crate::element::JitElement
        {

            let mut scope = burn_cube::dialect::Scope::root();
            let op = $ops(&mut scope, E::cube_elem(), burn_cube::dialect::Variable::Id);
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let input = $crate::codegen::InputInfo::Array {
                item: burn_cube::dialect::Item::Scalar(E::cube_elem()),
                visibility: burn_cube::dialect::Visibility::Read,
            };
            let out = $crate::codegen::OutputInfo::ArrayWrite {
                item: burn_cube::dialect::Item::Scalar(E::cube_elem()),
                local,
                position: burn_cube::dialect::Variable::Id,
            };
            let info = $crate::codegen::CompilationInfo {
                inputs: vec![input],
                outputs: vec![out],
                scope,
            };
            $crate::codegen::Compilation::new(info).compile(settings)
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::GpuComputeShaderPhase for Ops<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn compile(&self) -> burn_cube::dialect::ComputeShader {
                let settings = $crate::codegen::CompilationSettings::default();
                compile::<E>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::GpuComputeShaderPhase for OpsInplace<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn compile(&self) -> burn_cube::dialect::ComputeShader {
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = $crate::codegen::CompilationSettings::default()
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
            settings: $crate::codegen::CompilationSettings,
        ) -> burn_cube::dialect::ComputeShader
        where
            E: $crate::element::JitElement
        {

            let mut scope = burn_cube::dialect::Scope::root();
            let op = $ops(&mut scope, E::cube_elem(), burn_cube::dialect::Variable::Id);
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let input = $crate::codegen::InputInfo::Array {
                item: burn_cube::dialect::Item::Scalar(E::cube_elem()),
                visibility: burn_cube::dialect::Visibility::Read,
            };
            let scalars = $crate::codegen::InputInfo::Scalar {
                elem: E::cube_elem(),
                size: $num,
            };
            let out = $crate::codegen::OutputInfo::ArrayWrite {
                item: burn_cube::dialect::Item::Scalar(E::cube_elem()),
                local,
                position: burn_cube::dialect::Variable::Id,
            };
            let info = $crate::codegen::CompilationInfo {
                inputs: vec![input, scalars],
                outputs: vec![out],
                scope,
            };
            $crate::codegen::Compilation::new(info).compile(settings)
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::GpuComputeShaderPhase for Ops<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn compile(&self) -> burn_cube::dialect::ComputeShader {
                let settings = $crate::codegen::CompilationSettings::default();
                compile::<E>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::GpuComputeShaderPhase for OpsInplace<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn compile(&self) -> burn_cube::dialect::ComputeShader {
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(vec![mapping]);
                compile::<E>(settings)
            }
        }
    };
}

/// Launch an unary operation.
pub fn unary<Kernel, KernelInplace, R: JitRuntime, E, const D: usize>(
    tensor: JitTensor<R, E, D>,
    scalars: Option<&[E]>,
    inplace_enabled: bool,
    kernel: Kernel,
    kernel_inplace: KernelInplace,
) -> JitTensor<R, E, D>
where
    Kernel: GpuComputeShaderPhase,
    KernelInplace: GpuComputeShaderPhase,
    E: JitElement,
{
    if inplace_enabled && tensor.can_mut() {
        let input_handles = &[EagerHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )];

        let launch = WorkgroupLaunch::Input { pos: 0 };

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

        let input_handles = &[EagerHandle::<R>::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )];

        let output_handles = &[EagerHandle::<R>::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )];

        let launch = WorkgroupLaunch::Output { pos: 0 };

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
