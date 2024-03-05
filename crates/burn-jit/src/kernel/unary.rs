use super::StaticKernelSource;
use crate::{
    codegen::{execute_static, EagerHandle, WorkgroupLaunch},
    element::JitElement,
    tensor::JitTensor,
    Runtime,
};

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
        >($input, None, true)
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
        >($input, Some(&[$scalar]), true)
    }};

    (
        operation: $ops:expr,
        compiler: $compiler:ty
    ) => {
        pub struct Ops<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplace<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }

        #[allow(clippy::redundant_closure_call)]
        fn compile<C, E>(
            settings: $crate::codegen::CompilationSettings,
        ) -> $crate::kernel::SourceTemplate
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement
        {

            let mut scope = $crate::codegen::dialect::gpu::Scope::root();
            let op = $ops(&mut scope, E::gpu_elem());
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let input = $crate::codegen::InputInfo::Array {
                item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                visibility: $crate::codegen::dialect::gpu::Visibility::Read,
            };
            let out = $crate::codegen::OutputInfo::ArrayWrite {
                item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                local,
            };
            let info = $crate::codegen::CompilationInfo {
                inputs: vec![input],
                outputs: vec![out],
                scope,
            };
            let shader = $crate::codegen::Compilation::new(info).compile(settings);

            let compiled = C::compile(shader);
            $crate::kernel::SourceTemplate::new(compiled.to_string())
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::StaticKernelSource for Ops<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let settings = $crate::codegen::CompilationSettings::default();
                compile::<C, E>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::StaticKernelSource for OpsInplace<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(vec![mapping]);
                compile::<C, E>(settings)
            }
        }
    };
    (
        operation: $ops:expr,
        compiler: $compiler:ty,
        scalar $num:expr
    ) => {
        pub struct Ops<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplace<C, E> {
            _c: core::marker::PhantomData<C>,
            _e: core::marker::PhantomData<E>,
        }

        #[allow(clippy::redundant_closure_call)]
        fn compile<C, E>(
            settings: $crate::codegen::CompilationSettings,
        ) -> $crate::kernel::SourceTemplate
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement
        {

            let mut scope = $crate::codegen::dialect::gpu::Scope::root();
            let op = $ops(&mut scope, E::gpu_elem());
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let input = $crate::codegen::InputInfo::Array {
                item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                visibility: $crate::codegen::dialect::gpu::Visibility::Read,
            };
            let scalars = $crate::codegen::InputInfo::Scalar {
                elem: E::gpu_elem(),
                size: $num,
            };
            let out = $crate::codegen::OutputInfo::ArrayWrite {
                item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                local,
            };
            let info = $crate::codegen::CompilationInfo {
                inputs: vec![input, scalars],
                outputs: vec![out],
                scope,
            };
            let shader = $crate::codegen::Compilation::new(info).compile(settings);

            let compiled = C::compile(shader);
            $crate::kernel::SourceTemplate::new(compiled.to_string())
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::StaticKernelSource for Ops<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let settings = $crate::codegen::CompilationSettings::default();
                compile::<C, E>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::StaticKernelSource for OpsInplace<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::JitElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(vec![mapping]);
                compile::<C, E>(settings)
            }
        }
    };
}

/// Launch an unary operation.
pub fn unary<Kernel, KernelInplace, R: Runtime, E, const D: usize>(
    tensor: JitTensor<R, E, D>,
    scalars: Option<&[E]>,
    inplace_enabled: bool,
) -> JitTensor<R, E, D>
where
    Kernel: StaticKernelSource,
    KernelInplace: StaticKernelSource,
    E: JitElement,
{
    if inplace_enabled && tensor.can_mut() {
        execute_static::<R, KernelInplace, E>(
            &[EagerHandle::new(
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            )],
            &[],
            scalars,
            WorkgroupLaunch::Input { pos: 0 },
            tensor.client.clone(),
        );

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

        execute_static::<R, Kernel, E>(
            &[EagerHandle::new(
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            )],
            &[EagerHandle::new(
                &output.handle,
                &output.strides,
                &output.shape.dims,
            )],
            scalars,
            WorkgroupLaunch::Output { pos: 0 },
            tensor.client,
        );

        output
    }
}
