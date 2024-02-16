use crate::{
    codegen::{execute_static, StaticHandle, WorkgroupLaunch},
    element::JitElement,
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::Shape;

/// Creates a binary kernel.
#[macro_export]
macro_rules! binary {
    (
        operation: $ops:expr,
        runtime: $runtime:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler, elem_in: $elem, elem_out: $elem);

        $crate::kernel::binary::<
            Ops<<$runtime as Runtime>::Compiler, $elem, $elem>,
            OpsInplaceLhs<<$runtime as Runtime>::Compiler, $elem, $elem>,
            OpsInplaceRhs<<$runtime as Runtime>::Compiler, $elem, $elem>,
            $runtime,
            $elem,
            D
        >($lhs, $rhs, true)
    }};

    (
        operation: $ops:expr,
        compiler: $compiler:ty,
        elem_in: $elem_in:ty,
        elem_out: $elem_out:ty
    ) => {
        pub struct Ops<C, I, O> {
            _c: core::marker::PhantomData<C>,
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }
        pub struct OpsInplaceLhs<C, I, O> {
            _c: core::marker::PhantomData<C>,
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }
        pub struct OpsInplaceRhs<C, I, O> {
            _c: core::marker::PhantomData<C>,
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }

        #[allow(clippy::redundant_closure_call)]
        fn compile<C, I, O>(
            settings: $crate::codegen::CompilationSettings,
            mappings: Vec<$crate::codegen::InplaceMapping>,
        ) -> $crate::kernel::SourceTemplate
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            let mut scope = $crate::codegen::dialect::gpu::Scope::root();
            let op = $ops(&mut scope, I::gpu_elem());
            scope.register(op);

            let local = scope.last_local_index().unwrap().index().unwrap();

            let lhs = $crate::codegen::InputInfo::Array {
                item: $crate::codegen::dialect::gpu::Item::Scalar(I::gpu_elem()),
                visibility: $crate::codegen::dialect::gpu::Visibility::Read,
            };
            let rhs = $crate::codegen::InputInfo::Array {
                item: $crate::codegen::dialect::gpu::Item::Scalar(I::gpu_elem()),
                visibility: $crate::codegen::dialect::gpu::Visibility::Read,
            };
            let out = $crate::codegen::OutputInfo::ArrayWrite {
                item: $crate::codegen::dialect::gpu::Item::Scalar(O::gpu_elem()),
                local,
            };
            let info = $crate::codegen::CompilationInfo {
                inputs: vec![lhs, rhs],
                outputs: vec![out],
                scope,
                mappings,
            };
            let shader = $crate::codegen::Compilation::new(info).compile(settings);

            let compiled = C::compile(shader);
            $crate::kernel::SourceTemplate::new(compiled.to_string())
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, I, O> $crate::kernel::StaticKernelSource for Ops<C, I, O>
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let settings = $crate::codegen::CompilationSettings::default();
                compile::<C, I, O>(settings, Vec::new())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, I, O> $crate::kernel::StaticKernelSource
            for OpsInplaceLhs<C, I, O>
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(true);
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                compile::<C, I, O>(settings, vec![mapping])
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, I, O> $crate::kernel::StaticKernelSource
            for OpsInplaceRhs<C, I, O>
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(true);
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 1,
                    pos_output: 0,
                };
                compile::<C, I, O>(settings, vec![mapping])
            }
        }
    };
}

/// Launch an binary operation.
pub fn binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, R: Runtime, E, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    inplace_enabled: bool,
) -> JitTensor<R, E, D>
where
    Kernel: crate::kernel::StaticKernelSource,
    KernelInplaceLhs: crate::kernel::StaticKernelSource,
    KernelInplaceRhs: crate::kernel::StaticKernelSource,
    E: JitElement,
{
    if inplace_enabled && lhs.can_mut_broadcast(&rhs) {
        execute_static::<R, KernelInplaceLhs, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[],
            None,
            WorkgroupLaunch::Input { pos: 0 },
            rhs.client,
        );

        lhs
    } else if inplace_enabled && rhs.can_mut_broadcast(&lhs) {
        execute_static::<R, KernelInplaceRhs, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[],
            None,
            WorkgroupLaunch::Input { pos: 1 },
            lhs.client,
        );

        rhs
    } else {
        let mut shape_out = [0; D];
        lhs.shape
            .dims
            .iter()
            .zip(rhs.shape.dims.iter())
            .enumerate()
            .for_each(|(index, (dim_lhs, dim_rhs))| {
                shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
            });

        let shape_out = Shape::new(shape_out);
        let num_elems = shape_out.num_elements();
        let buffer = lhs.client.empty(num_elems * core::mem::size_of::<E>());
        let out = JitTensor::new(lhs.client.clone(), lhs.device, shape_out, buffer);

        execute_static::<R, Kernel, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[StaticHandle::new(
                &out.handle,
                &out.strides,
                &out.shape.dims,
            )],
            None,
            WorkgroupLaunch::Output { pos: 0 },
            lhs.client,
        );

        out
    }
}
