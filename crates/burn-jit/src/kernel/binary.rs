use crate::{
    codegen::{EagerHandle, Execution, WorkgroupLaunch},
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
        >($lhs, $rhs, true, Ops::new(), OpsInplaceLhs::new(), OpsInplaceRhs::new())
    }};

    (
        operation: $ops:expr,
        compiler: $compiler:ty,
        elem_in: $elem_in:ty,
        elem_out: $elem_out:ty
    ) => {
        #[derive(new)]
        pub struct Ops<C, I, O> {
            _c: core::marker::PhantomData<C>,
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }
        #[derive(new)]
        pub struct OpsInplaceLhs<C, I, O> {
            _c: core::marker::PhantomData<C>,
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }
        #[derive(new)]
        pub struct OpsInplaceRhs<C, I, O> {
            _c: core::marker::PhantomData<C>,
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }

        #[allow(clippy::redundant_closure_call)]
        fn compile<I, O>(
            settings: $crate::codegen::CompilationSettings,
        ) -> $crate::gpu::ComputeShader
        where
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            let mut scope = $crate::codegen::dialect::gpu::Scope::root();
            let position = $crate::codegen::dialect::gpu::Variable::Id;

            let op = $ops(&mut scope, I::gpu_elem(), position);
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
                position,
            };
            let info = $crate::codegen::CompilationInfo {
                inputs: vec![lhs, rhs],
                outputs: vec![out],
                scope,
            };
            $crate::codegen::Compilation::new(info).compile(settings)
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, I, O> $crate::kernel::GpuComputeShaderPhase for Ops<C, I, O>
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            fn compile(&self) -> $crate::gpu::ComputeShader {
                let settings = $crate::codegen::CompilationSettings::default();
                compile::<I, O>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, I, O> $crate::kernel::GpuComputeShaderPhase
            for OpsInplaceLhs<C, I, O>
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            fn compile(&self) -> $crate::gpu::ComputeShader {
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 0,
                    pos_output: 0,
                };
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(vec![mapping]);
                compile::<I, O>(settings)
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, I, O> $crate::kernel::GpuComputeShaderPhase
            for OpsInplaceRhs<C, I, O>
        where
            C: $crate::codegen::Compiler,
            I: $crate::element::JitElement,
            O: $crate::element::JitElement
        {
            fn compile(&self) -> $crate::gpu::ComputeShader {
                let mapping = $crate::codegen::InplaceMapping {
                    pos_input: 1,
                    pos_output: 0,
                };
                let settings = $crate::codegen::CompilationSettings::default()
                    .inplace(vec![mapping]);
                compile::<I, O>(settings)
            }
        }
    };
}

/// Launch an binary operation.
pub fn binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, R: Runtime, E, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    inplace_enabled: bool,
    kernel: Kernel,
    kernel_inplace_lhs: KernelInplaceLhs,
    kernel_inplace_rhs: KernelInplaceRhs,
) -> JitTensor<R, E, D>
where
    Kernel: crate::kernel::GpuComputeShaderPhase,
    KernelInplaceLhs: crate::kernel::GpuComputeShaderPhase,
    KernelInplaceRhs: crate::kernel::GpuComputeShaderPhase,
    E: JitElement,
{
    if inplace_enabled && lhs.can_mut_broadcast(&rhs) {
        Execution::start(kernel_inplace_lhs, rhs.client)
            .inputs(&[
                EagerHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                EagerHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ])
            .execute(WorkgroupLaunch::Input { pos: 0 });

        lhs
    } else if inplace_enabled && rhs.can_mut_broadcast(&lhs) {
        Execution::start(kernel_inplace_rhs, lhs.client)
            .inputs(&[
                EagerHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                EagerHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ])
            .execute(WorkgroupLaunch::Input { pos: 1 });

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

        Execution::start(kernel, lhs.client)
            .inputs(&[
                EagerHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                EagerHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ])
            .outputs(&[EagerHandle::new(&out.handle, &out.strides, &out.shape.dims)])
            .execute(WorkgroupLaunch::Output { pos: 0 });

        out
    }
}
