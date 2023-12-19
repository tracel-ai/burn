use crate::{
    codegen::{execute_static, StaticHandle, WorkgroupLaunch},
    element::WgpuElement,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

/// Creates a binary kernel.
#[macro_export]
macro_rules! binary {
    (
        operator: $ops:expr,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operator: $ops, elem_in: $elem, elem_out: $elem);

        $crate::kernel::binary::<Ops<$elem, $elem>, OpsInplaceLhs<$elem, $elem>, OpsInplaceRhs<$elem, $elem>, $elem, D>(
            $lhs, $rhs, true
        )
    }};

    (
        operator: $ops:expr,
        elem_in: $elem_in:ty,
        elem_out: $elem_out:ty
    ) => {
        pub struct Ops<I, O> {
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }
        pub struct OpsInplaceLhs<I, O> {
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }
        pub struct OpsInplaceRhs<I, O> {
            _i: core::marker::PhantomData<I>,
            _o: core::marker::PhantomData<O>,
        }

        #[allow(clippy::redundant_closure_call)]
        impl<I, O> $crate::kernel::StaticKernelSource for Ops<I, O>
        where
            I: $crate::element::WgpuElement,
            O: $crate::element::WgpuElement
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: I::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::OutputLayout,
                        },
                        $crate::codegen::Input::Array {
                            elem: I::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::OutputLayout,
                        },
                    ])
                    .body(&[$ops(I::elem_type())])
                    .outputs(&[$crate::codegen::Output::Array {
                        elem: O::elem_type(),
                        local: 0,
                    }])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<I, O> $crate::kernel::StaticKernelSource
            for OpsInplaceLhs<I, O>
        where
            I: $crate::element::WgpuElement,
            O: $crate::element::WgpuElement
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: I::elem_type(),
                            visibility: $crate::codegen::Visibility::ReadWrite,
                            strategy: $crate::codegen::ReadingStrategy::Plain,
                        },
                        $crate::codegen::Input::Array {
                            elem: I::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::OutputLayout,
                        },
                    ])
                    .body(&[$ops(I::elem_type())])
                    .outputs(&[$crate::codegen::Output::Input {
                        elem: I::elem_type(),
                        input: 0,
                        local: 0,
                    }])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<I, O> $crate::kernel::StaticKernelSource
            for OpsInplaceRhs<I, O>
        where
            I: $crate::element::WgpuElement,
            O: $crate::element::WgpuElement
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: I::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::OutputLayout,
                        },
                        $crate::codegen::Input::Array {
                            elem: I::elem_type(),
                            visibility: $crate::codegen::Visibility::ReadWrite,
                            strategy: $crate::codegen::ReadingStrategy::Plain,
                        },
                    ])
                    .body(&[$ops(I::elem_type())])
                    .outputs(&[$crate::codegen::Output::Input {
                        elem: I::elem_type(),
                        input: 1,
                        local: 0,
                    }])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }
    };
}

/// Launch an binary operation.
pub fn binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, E, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    inplace_enabled: bool,
) -> WgpuTensor<E, D>
where
    Kernel: crate::kernel::StaticKernelSource,
    KernelInplaceLhs: crate::kernel::StaticKernelSource,
    KernelInplaceRhs: crate::kernel::StaticKernelSource,
    E: WgpuElement,
{
    if inplace_enabled && lhs.can_mut_broadcast(&rhs) {
        execute_static::<KernelInplaceLhs, E>(
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
        execute_static::<KernelInplaceRhs, E>(
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
        let out = WgpuTensor::new(lhs.client.clone(), lhs.device, shape_out, buffer);

        execute_static::<Kernel, E>(
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
