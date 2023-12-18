use crate::{
    codegen::{execute_static, GridLaunch, StaticHandle},
    element::WgpuElement,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

/// Creates a unary kernel.
#[macro_export]
macro_rules! binary {
    (
        operator: $ops:expr,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!($ops);

        $crate::kernel::binary::<Ops<$elem>, OpsInplaceLhs<$elem>, OpsInplaceRhs<$elem>, $elem, D>(
            $lhs, $rhs,
        )
    }};

    ($ops:expr) => {
        pub struct Ops<E> {
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplaceLhs<E> {
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplaceRhs<E> {
            _e: core::marker::PhantomData<E>,
        }

        #[allow(clippy::redundant_closure_call)]
        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for Ops<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::IntoContiguous,
                        },
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::IntoContiguous,
                        },
                    ])
                    .body(&[$ops(E::elem_type())])
                    .outputs(&[$crate::codegen::Output::Array {
                        elem: E::elem_type(),
                        local: 0,
                    }])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource
            for OpsInplaceLhs<E>
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::ReadWrite,
                            strategy: $crate::codegen::ReadingStrategy::Plain,
                        },
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::IntoContiguous,
                        },
                    ])
                    .body(&[$ops(E::elem_type())])
                    .outputs(&[$crate::codegen::Output::Input {
                        elem: E::elem_type(),
                        input: 0,
                        local: 0,
                    }])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource
            for OpsInplaceRhs<E>
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::IntoContiguous,
                        },
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::ReadWrite,
                            strategy: $crate::codegen::ReadingStrategy::Plain,
                        },
                    ])
                    .body(&[$ops(E::elem_type())])
                    .outputs(&[$crate::codegen::Output::Input {
                        elem: E::elem_type(),
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
) -> WgpuTensor<E, D>
where
    Kernel: crate::kernel::StaticKernelSource,
    KernelInplaceLhs: crate::kernel::StaticKernelSource,
    KernelInplaceRhs: crate::kernel::StaticKernelSource,
    E: WgpuElement,
{
    if lhs.can_mut_broadcast(&rhs) {
        execute_static::<KernelInplaceLhs, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[],
            None,
            GridLaunch::Input { pos: 0 },
            rhs.client,
        );

        lhs
    } else if rhs.can_mut_broadcast(&lhs) {
        execute_static::<KernelInplaceRhs, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[],
            None,
            GridLaunch::Input { pos: 1 },
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
            GridLaunch::Output { pos: 0 },
            lhs.client,
        );

        out
    }
}
