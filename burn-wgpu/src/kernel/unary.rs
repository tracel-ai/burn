use super::StaticKernelSource;
use crate::{
    codegen::{execute_static, StaticHandle, WorkgroupLaunch},
    element::WgpuElement,
    tensor::WgpuTensor,
};

/// Creates a unary kernel.
#[macro_export]
macro_rules! unary {
    (
        operator: $ops:expr,
        compiler: $compiler:ty,
        input: $input:expr,
        elem: $elem:ty
    ) => {{
        unary!(operator: $ops, compiler: $compiler);

        $crate::kernel::unary::<Ops<$compiler, $elem>, OpsInplace<$compiler, $elem>, $elem, D>($input, None, true)
    }};
    (
        operator: $ops:expr,
        compiler: $compiler:ty,
        input: $input:expr; $scalar:expr,
        elem: $elem:ty
    ) => {{
        unary!(operator: $ops, compiler: $compiler, scalar 1);

        $crate::kernel::unary::<Ops<$compiler, $elem>, OpsInplace<$compiler, $elem>, $elem, D>($input, Some(&[$scalar]), true)
    }};

    (
        operator: $ops:expr,
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
        impl<C, E> $crate::kernel::StaticKernelSource for Ops<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::WgpuElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[$crate::codegen::Input::Array {
                        item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                        visibility: $crate::codegen::dialect::gpu::Visibility::Read,
                        strategy: $crate::codegen::ReadingStrategy::OutputLayout,
                    }])
                    .body(&[$ops(E::gpu_elem())])
                    .outputs(&[$crate::codegen::Output::Array {
                        item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                        local: 0,
                    }])
                    .compile();

                let compiled = C::compile(shader);
                $crate::kernel::SourceTemplate::new(compiled.to_string())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::StaticKernelSource for OpsInplace<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::WgpuElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[$crate::codegen::Input::Array {
                        item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                        visibility: $crate::codegen::dialect::gpu::Visibility::ReadWrite,
                        strategy: $crate::codegen::ReadingStrategy::Plain,
                    }])
                    .body(&[$ops(E::gpu_elem())])
                    .outputs(&[$crate::codegen::Output::Input {
                        item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                        input: 0,
                        local: 0,
                    }])
                    .compile();

                let compiled = C::compile(shader);
                $crate::kernel::SourceTemplate::new(compiled.to_string())
            }
        }
    };
    (
        operator: $ops:expr,
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
        impl<C, E> $crate::kernel::StaticKernelSource for Ops<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::WgpuElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                            visibility: $crate::codegen::dialect::gpu::Visibility::Read,
                            strategy: $crate::codegen::ReadingStrategy::OutputLayout,
                        },
                        $crate::codegen::Input::Scalar {
                            elem: E::gpu_elem(),
                            size: $num,
                        },
                    ])
                    .body(&[$ops(E::gpu_elem())])
                    .outputs(&[$crate::codegen::Output::Array {
                        item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                        local: 0,
                    }])
                    .compile();

                let compiled = C::compile(shader);
                $crate::kernel::SourceTemplate::new(compiled.to_string())
            }
        }

        #[allow(clippy::redundant_closure_call)]
        impl<C, E> $crate::kernel::StaticKernelSource for OpsInplace<C, E>
        where
            C: $crate::codegen::Compiler,
            E: $crate::element::WgpuElement,
        {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                            visibility: $crate::codegen::dialect::gpu::Visibility::ReadWrite,
                            strategy: $crate::codegen::ReadingStrategy::Plain,
                        },
                        $crate::codegen::Input::Scalar {
                            elem: E::gpu_elem(),
                            size: $num,
                        },
                    ])
                    .body(&[$ops(E::gpu_elem())])
                    .outputs(&[$crate::codegen::Output::Input {
                        item: $crate::codegen::dialect::gpu::Item::Scalar(E::gpu_elem()),
                        input: 0,
                        local: 0,
                    }])
                    .compile();

                let compiled = C::compile(shader);
                $crate::kernel::SourceTemplate::new(compiled.to_string())
            }
        }
    };
}

/// Launch an unary operation.
pub fn unary<Kernel, KernelInplace, E, const D: usize>(
    tensor: WgpuTensor<E, D>,
    scalars: Option<&[E]>,
    inplace_enabled: bool,
) -> WgpuTensor<E, D>
where
    Kernel: StaticKernelSource,
    KernelInplace: StaticKernelSource,
    E: WgpuElement,
{
    if inplace_enabled && tensor.can_mut() {
        execute_static::<KernelInplace, E>(
            &[StaticHandle::new(
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
        let output = WgpuTensor::new(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );

        execute_static::<Kernel, E>(
            &[StaticHandle::new(
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            )],
            &[StaticHandle::new(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::dialect::gpu::{Item, Operation, UnaryOperation, Variable};
    use crate::tests::{ReferenceBackend, TestBackend, TestCompiler};
    use burn_tensor::{Distribution, Tensor};

    unary!(
        operator: |elem| Operation::Tanh(UnaryOperation {
            input: Variable::Input(0, Item::Scalar(elem)),
            out: Variable::Local(0, Item::Scalar(elem)),
        }),
        compiler: TestCompiler
    );

    #[test]
    fn unary_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let actual = unary::<Ops<TestCompiler, f32>, OpsInplace<TestCompiler, f32>, f32, 2>(
            tensor.into_primitive(),
            None,
            true,
        );
        let expected = tensor_ref.tanh();

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn unary_inplace_should_work_with_multiple_invocations() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());

        let actual = unary::<Ops<TestCompiler, f32>, OpsInplace<TestCompiler, f32>, f32, 2>(
            tensor.into_primitive(),
            None,
            true,
        );
        let expected = tensor_ref.tanh();

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn tanh_should_not_have_numerical_bugs_on_macos() {
        fn tanh_one_value(input: f32) -> f32 {
            let tensor = Tensor::<TestBackend, 1>::ones([1], &Default::default()) * input;
            let output = tensor.tanh().into_primitive();
            Tensor::<TestBackend, 1>::from_primitive(output)
                .into_data()
                .value[0]
        }

        let ok = tanh_one_value(43.0); // metal tanh gives 1.0 which is the right answer
        let zero = tanh_one_value(44.0); // metal tanh gives zero when within 43.67..44.36
        let nan = tanh_one_value(45.0); // metal tanh gives nan when over 44.36
        let neg = tanh_one_value(-45.0); //  metal works correctly here

        assert!(!ok.is_nan() && ok == 1.0);
        assert!(!zero.is_nan() && zero == 1.0);
        assert!(!nan.is_nan() && nan == 1.0);
        assert!(!neg.is_nan() && neg == -1.0);
    }
}
