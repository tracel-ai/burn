use super::StaticKernelSource;
use crate::{
    codegen::{execute_static, StaticHandle},
    element::WgpuElement,
    tensor::WgpuTensor,
};

/// Creates a unary kernel.
#[macro_export]
macro_rules! unary {
    (
        operator: $ops:expr,
        input: $input:expr,
        elem: $elem:ty
    ) => {{
        unary!($ops);

        $crate::kernel::unary::<Ops<$elem>, OpsInplace<$elem>, $elem, D>($input, None)
    }};
    (
        operator: $ops:expr,
        input: $input:expr; $scalar:expr,
        elem: $elem:ty
    ) => {{
        unary!($ops, scalar 1);

        $crate::kernel::unary::<Ops<$elem>, OpsInplace<$elem>, $elem, D>($input, Some(&[$scalar]))
    }};

    (
        $ops:expr
    ) => {
        pub struct Ops<E> {
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplace<E> {
            _e: core::marker::PhantomData<E>,
        }

        #[allow(clippy::redundant_closure_call)]
        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for Ops<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[$crate::codegen::Input::Array {
                        elem: E::elem_type(),
                        visibility: $crate::codegen::Visibility::Read,
                        strategy: $crate::codegen::ReadingStrategy::IntoContiguous,
                    }])
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
        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for OpsInplace<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[$crate::codegen::Input::Array {
                        elem: E::elem_type(),
                        visibility: $crate::codegen::Visibility::ReadWrite,
                        strategy: $crate::codegen::ReadingStrategy::Plain,
                    }])
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
    };
    (
        $ops:expr,
        scalar $num:expr
    ) => {
        pub struct Ops<E> {
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplace<E> {
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
                        $crate::codegen::Input::Scalar {
                            elem: E::elem_type(),
                            size: $num,
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
        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for OpsInplace<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::ElemWiseKernelCodegen::new()
                    .inputs(&[
                        $crate::codegen::Input::Array {
                            elem: E::elem_type(),
                            visibility: $crate::codegen::Visibility::ReadWrite,
                            strategy: $crate::codegen::ReadingStrategy::Plain,
                        },
                        $crate::codegen::Input::Scalar {
                            elem: E::elem_type(),
                            size: $num,
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
    };
}

/// Launch an unary operation.
pub fn unary<K, KI, E, const D: usize>(
    tensor: WgpuTensor<E, D>,
    scalars: Option<&[E]>,
) -> WgpuTensor<E, D>
where
    K: StaticKernelSource,
    KI: StaticKernelSource,
    E: WgpuElement,
{
    if !tensor.can_mut() {
        let num_elems = tensor.shape.num_elements();
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let output = WgpuTensor::new(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );

        execute_static::<K, E>(
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
            tensor.client,
        );

        output
    } else {
        execute_static::<KI, E>(
            &[],
            &[StaticHandle::new(
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            )],
            scalars,
            tensor.client.clone(),
        );

        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{Operator, Variable};
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    unary!(|elem| Operator::Tanh {
        input: Variable::Input(0, elem),
        out: Variable::Local(0, elem),
    });

    #[test]
    fn unary_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual = unary::<Ops<f32>, OpsInplace<f32>, f32, 2>(tensor.into_primitive(), None);
        let expected = tensor_ref.tanh();

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn unary_inplace_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual = unary::<Ops<f32>, OpsInplace<f32>, f32, 2>(tensor.into_primitive(), None);
        let expected = tensor_ref.tanh();

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn tanh_should_not_have_numerical_bugs_on_macos() {
        fn tanh_one_value(input: f32) -> f32 {
            let tensor = Tensor::<TestBackend, 1>::ones([1]) * input;
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
