use super::StaticKernelSource;
use crate::{codegen::execute_static, element::WgpuElement, tensor::WgpuTensor};

/// Creates a unary kernel.
#[macro_export]
macro_rules! unary {
    (
        operator: $ops:expr,
        input: $input:expr,
        elem: $elem:ty
    ) => {{
        unary!($ops, $input);

        $crate::kernel::unary::<Ops<$elem>, OpsInplace<$elem>, $elem, D>($input, None)
    }};
    (
        operator: $ops:expr,
        input: $input:expr; $scalar:expr,
        elem: $elem:ty
    ) => {{
        unary!($ops, $input, scalar);

        $crate::kernel::unary::<Ops<$elem>, OpsInplace<$elem>, $elem, D>($input, Some(&[$scalar]))
    }};

    (
        $ops:expr,
        $input:expr
    ) => {
        pub struct Ops<E> {
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplace<E> {
            _e: core::marker::PhantomData<E>,
        }

        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for Ops<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::KernelCodegen::new()
                    .inputs(
                        &[$crate::codegen::ArrayInput::new(
                            E::elem_type(),
                            $crate::codegen::Visibility::Read,
                        )],
                        &[],
                    )
                    .body(&[$ops(E::elem_type())])
                    .outputs(&[$crate::codegen::ArrayOutput::new(E::elem_type(), 0)])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }

        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for OpsInplace<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::KernelCodegen::new()
                    .inputs(
                        &[$crate::codegen::ArrayInput::new(
                            E::elem_type(),
                            $crate::codegen::Visibility::ReadWrite,
                        )],
                        &[],
                    )
                    .body(&[
                        $ops(E::elem_type()),
                        Operator::AssignGlobal {
                            input: Variable::Local(0, E::elem_type()),
                            out: Variable::Input(0, E::elem_type()),
                        },
                    ])
                    .outputs(&[])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }
    };
    (
        $ops:expr,
        $input:expr,
        scalar
    ) => {
        pub struct Ops<E> {
            _e: core::marker::PhantomData<E>,
        }
        pub struct OpsInplace<E> {
            _e: core::marker::PhantomData<E>,
        }

        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for Ops<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::KernelCodegen::new()
                    .inputs(
                        &[$crate::codegen::ArrayInput::new(
                            E::elem_type(),
                            $crate::codegen::Visibility::Read,
                        )],
                        &[$crate::codegen::ScalarInput::new(E::elem_type(), 1)],
                    )
                    .body(&[$ops(E::elem_type())])
                    .outputs(&[$crate::codegen::ArrayOutput::new(E::elem_type(), 0)])
                    .compile();

                $crate::kernel::SourceTemplate::new(shader.to_string())
            }
        }

        impl<E: $crate::element::WgpuElement> $crate::kernel::StaticKernelSource for OpsInplace<E> {
            fn source() -> $crate::kernel::SourceTemplate {
                let shader = $crate::codegen::KernelCodegen::new()
                    .inputs(
                        &[$crate::codegen::ArrayInput::new(
                            E::elem_type(),
                            $crate::codegen::Visibility::ReadWrite,
                        )],
                        &[$crate::codegen::ScalarInput::new(E::elem_type(), 1)],
                    )
                    .body(&[
                        $ops(E::elem_type()),
                        Operator::AssignGlobal {
                            input: Variable::Local(0, E::elem_type()),
                            out: Variable::Input(0, E::elem_type()),
                        },
                    ])
                    .outputs(&[])
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
    println!("Unary scalar");
    if !tensor.can_mut() {
        let num_elems = tensor.shape.num_elements();
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let mut output = WgpuTensor::new(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );
        // Since we don't handle the stride inside the kernel, the output tensor have the same strides
        // as the input tensor. It might not be in the default format.
        output.strides = tensor.strides;

        execute_static::<K, E>(
            &[(&tensor.handle, &tensor.strides, &tensor.shape.dims)],
            &[(&output.handle, &output.strides, &output.shape.dims)],
            scalars,
            tensor.client,
        );

        output
    } else {
        execute_static::<KI, E>(
            &[],
            &[(&tensor.handle, &tensor.strides, &tensor.shape.dims)],
            scalars,
            tensor.client.clone(),
        );

        tensor
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tests::{ReferenceBackend, TestBackend};
//     use burn_tensor::{Distribution, Tensor};
//
//     unary!(TestKernel, func "log");
//     unary_inplace!(TestKernelInplace, func "log");
//
//     #[test]
//     fn unary_should_work_with_multiple_invocations() {
//         let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
//         let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
//
//         let actual = unary::<TestKernel, _, 2, 16>(tensor.into_primitive());
//         let expected = tensor_ref.log();
//
//         expected.into_data().assert_approx_eq(
//             &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
//             3,
//         );
//     }
//
//     #[test]
//     fn unary_inplace_should_work_with_multiple_invocations() {
//         let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
//         let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
//
//         let actual = unary_inplace::<TestKernelInplace, _, 2, 16>(tensor.into_primitive());
//         let expected = tensor_ref.log();
//
//         expected.into_data().assert_approx_eq(
//             &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
//             3,
//         );
//     }
//
//     #[test]
//     fn tanh_should_not_have_numerical_bugs_on_macos() {
//         fn tanh_one_value(input: f32) -> f32 {
//             let tensor = Tensor::<TestBackend, 1>::ones([1]) * input;
//             let output = tensor.tanh().into_primitive();
//             Tensor::<TestBackend, 1>::from_primitive(output)
//                 .into_data()
//                 .value[0]
//         }
//
//         let ok = tanh_one_value(43.0); // metal tanh gives 1.0 which is the right answer
//         let zero = tanh_one_value(44.0); // metal tanh gives zero when within 43.67..44.36
//         let nan = tanh_one_value(45.0); // metal tanh gives nan when over 44.36
//         let neg = tanh_one_value(-45.0); //  metal works correctly here
//
//         assert!(!ok.is_nan() && ok == 1.0);
//         assert!(!zero.is_nan() && zero == 1.0);
//         assert!(!nan.is_nan() && nan == 1.0);
//         assert!(!neg.is_nan() && neg == -1.0);
//     }
// }
