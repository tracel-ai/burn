use super::{elemwise_workgroup, KernelSettings, StaticKernel};
use crate::{element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};

kernel_wgsl!(UnaryRaw, "../template/unary.wgsl");
kernel_wgsl!(UnaryInplaceRaw, "../template/unary_inplace.wgsl");

/// Creates a unary kernel.
#[macro_export]
macro_rules! unary {
    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                let source = $crate::kernel::UnaryRaw::source_template();
                source.register("body", format!("output[id] = {}(input[id]);", $func))
            }
        }
    };
    (
        $struct:ident,
        body $body:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryRaw::source_template().register("body", $body)
            }
        }
    };
    (
        $struct:ident,
        func $func:expr,
        include $file:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryRaw::source_template()
                    .register("body", format!("output[id] = {}(input[id]);", $func))
                    .add_template(include_str!($file))
            }
        }
    };
}

/// Creates a unary inplace kernel.
#[macro_export]
macro_rules! unary_inplace {
    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryInplaceRaw::source_template()
                    .register("body", format!("input[id] = {}(input[id]);", $func))
            }
        }
    };
    (
        $struct:ident,
        body $body:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryInplaceRaw::source_template().register("body", $body)
            }
        }
    };
    (
        $struct:ident,
        func $func:expr,
        include $file:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryInplaceRaw::source_template()
                    .register("body", format!("input[id] = {}(input[id]);", $func))
                    .add_template(include_str!($file))
            }
        }
    };
}

/// Execute a unary kernel using the default settings.
pub fn unary_default<K: StaticKernel, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    unary::<K, E, D, 32>(input)
}

/// Execute a unary inplace kernel using the default settings.
pub fn unary_inplace_default<K: StaticKernel, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    unary_inplace::<K, E, D, 32>(input)
}

/// Execute a unary inplace kernel using the provided WORKGROUP.
pub fn unary_inplace<K: StaticKernel, E: WgpuElement, const D: usize, const WORKGROUP: usize>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let num_elems = input.shape.num_elements();
    let kernel = input
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();

    input.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&input.buffer],
    );

    input
}

/// Execute a unary kernel using the provided WORKGROUP.
pub fn unary<K: StaticKernel, E: WgpuElement, const D: usize, const WORKGROUP: usize>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let num_elems = input.shape.num_elements();
    let buffer = input
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let mut output = WgpuTensor::new(input.context.clone(), input.shape, buffer);
    // Since we don't handle the stride inside the kernel, the output tensor have the same strides
    // as the input tensor. It might not be in the default format.
    output.strides = input.strides;

    let kernel = input
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();

    input.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&input.buffer, &output.buffer],
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    unary!(TestKernel, func "log");
    unary_inplace!(TestKernelInplace, func "log");

    #[test]
    fn unary_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual = unary::<TestKernel, _, 2, 16>(tensor.into_primitive());
        let expected = tensor_ref.log();

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn unary_inplace_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual = unary_inplace::<TestKernelInplace, _, 2, 16>(tensor.into_primitive());
        let expected = tensor_ref.log();

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
