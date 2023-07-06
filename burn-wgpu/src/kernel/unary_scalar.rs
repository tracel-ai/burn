use super::{elemwise_workgroup, KernelSettings, StaticKernel};
use crate::{element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};

kernel_wgsl!(UnaryScalarRaw, "../template/unary_scalar.wgsl");
kernel_wgsl!(
    UnaryScalarInplaceRaw,
    "../template/unary_scalar_inplace.wgsl"
);

/// Creates a unary scalar kernel.
#[macro_export]
macro_rules! unary_scalar {
    (
        $struct:ident,
        ops $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarRaw::source_template()
                    .register("body", format!("output[id] = lhs[id] {} rhs;", $ops))
            }
        }
    };

    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarRaw::source_template()
                    .register("body", format!("output[id] = {}(lhs[id], rhs);", $func))
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
                $crate::kernel::UnaryScalarRaw::source_template()
                    .register("body", format!("output[id] = {}(lhs[id], rhs);", $func))
                    .add_template(include_str!($file))
            }
        }
    };
}

/// Creates a unary scalar inplace kernel.
#[macro_export]
macro_rules! unary_scalar_inplace {
    (
        $struct:ident,
        ops $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarInplaceRaw::source_template()
                    .register("body", format!("lhs[id] = lhs[id] {} rhs;", $ops))
            }
        }
    };

    (
        $struct:ident,
        func $func:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::UnaryScalarInplaceRaw::source_template()
                    .register("body", format!("lhs[id] = {}(lhs[id], rhs);", $func))
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
                $crate::kernel::UnaryScalarInplaceRaw::source_template()
                    .register("body", format!("lhs[id] = {}(lhs[id], rhs);", $func))
                    .add_template(include_str!($file))
            }
        }
    };
}

/// Execute a unary scalar kernel using the default settings.
pub fn unary_scalar_default<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    scalar: E,
) -> WgpuTensor<E, D> {
    unary_scalar::<K, E, D, 32>(lhs, scalar)
}

/// Execute a unary scalar kernel using the provided WORKGROUP.
pub fn unary_scalar<K: StaticKernel, E: WgpuElement, const D: usize, const WORKGROUP: usize>(
    lhs: WgpuTensor<E, D>,
    scalar: E,
) -> WgpuTensor<E, D> {
    let num_elems = lhs.shape.num_elements();
    let buffer = lhs
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), lhs.shape, buffer);
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[scalar]));

    lhs.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&lhs.buffer, &rhs_buffer, &output.buffer],
    );

    output
}

/// Execute a unary scalar inplace kernel using the default settings.
pub fn unary_scalar_inplace_default<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    scalar: E,
) -> WgpuTensor<E, D> {
    unary_scalar_inplace::<K, E, D, 32>(lhs, scalar)
}

/// Execute a unary scalar inplace kernel using the provided WORKGROUP.
pub fn unary_scalar_inplace<
    K: StaticKernel,
    E: WgpuElement,
    const D: usize,
    const WORKGROUP: usize,
>(
    lhs: WgpuTensor<E, D>,
    scalar: E,
) -> WgpuTensor<E, D> {
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[scalar]));

    lhs.context.execute(
        {
            let num_elems = lhs.shape.num_elements();
            elemwise_workgroup(num_elems, WORKGROUP)
        },
        kernel,
        &[&lhs.buffer, &rhs_buffer],
    );

    lhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    unary_scalar!(TestKernel, ops "*");
    unary_scalar_inplace!(TestKernelInplace, ops "*");

    #[test]
    fn unary_scalar_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual = unary_scalar::<TestKernel, _, 2, 16>(tensor.into_primitive(), 5.0);
        let expected = tensor_ref.mul_scalar(5.0);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn unary_scalar_inplace_should_work_with_multiple_invocations() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual =
            unary_scalar_inplace::<TestKernelInplace, _, 2, 16>(tensor.into_primitive(), 5.0);
        let expected = tensor_ref.mul_scalar(5.0);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }
}
