use crate::{
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings, StaticKernel},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(ComparisonElemRaw, "../../template/comparison/elem.wgsl");
kernel_wgsl!(
    ComparisonElemInplaceRaw,
    "../../template/comparison/elem_inplace.wgsl"
);

/// Creates a comparison elementwise kernel.
#[macro_export]
macro_rules! comparison_elem {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonElemRaw::source_template()
                    .register("body", format!("output[id] = u32(lhs[id] {} rhs);", $ops))
            }
        }
    };
}

/// Creates a comparison elementwise inplace kernel.
#[macro_export]
macro_rules! comparison_elem_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonElemInplaceRaw::source_template()
                    .register("body", "lhs[id] = compare(lhs[id], rhs);")
                    .add_template(format!(
                        "{}return {{{{ elem }}}}(lhs {} rhs);{}",
                        "fn compare(lhs: {{ elem }}, rhs: {{ elem }}) -> {{ elem }} {\n",
                        $ops,
                        "\n}\n"
                    ))
            }
        }
    };
}

pub fn comparison_elem<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    const WORKGROUP: usize = 32;
    let num_elems = lhs.shape.num_elements();

    let buffer = lhs
        .context
        .create_buffer(num_elems * core::mem::size_of::<u32>());
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[rhs]));
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();

    lhs.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&lhs.buffer, &rhs_buffer, &buffer],
    );

    WgpuTensor::new(lhs.context, lhs.shape, buffer)
}

pub fn comparison_elem_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    const WORKGROUP: usize = 32;

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[rhs]));
    lhs.context.execute(
        elemwise_workgroup(lhs.shape.num_elements(), WORKGROUP),
        kernel,
        &[&lhs.buffer, &rhs_buffer],
    );

    WgpuTensor::new(lhs.context, lhs.shape, lhs.buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Bool, Distribution, Tensor};

    comparison_elem!(LowerEqual, "<=");
    comparison_elem_inplace!(LowerEqualInplace, "<=");

    #[test]
    fn comparison_elem_should_work_with_multiple_invocations() {
        let (lhs, lhs_ref, rhs) = inputs();

        let value =
            Tensor::<TestBackend, 3, Bool>::from_primitive(comparison_elem::<LowerEqual, f32, 3>(
                lhs.into_primitive(),
                rhs,
            ));

        let value_ref = lhs_ref.lower_equal_elem(rhs);
        value
            .into_data()
            .assert_approx_eq(&value_ref.into_data(), 3);
    }

    #[test]
    fn comparison_elem_inplace_should_work_with_multiple_invocations() {
        let (lhs, lhs_ref, rhs) = inputs();

        let value =
            Tensor::<TestBackend, 3, Bool>::from_primitive(comparison_elem_inplace::<
                LowerEqualInplace,
                f32,
                3,
            >(lhs.into_primitive(), rhs));

        let value_ref = lhs_ref.lower_equal_elem(rhs);
        value
            .into_data()
            .assert_approx_eq(&value_ref.into_data(), 3);
    }

    #[allow(clippy::type_complexity)]
    fn inputs() -> (Tensor<TestBackend, 3>, Tensor<ReferenceBackend, 3>, f32) {
        TestBackend::seed(0);
        let lhs = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0.0, 1.0));
        let lhs_ref = Tensor::<ReferenceBackend, 3>::from_data(lhs.to_data());

        (lhs, lhs_ref, 5.0)
    }
}
