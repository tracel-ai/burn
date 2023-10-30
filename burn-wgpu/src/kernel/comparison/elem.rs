use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings, StaticKernelSource, WORKGROUP_DEFAULT},
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

        impl $crate::kernel::StaticKernelSource for $struct {
            fn source() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonElemRaw::source()
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

        impl $crate::kernel::StaticKernelSource for $struct {
            fn source() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonElemInplaceRaw::source()
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

pub fn comparison_elem<K: StaticKernelSource, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    let num_elems = lhs.shape.num_elements();

    let handle = lhs.client.empty(num_elems * core::mem::size_of::<u32>());
    let rhs_handle = lhs.client.create(E::as_bytes(&[rhs]));
    let kernel =
        StaticKernel::<KernelSettings<K, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>>::new(
            elemwise_workgroup(num_elems, WORKGROUP_DEFAULT),
        );

    lhs.client
        .execute(Box::new(kernel), &[&lhs.handle, &rhs_handle, &handle]);

    WgpuTensor::new(lhs.client, lhs.device, lhs.shape, handle)
}

pub fn comparison_elem_inplace<K: StaticKernelSource, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    let kernel =
        StaticKernel::<KernelSettings<K, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>>::new(
            elemwise_workgroup(lhs.shape.num_elements(), WORKGROUP_DEFAULT),
        );
    let rhs_handle = lhs.client.create(E::as_bytes(&[rhs]));
    lhs.client
        .execute(Box::new(kernel), &[&lhs.handle, &rhs_handle]);

    WgpuTensor::new(lhs.client, lhs.device, lhs.shape, lhs.handle)
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
