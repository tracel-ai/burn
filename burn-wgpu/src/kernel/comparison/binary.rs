use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings, StaticKernel},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

kernel_wgsl!(ComparisonRaw, "../../template/comparison/binary.wgsl");
kernel_wgsl!(
    ComparisonInplaceRaw,
    "../../template/comparison/binary_inplace.wgsl"
);

/// Creates a comparison kernel.
#[macro_export]
macro_rules! comparison {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonRaw::source_template().register(
                    "body",
                    format!("output[id] = u32(lhs[index_lhs] {} rhs[index_rhs]);", $ops),
                )
            }
        }
    };
}

/// Creates a comparison inplace kernel.
#[macro_export]
macro_rules! comparison_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::ComparisonInplaceRaw::source_template()
                    .register(
                        "body",
                        "lhs[index_lhs] = compare(lhs[index_lhs], rhs[index_rhs]);",
                    )
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

pub fn comparison<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    lhs.assert_is_on_same_device(&rhs);
    const WORKGROUP: usize = 32;

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

    let buffer = lhs
        .context
        .create_buffer(num_elems * core::mem::size_of::<u32>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let info = build_info(&[&lhs, &rhs, &output]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    WgpuTensor::new(output.context, output.shape, output.buffer)
}

pub fn comparison_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    const WORKGROUP: usize = 32;

    lhs.assert_is_on_same_device(&rhs);

    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let info = build_info(&[&lhs, &rhs]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    lhs.context.execute(
        elemwise_workgroup(lhs.shape.num_elements(), WORKGROUP),
        kernel,
        &[&lhs.buffer, &rhs.buffer, &info_buffers],
    );

    WgpuTensor::new(lhs.context, lhs.shape, lhs.buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Bool, Distribution, Tensor};

    comparison!(LowerEqual, "<=");
    comparison_inplace!(LowerEqualInplace, "<=");

    #[test]
    fn comparison_should_work_with_multiple_invocations() {
        let (lhs, rhs, lhs_ref, rhs_ref) = inputs();

        let value = Tensor::<TestBackend, 3, Bool>::from_primitive(
            comparison::<LowerEqual, f32, 3>(lhs.into_primitive(), rhs.into_primitive()),
        );

        let value_ref = lhs_ref.lower_equal(rhs_ref);
        value
            .into_data()
            .assert_approx_eq(&value_ref.into_data(), 3);
    }

    #[test]
    fn comparison_inplace_should_work_with_multiple_invocations() {
        let (lhs, rhs, lhs_ref, rhs_ref) = inputs();

        let value = Tensor::<TestBackend, 3, Bool>::from_primitive(comparison_inplace::<
            LowerEqualInplace,
            f32,
            3,
        >(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ));

        let value_ref = lhs_ref.lower_equal(rhs_ref);
        value
            .into_data()
            .assert_approx_eq(&value_ref.into_data(), 3);
    }

    #[allow(clippy::type_complexity)]
    fn inputs() -> (
        Tensor<TestBackend, 3>,
        Tensor<TestBackend, 3>,
        Tensor<ReferenceBackend, 3>,
        Tensor<ReferenceBackend, 3>,
    ) {
        TestBackend::seed(0);
        let lhs = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0.0, 1.0));
        let rhs = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0.0, 1.0));
        let lhs_ref = Tensor::<ReferenceBackend, 3>::from_data(lhs.to_data());
        let rhs_ref = Tensor::<ReferenceBackend, 3>::from_data(rhs.to_data());

        (lhs, rhs, lhs_ref, rhs_ref)
    }
}
