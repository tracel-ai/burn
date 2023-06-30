use super::{KernelSettings, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};

kernel_wgsl!(ComparisonElemRaw, "../template/comparison/elem.wgsl");
kernel_wgsl!(
    ComparisonElemInplaceRaw,
    "../template/comparison/elem_inplace.wgsl"
);

// Fn bool 2 u32 conversion.
pub const BOOL_2_U32: &str = "fn bool2u32(val: bool) -> u32 {
    if val {
        return 1u;
    }

    return 0u;
}";

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
                    .add_template($crate::kernel::BOOL_2_U32)
                    .register(
                        "body",
                        format!(
                            "output[global_id.x] = bool2u32(lhs[global_id.x] {} rhs);",
                            $ops
                        ),
                    )
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
                    .register("body", "lhs[global_id.x] = compare(lhs[global_id.x], rhs);")
                    .add_template($crate::kernel::BOOL_2_U32)
                    .add_template(format!(
                        "{}return {{{{ elem }}}}(bool2u32(lhs {} rhs));{}",
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
    let buffer = lhs
        .context
        .create_buffer(lhs.shape.num_elements() * core::mem::size_of::<u32>());
    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[rhs]));
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();

    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs_buffer, &buffer],
    );

    WgpuTensor::new(lhs.context, lhs.shape, buffer)
}

pub fn comparison_elem_inplace<K: StaticKernel, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    let kernel = lhs
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();

    let rhs_buffer = lhs.context.create_buffer_with_data(E::as_bytes(&[rhs]));
    lhs.context.execute(
        WorkGroup::new(
            f32::ceil(lhs.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&lhs.buffer, &rhs_buffer],
    );

    WgpuTensor::new(lhs.context, lhs.shape, lhs.buffer)
}
