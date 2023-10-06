use super::{
    build_info, elemwise_workgroup, KernelSettings, StaticKernelSource, WORKGROUP_DEFAULT,
};
use crate::compute::StaticKernel;
use crate::{element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

kernel_wgsl!(BinaryElemwiseRaw, "../template/binary_elemwise.wgsl");
kernel_wgsl!(
    BinaryElemwiseInplaceRaw,
    "../template/binary_elemwise_inplace.wgsl"
);

/// Creates a binary elementwise kernel.
#[macro_export]
macro_rules! binary_elemwise {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernelSource for $struct {
            fn source() -> $crate::kernel::SourceTemplate {
                $crate::kernel::BinaryElemwiseRaw::source().register(
                    "body",
                    format!("output[id] = lhs[index_lhs] {} rhs[index_rhs];", $ops),
                )
            }
        }
    };
}

/// Creates a binary elementwise inplace kernel.
#[macro_export]
macro_rules! binary_elemwise_inplace {
    (
        $struct:ident,
        $ops:expr
    ) => {
        pub struct $struct;

        impl $crate::kernel::StaticKernelSource for $struct {
            fn source() -> $crate::kernel::SourceTemplate {
                $crate::kernel::BinaryElemwiseInplaceRaw::source().register(
                    "body",
                    format!("lhs[id] = lhs[id] {} rhs[index_rhs];", $ops),
                )
            }
        }
    };
}

/// Execute a binary kernel using the default settings.
pub fn binary_elemwise_default<K: StaticKernelSource, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary_elemwise::<K, E, D, WORKGROUP_DEFAULT>(lhs, rhs)
}

/// Execute a binary kernel using the provided WORKGROUP.
pub fn binary_elemwise<
    K: StaticKernelSource,
    E: WgpuElement,
    const D: usize,
    const WORKGROUP: usize,
>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_same_device(&rhs);

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

    let handle = lhs.client.empty(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.client.clone(), lhs.device.clone(), shape_out, handle);

    let info = build_info(&[&lhs, &rhs, &output]);
    let info_handle = lhs.client.create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
        elemwise_workgroup(num_elems, WORKGROUP),
    );

    lhs.client.execute(
        Box::new(kernel),
        &[&lhs.handle, &rhs.handle, &output.handle, &info_handle],
    );

    output
}

/// Execute a binary inplace kernel using the default settings.
pub fn binary_elemwise_inplace_default<K: StaticKernelSource, E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    binary_elemwise_inplace::<K, E, D, WORKGROUP_DEFAULT>(lhs, rhs)
}

/// Execute a binary inplace kernel using the provided WORKGROUP.
pub fn binary_elemwise_inplace<
    K: StaticKernelSource,
    E: WgpuElement,
    const D: usize,
    const WORKGROUP: usize,
>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_same_device(&rhs);

    let info = build_info(&[&lhs, &rhs]);
    let info_handle = lhs.client.create(bytemuck::cast_slice(&info));
    let kernel = StaticKernel::<KernelSettings<K, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
        elemwise_workgroup(lhs.shape.num_elements(), WORKGROUP),
    );

    lhs.client
        .execute(Box::new(kernel), &[&lhs.handle, &rhs.handle, &info_handle]);

    lhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    binary_elemwise!(TestKernel, "*");
    binary_elemwise_inplace!(TestKernelInplace, "*");

    #[test]
    fn binary_should_work_with_multiple_invocations() {
        let lhs = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let rhs = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let lhs_ref = Tensor::<ReferenceBackend, 2>::from_data(lhs.to_data());
        let rhs_ref = Tensor::<ReferenceBackend, 2>::from_data(rhs.to_data());

        let actual =
            binary_elemwise::<TestKernel, _, 2, 16>(lhs.into_primitive(), rhs.into_primitive());
        let expected = lhs_ref * rhs_ref;

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn binary_inplace_should_work_with_multiple_invocations() {
        let lhs = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let rhs = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let lhs_ref = Tensor::<ReferenceBackend, 2>::from_data(lhs.to_data());
        let rhs_ref = Tensor::<ReferenceBackend, 2>::from_data(rhs.to_data());

        let actual = binary_elemwise_inplace::<TestKernelInplace, _, 2, 16>(
            lhs.into_primitive(),
            rhs.into_primitive(),
        );
        let expected = lhs_ref * rhs_ref;

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }
}
