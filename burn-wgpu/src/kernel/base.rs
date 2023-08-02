use super::SourceTemplate;
use crate::{context::WorkGroup, element::WgpuElement, tensor::WgpuTensor};
use std::marker::PhantomData;

/// Static wgpu kernel to create a [source template](SourceTemplate).
pub trait StaticKernel: 'static {
    /// Source template for the kernel.
    fn source_template() -> SourceTemplate;
}

/// Dynamic wgpu kernel to create a [source template](SourceTemplate).
pub trait DynamicKernel {
    /// Source template for the kernel.
    fn source_template(self) -> SourceTemplate;
    /// Identifier for the kernel, used for caching kernel compilation.
    fn id(&self) -> String;
}

/// Generates kernel source code by replacing some information using templating.
#[macro_export]
macro_rules! kernel_wgsl {
    (
        $struct:ident,
        $file:expr
    ) => {
        /// Generated kernel from wgsl file.
        #[derive(new)]
        pub struct $struct;

        impl $crate::kernel::StaticKernel for $struct {
            fn source_template() -> $crate::kernel::SourceTemplate {
                $crate::kernel::SourceTemplate::new(include_str!($file))
            }
        }
    };
}

kernel_wgsl!(ContiguousRaw, "../template/contiguous.wgsl");

pub(crate) fn into_contiguous<E: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    if tensor.is_contiguous() {
        return tensor;
    }

    const WORKGROUP: usize = 32;

    let num_elems = tensor.shape.num_elements();
    let buffer = tensor
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(tensor.context.clone(), tensor.shape.clone(), buffer);
    let info = build_info(&[&tensor, &output]);
    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor
        .context
        .compile_static::<KernelSettings<ContiguousRaw, E, i32, WORKGROUP, WORKGROUP, 1>>();

    tensor.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&tensor.buffer, &output.buffer, &info_buffer],
    );

    output
}

/// Generates kernel source code by replacing some information using templating.
pub struct KernelSettings<
    K: StaticKernel,
    E: WgpuElement,
    I: WgpuElement,
    const WORKGROUP_X_SIZE: usize,
    const WORKGROUP_Y_SIZE: usize,
    const WORKGROUP_Z_SIZE: usize,
> {
    _k: PhantomData<K>,
    _e: PhantomData<E>,
    _i: PhantomData<I>,
}

impl<
        K: StaticKernel,
        E: WgpuElement,
        I: WgpuElement,
        const WORKGROUP_X_SIZE: usize,
        const WORKGROUP_Y_SIZE: usize,
        const WORKGROUP_Z_SIZE: usize,
    > StaticKernel
    for KernelSettings<K, E, I, WORKGROUP_X_SIZE, WORKGROUP_Y_SIZE, WORKGROUP_Z_SIZE>
{
    fn source_template() -> SourceTemplate {
        K::source_template()
            .register("workgroup_size_x", WORKGROUP_X_SIZE.to_string())
            .register("workgroup_size_y", WORKGROUP_Y_SIZE.to_string())
            .register("workgroup_size_z", WORKGROUP_Z_SIZE.to_string())
            .register(
                "workgroup_size",
                (WORKGROUP_X_SIZE * WORKGROUP_Y_SIZE * WORKGROUP_Z_SIZE).to_string(),
            )
            .register("elem", E::type_name())
            .register("int", I::type_name())
    }
}

/// Generate kernel source code by replacing some information using templating.
#[derive(new)]
pub struct DynamicKernelSettings<K: StaticKernel, E: WgpuElement, I: WgpuElement> {
    workgroup_x_size: usize,
    workgroup_y_size: usize,
    workgroup_z_size: usize,
    _k: PhantomData<K>,
    _e: PhantomData<E>,
    _i: PhantomData<I>,
}

impl<K: StaticKernel, E: WgpuElement, I: WgpuElement> DynamicKernel
    for DynamicKernelSettings<K, E, I>
{
    fn source_template(self) -> SourceTemplate {
        K::source_template()
            .register("workgroup_size_x", self.workgroup_x_size.to_string())
            .register("workgroup_size_y", self.workgroup_y_size.to_string())
            .register("workgroup_size_z", self.workgroup_z_size.to_string())
            .register(
                "workgroup_size",
                (self.workgroup_x_size * self.workgroup_y_size * self.workgroup_z_size).to_string(),
            )
            .register("elem", E::type_name())
            .register("int", I::type_name())
    }

    fn id(&self) -> String {
        let id = core::any::TypeId::of::<K>();

        format!(
            "{:?}-dyn-settings{}-{}-{}",
            id, self.workgroup_x_size, self.workgroup_y_size, self.workgroup_z_size
        )
    }
}

/// Create a vector containing the dimension, strides and shape of tensors.
///
/// # Example
///
/// With two tensors (lhs, rhs)
///
/// | Indexes                  | Value       |
/// |:------------------------:|:-----------:|
/// |           0..1           | D           |
/// |           1..D + 1       | lhs strides |
/// |     (D + 1)..(2 * D + 1) | rhs strides |
/// | (2 * D + 1)..(3 * D + 1) | lhs shape   |
/// | (3 * D + 1)..(4 * D + 1) | rhs shape   |
pub(crate) fn build_info<E: WgpuElement, const D: usize>(
    tensors: &[&WgpuTensor<E, D>],
) -> Vec<u32> {
    let mut info: Vec<u32> = vec![0; tensors.len() * 2 * D + 1];
    info[0] = D as u32;

    let mut current = 1;
    for tensor in tensors.iter() {
        for d in 0..D {
            info[current] = tensor.strides[d] as u32;
            current += 1;
        }
    }
    for tensor in tensors.iter() {
        for d in 0..D {
            info[current] = tensor.shape.dims[d] as u32;
            current += 1;
        }
    }
    info
}

pub(crate) fn elemwise_workgroup(num_elems: usize, workgroup_size: usize) -> WorkGroup {
    let num_elem_per_invocation = workgroup_size * workgroup_size;
    let workgroups = f32::ceil(num_elems as f32 / num_elem_per_invocation as f32);
    let workgroup_x = f32::ceil(f32::sqrt(workgroups));
    let workgroup_y = f32::ceil(num_elems as f32 / (workgroup_x * num_elem_per_invocation as f32));

    WorkGroup::new(workgroup_x as u32, workgroup_y as u32, 1)
}

pub(crate) fn prng_workgroup(
    num_elems: usize,
    workgroup_size: usize,
    n_values_per_thread: usize,
) -> WorkGroup {
    let num_threads = f32::ceil(num_elems as f32 / n_values_per_thread as f32);
    let num_elem_per_invocation = workgroup_size * workgroup_size;
    let num_invocations = f32::ceil(num_threads / num_elem_per_invocation as f32);
    let workgroup_x = f32::ceil(f32::sqrt(num_invocations));
    let workgroup_y = f32::ceil(num_invocations / workgroup_x);

    WorkGroup::new(workgroup_x as u32, workgroup_y as u32, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::any::TypeId;

    #[test]
    fn test_kernel_type_id() {
        kernel_wgsl!(Add, "../template/binary_elemwise.wgsl");

        let type_id_1 = TypeId::of::<KernelSettings<Add, f32, i32, 2, 3, 4>>();
        let type_id_2 = TypeId::of::<KernelSettings<Add, f32, i32, 2, 3, 5>>();
        let type_id_3 = TypeId::of::<KernelSettings<Add, f32, i32, 2, 3, 4>>();

        assert_ne!(type_id_1, type_id_2);
        assert_eq!(type_id_1, type_id_3);
    }
}
