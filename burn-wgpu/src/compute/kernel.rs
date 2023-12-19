use super::Kernel;
use crate::kernel::{DynamicKernelSource, SourceTemplate, StaticKernelSource};
use core::marker::PhantomData;

/// Provides launch information specifying the number of work groups to be used by a compute shader.
#[derive(new, Clone, Debug)]
pub struct WorkGroup {
    /// Work groups for the x axis.
    pub x: u32,
    /// Work groups for the y axis.
    pub y: u32,
    /// Work groups for the z axis.
    pub z: u32,
}

impl WorkGroup {
    /// Calculate the number of invocations of a compute shader.
    pub fn num_invocations(&self) -> usize {
        (self.x * self.y * self.z) as usize
    }
}

/// Wraps a [dynamic kernel source](DynamicKernelSource) into a [kernel](Kernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct DynamicKernel<K> {
    kernel: K,
    workgroup: WorkGroup,
}

/// Wraps a [static kernel source](StaticKernelSource) into a [kernel](Kernel) with launch
/// information such as [workgroup](WorkGroup).
#[derive(new)]
pub struct StaticKernel<K> {
    workgroup: WorkGroup,
    _kernel: PhantomData<K>,
}

impl<K> Kernel for DynamicKernel<K>
where
    K: DynamicKernelSource + 'static,
{
    fn source(&self) -> SourceTemplate {
        self.kernel.source()
    }

    fn id(&self) -> String {
        self.kernel.id()
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

impl<K> Kernel for StaticKernel<K>
where
    K: StaticKernelSource + 'static,
{
    fn source(&self) -> SourceTemplate {
        K::source()
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<K>())
    }

    fn workgroup(&self) -> WorkGroup {
        self.workgroup.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        binary,
        codegen::{Elem, Operator, Variable},
        compute::compute_client,
        kernel::{KernelSettings, WORKGROUP_DEFAULT},
        AutoGraphicsApi, WgpuDevice,
    };

    #[test]
    fn can_run_kernel() {
        binary!(
            operator: |elem: Elem| Operator::Add {
                lhs: Variable::Input(0, elem),
                rhs: Variable::Input(1, elem),
                out: Variable::Local(0, elem),
            },
            elem_in: f32,
            elem_out: f32
        );

        let client = compute_client::<AutoGraphicsApi>(&WgpuDevice::default());

        let lhs: Vec<f32> = vec![0., 1., 2., 3., 4., 5., 6., 7.];
        let rhs: Vec<f32> = vec![10., 11., 12., 6., 7., 3., 1., 0.];
        let info: Vec<u32> = vec![1, 1, 8, 1, 8, 1, 8];

        let lhs = client.create(bytemuck::cast_slice(&lhs));
        let rhs = client.create(bytemuck::cast_slice(&rhs));
        let out = client.empty(core::mem::size_of::<f32>() * 8);
        let info = client.create(bytemuck::cast_slice(&info));

        type Kernel =
            KernelSettings<Ops<f32, f32>, f32, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>;
        let kernel = Box::new(StaticKernel::<Kernel>::new(WorkGroup::new(1, 1, 1)));

        client.execute(kernel, &[&lhs, &rhs, &out, &info]);

        let data = client.read(&out).read_sync().unwrap();
        let output: &[f32] = bytemuck::cast_slice(&data);

        assert_eq!(output, [10., 12., 14., 9., 11., 8., 7., 7.]);
    }
}
