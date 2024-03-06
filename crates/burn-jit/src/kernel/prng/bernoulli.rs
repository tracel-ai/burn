use std::marker::PhantomData;

use burn_tensor::Shape;

use crate::{
    codegen::{
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, EagerHandle, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, Elem, Scope, Variable},
    kernel::{prng::get_seeds, DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

/// Pseudo-random generator for bernoulli
pub fn random_bernoulli<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    prob: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);
    let kernel = BernoulliEagerKernel::new();
    let num_elems = shape.num_elements();
    let buffer = client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(client.clone(), device.clone(), shape.clone(), buffer);

    execute_dynamic::<R, BernoulliEagerKernel<R, E>, E>(
        &[],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        Some(&[prob]),
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        client,
    );

    output
}

pub(crate) struct BernoulliShader {
    output: Variable,
    n_values_per_thread: usize,
    probability: Variable,
    seeds: [u32; 4],
}

pub(crate) struct BernoulliEagerKernel<R: Runtime, E: JitElement> {
    seeds: [u32; 4],
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> BernoulliEagerKernel<R, E> {

}

impl<R: Runtime, E: JitElement> DynamicKernelSource for BernoulliEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let output = Variable::GlobalOutputArray(0, item);
        const N_VALUES_PER_THREAD: usize = 128;
        let probability = Variable::GlobalScalar(0, E::gpu_elem());
        let seeds = get_seeds();

        BernoulliShader {
            output,
            n_values_per_thread: N_VALUES_PER_THREAD,
            probability,
            seeds,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let input = InputInfo::Scalar {
            elem: E::gpu_elem(),
            size: 1,
        };
        let out = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!(
            "{:?} seeds={:?}",
            core::any::TypeId::of::<Self>(),
            self.seeds
        )
    }
}

impl BernoulliShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let id = Variable::Id;
        let output = self.output;
        let n_values_per_thread: Variable = self.n_values_per_thread.into();
        let probability = self.probability;

        let workgroup_size_y = Variable::WorkgroupSizeY;
        let workgroup_id_x = Variable::WorkgroupIdX;
        let workgroup_id_y = Variable::WorkgroupIdY;
        let num_workgroups_y = Variable::NumWorkgroupsY;
        let local_index = Variable::LocalInvocationIndex;

        let n_invocations = Variable::WorkgroupSizeX;
        gpu!(scope, n_invocations *= workgroup_size_y);

        let workgroup_offset = scope.create_local(Elem::UInt);
        gpu!(scope, workgroup_offset = workgroup_id_x * num_workgroups_y);
        gpu!(scope, workgroup_offset += workgroup_id_y);
        gpu!(scope, workgroup_offset *= n_invocations);

        let write_index_base = scope.create_local(Elem::UInt);
        gpu!(scope, write_index_base = workgroup_offset);
        gpu!(scope, write_index_base += local_index);

        let thread_seed = scope.create_local(Elem::UInt);
        gpu!(scope, thread_seed = cast(1000000007));
        let thread_seed_index = scope.create_local(Elem::UInt);
        gpu!(scope, thread_seed_index = workgroup_offset + local_index);
        gpu!(scope, thread_seed *= thread_seed_index);

        let state_0 = scope.create_local(Elem::UInt);
        let state_1 = scope.create_local(Elem::UInt);
        let state_2 = scope.create_local(Elem::UInt);
        let state_3 = scope.create_local(Elem::UInt);
    }
}
