use std::marker::PhantomData;

use burn_tensor::Shape;
use num_traits::float;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
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
    let kernel: BernoulliEagerKernel<R, E> = BernoulliEagerKernel::new();
    let num_elems = shape.num_elements();
    let buffer = client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(client.clone(), device.clone(), shape.clone(), buffer);
    let seeds = get_seeds();

    Execution::start(kernel, client)
        .outputs(&[EagerHandle::<R>::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .with_scalars(&[prob])
        .with_scalars(&seeds)
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}

pub(crate) struct BernoulliShader {
    output: Variable,
    n_values_per_thread: usize,
    probability: Variable,
    seeds: [Variable; 4],
}

#[derive(new)]
pub(crate) struct BernoulliEagerKernel<R: Runtime, E: JitElement> {
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for BernoulliEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let output = Variable::GlobalOutputArray(0, item);
        const N_VALUES_PER_THREAD: usize = 128;
        let probability = Variable::GlobalScalar(0, E::gpu_elem());
        let seed0 = Variable::GlobalScalar(0, Elem::UInt);
        let seed1 = Variable::GlobalScalar(1, Elem::UInt);
        let seed2 = Variable::GlobalScalar(2, Elem::UInt);
        let seed3 = Variable::GlobalScalar(3, Elem::UInt);
        let seeds = [seed0, seed1, seed2, seed3];

        BernoulliShader {
            output,
            n_values_per_thread: N_VALUES_PER_THREAD,
            probability,
            seeds,
        }
        .expand(&mut scope);

        scope.write_global_custom(output);

        let prob = InputInfo::Scalar {
            elem: E::gpu_elem(),
            size: 1,
        };
        let seeds = InputInfo::Scalar {
            elem: Elem::UInt,
            size: 4,
        };
        let out = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![prob, seeds],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}", core::any::TypeId::of::<Self>(),)
    }
}

impl BernoulliShader {
    pub(crate) fn expand(self, scope: &mut Scope) {
        let output = self.output;
        let [seed_0, seed_1, seed_2, seed_3] = self.seeds;

        let workgroup_size_x = Variable::WorkgroupSizeX;
        let workgroup_size_y = Variable::WorkgroupSizeY;
        let workgroup_id_x = Variable::WorkgroupIdX;
        let workgroup_id_y = Variable::WorkgroupIdY;
        let num_workgroups_y = Variable::NumWorkgroupsY;
        let local_index = Variable::LocalInvocationIndex;

        let n_invocations = scope.create_local(Elem::UInt);
        gpu!(scope, n_invocations = workgroup_size_x);
        gpu!(scope, n_invocations *= workgroup_size_y);

        let workgroup_offset = scope.create_local(Elem::UInt);
        gpu!(scope, workgroup_offset = workgroup_id_x * num_workgroups_y);
        gpu!(scope, workgroup_offset += workgroup_id_y);
        gpu!(scope, workgroup_offset *= n_invocations);

        let write_index_base = scope.create_local(Elem::UInt);
        gpu!(scope, write_index_base = workgroup_offset);
        gpu!(scope, write_index_base += local_index);

        // Set state with unique seeds
        let thread_seed = scope.create_local(Elem::UInt);
        gpu!(scope, thread_seed = cast(1000000007));
        let thread_seed_index = scope.create_local(Elem::UInt);
        gpu!(scope, thread_seed_index = workgroup_offset + local_index);
        gpu!(scope, thread_seed *= thread_seed_index);

        let state_0 = scope.create_local(Elem::UInt);
        gpu!(scope, state_0 = thread_seed);
        gpu!(scope, state_0 += seed_0);

        let state_1 = scope.create_local(Elem::UInt);
        gpu!(scope, state_1 = thread_seed);
        gpu!(scope, state_1 += seed_1);

        let state_2 = scope.create_local(Elem::UInt);
        gpu!(scope, state_2 = thread_seed);
        gpu!(scope, state_2 += seed_2);

        let state_3 = scope.create_local(Elem::UInt);
        gpu!(scope, state_3 = thread_seed);
        gpu!(scope, state_3 += seed_3);

        // Creation of n_values_per_thread values, specific to the distribution

        ///////////////////////
        // BERNOULLI specific
        let prob = self.probability;
        gpu!(
            scope,
            range(0u32, self.n_values_per_thread).for_each(|i, scope| {
                taus_step(
                    scope,
                    state_0,
                    13u32.into(),
                    19u32.into(),
                    12u32.into(),
                    4294967294u32.into(),
                );
                taus_step(
                    scope,
                    state_1,
                    2u32.into(),
                    25u32.into(),
                    4u32.into(),
                    4294967288u32.into(),
                );
                taus_step(
                    scope,
                    state_2,
                    3u32.into(),
                    11u32.into(),
                    17u32.into(),
                    4294967280u32.into(),
                );
                lcg_step(scope, state_3);

                let int_random = scope.create_local(Elem::UInt);
                gpu!(scope, int_random = state_0 ^ state_1);
                gpu!(scope, int_random = int_random ^ state_2);
                gpu!(scope, int_random = int_random ^ state_3);

                let float_random = scope.create_local(Elem::Float);
                cast_uint_to_float(scope, int_random, float_random);

                let bernoulli = scope.create_local(Elem::Bool);
                gpu!(scope, bernoulli = float_random < prob);

                let i: Variable = i.into();
                let write_index = scope.create_local(Elem::UInt);
                gpu!(scope, write_index = i * n_invocations);
                gpu!(scope, write_index += write_index_base);
                gpu!(scope, output[write_index] = bernoulli);
            })
        );
        //
        //////////////////////////
    }
}

fn taus_step(
    scope: &mut Scope,
    z: Variable,
    s1: Variable,
    s2: Variable,
    s3: Variable,
    m: Variable,
) {
    let b = scope.create_local(Elem::UInt);
    gpu!(scope, b = z << s1);
    gpu!(scope, b = b ^ z);
    gpu!(scope, b = b >> s2);
    gpu!(scope, z = z & m);
    gpu!(scope, z = z << s3);
    gpu!(scope, z = z ^ b);
}

fn lcg_step(scope: &mut Scope, z: Variable) {
    let a: Variable = 1664525u32.into();
    let b: Variable = 1013904223u32.into();
    gpu!(scope, z *= a);
    gpu!(scope, z += b);
}

fn cast_uint_to_float(scope: &mut Scope, int_random: Variable, float_random: Variable) {
    let tmp: Variable = 2.3283064365387e-10.into();
    gpu!(scope, float_random = cast(int_random));
    gpu!(scope, float_random *= tmp);
}
