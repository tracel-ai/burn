use std::marker::PhantomData;

use burn_tensor::Shape;

use crate::{
    codegen::{
        Compilation, CompilationInfo, CompilationSettings, EagerHandle, Execution, InputInfo,
        OutputInfo, WorkgroupLaunch,
    },
    gpu::{gpu, Elem, Scope, Variable},
    kernel::{
        prng::{
            cast_uint_to_float, get_seeds, lcg_step, taus_step_0, taus_step_1, taus_step_2,
            PrngShader,
        },
        DynamicKernelSource, SourceTemplate,
    },
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

use super::Prng;

pub(crate) struct Bernoulli {
    probability: Variable,
}

impl Prng for Bernoulli {
    fn inner_loop(
        &self,
        scope: &mut Scope,
        write_index_base: Variable,
        n_invocations: Variable,
        n_values_per_thread: usize,
        state_0: Variable,
        state_1: Variable,
        state_2: Variable,
        state_3: Variable,
        output: Variable,
    ) {
        let prob = self.probability;
        gpu!(
            scope,
            range(0u32, n_values_per_thread).for_each(|i, scope| {
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
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
    }
}

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    prob: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);
    let kernel: PrngEagerKernel<Bernoulli, R, E> = PrngEagerKernel::new();
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
#[derive(new)]
pub(crate) struct PrngEagerKernel<P: Prng, R: Runtime, E: JitElement> {
    _prng: PhantomData<P>,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<P: Prng, R: Runtime, E: JitElement> DynamicKernelSource for PrngEagerKernel<P, R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();
        const N_VALUES_PER_THREAD: usize = 128;

        let output = Variable::GlobalOutputArray(0, item);
        let probability = Variable::GlobalScalar(0, E::gpu_elem());

        let seed0 = Variable::GlobalScalar(0, Elem::UInt);
        let seed1 = Variable::GlobalScalar(1, Elem::UInt);
        let seed2 = Variable::GlobalScalar(2, Elem::UInt);
        let seed3 = Variable::GlobalScalar(3, Elem::UInt);
        let seeds = [seed0, seed1, seed2, seed3];

        PrngShader::new(
            output,
            N_VALUES_PER_THREAD,
            seeds,
            Bernoulli { probability },
        )
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
