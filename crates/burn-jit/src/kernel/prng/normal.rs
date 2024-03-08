use std::f32::consts::PI;

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
        prng_workgroup, DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT,
    },
    tensor::JitTensor,
    Compiler, JitElement, Runtime,
};

use super::{Prng, PrngEagerKernel, N_VALUES_PER_THREAD};

pub(crate) struct Normal {
    mean: Variable,
    std: Variable,
}

impl Prng for Normal {
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
        let mean = self.mean;
        let std = self.std;
        let two_pi = scope.create_with_value(2. * PI, Elem::Float);
        let t_neg = scope.create_with_value(-2.0, output.item());
        let two: Variable = 2u32.into();

        gpu!(
            scope,
            range(0u32, n_values_per_thread / 2).for_each(|i, scope| {
                let int_random = scope.create_local(Elem::UInt);

                // First random uniform integer
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
                lcg_step(scope, state_3);

                gpu!(scope, int_random = state_0 ^ state_1);
                gpu!(scope, int_random = int_random ^ state_2);
                gpu!(scope, int_random = int_random ^ state_3);

                let unit_0 = scope.create_local(Elem::Float);
                cast_uint_to_float(scope, int_random, unit_0);

                // Second random uniform integer
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
                lcg_step(scope, state_3);

                gpu!(scope, int_random = state_0 ^ state_1);
                gpu!(scope, int_random = int_random ^ state_2);
                gpu!(scope, int_random = int_random ^ state_3);

                let unit_1 = scope.create_local(Elem::Float);
                cast_uint_to_float(scope, int_random, unit_1);

                // Box-Muller transform
                let coeff = scope.create_local(output.item());
                gpu!(scope, coeff = log(unit_0));
                gpu!(scope, coeff *= t_neg);
                gpu!(scope, coeff = sqrt(coeff));
                gpu!(scope, coeff *= std);

                let trigo_arg = scope.create_local(output.item());
                gpu!(scope, trigo_arg = two_pi * unit_1);

                let normal_0 = scope.create_local(output.item());
                let normal_1 = scope.create_local(output.item());
                gpu!(scope, normal_0 = cos(trigo_arg));
                gpu!(scope, normal_0 *= coeff);
                gpu!(scope, normal_0 += mean);
                gpu!(scope, normal_1 = sin(trigo_arg));
                gpu!(scope, normal_1 *= coeff);
                gpu!(scope, normal_1 += mean);

                // Write to output
                let write_index_0 = scope.create_local(Elem::UInt);
                let write_index_1 = scope.create_local(Elem::UInt);
                let iteration_offset = scope.create_local(Elem::UInt);
                gpu!(scope, write_index_0 = write_index_base);
                gpu!(scope, iteration_offset = two * i);
                gpu!(scope, iteration_offset *= n_invocations);
                gpu!(scope, write_index_0 += iteration_offset);
                gpu!(scope, write_index_1 = write_index_0 + n_invocations);

                gpu!(scope, output[write_index_0] = normal_0);
                gpu!(scope, output[write_index_1] = normal_1);
            })
        );
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    mean: E,
    std: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);
    let kernel: PrngEagerKernel<Normal, R, E> = PrngEagerKernel::new();
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
        .with_scalars(&[mean, std])
        .with_scalars(&seeds)
        .execute(WorkgroupLaunch::Custom(prng_workgroup(
            num_elems,
            WORKGROUP_DEFAULT,
            N_VALUES_PER_THREAD,
        )));

    output
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for PrngEagerKernel<Normal, R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let output = Variable::GlobalOutputArray(0, item);
        let mean = Variable::GlobalScalar(0, E::gpu_elem());
        let std = Variable::GlobalScalar(1, E::gpu_elem());

        let seed0 = Variable::GlobalScalar(0, Elem::UInt);
        let seed1 = Variable::GlobalScalar(1, Elem::UInt);
        let seed2 = Variable::GlobalScalar(2, Elem::UInt);
        let seed3 = Variable::GlobalScalar(3, Elem::UInt);
        let seeds = [seed0, seed1, seed2, seed3];

        PrngShader::new(output, N_VALUES_PER_THREAD, seeds, Normal { mean, std })
            .expand(&mut scope);

        scope.write_global_custom(output);

        let prob = InputInfo::Scalar {
            elem: E::gpu_elem(),
            size: 2,
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
