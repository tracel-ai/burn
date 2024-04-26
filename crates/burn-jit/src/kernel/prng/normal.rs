use std::f32::consts::PI;

use burn_tensor::Shape;

use crate::{
    gpu::{gpu, Elem, Scope, Variable},
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::JitTensor,
    JitElement, Runtime,
};

use super::{random, Prng};

pub(crate) struct Normal<E> {
    mean: E,
    std: E,
}

impl<E: JitElement> Prng<E> for Normal<E> {
    fn args(self) -> Vec<E> {
        vec![self.mean, self.std]
    }

    fn inner_loop(
        scope: &mut Scope,
        args: Vec<Variable>,
        write_index_base: Variable,
        n_invocations: Variable,
        n_values_per_thread: usize,
        state_0: Variable,
        state_1: Variable,
        state_2: Variable,
        state_3: Variable,
        output: Variable,
    ) {
        let elem = E::gpu_elem();
        let item = output.item();
        let mean = args[0];
        let std = args[1];
        let two_pi = scope.create_with_value(2. * PI, elem);
        let t_neg = scope.create_with_value(-2.0, item);
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

                let unit_0 = scope.create_local(elem);
                cast_uint_to_float(scope, int_random, unit_0);

                // Second random uniform integer
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
                lcg_step(scope, state_3);

                gpu!(scope, int_random = state_0 ^ state_1);
                gpu!(scope, int_random = int_random ^ state_2);
                gpu!(scope, int_random = int_random ^ state_3);

                let unit_1 = scope.create_local(elem);
                cast_uint_to_float(scope, int_random, unit_1);

                // Box-Muller transform
                let coeff = scope.create_local(item);
                gpu!(scope, coeff = log(unit_0));
                gpu!(scope, coeff *= t_neg);
                gpu!(scope, coeff = sqrt(coeff));
                gpu!(scope, coeff *= std);

                let trigo_arg = scope.create_local(item);
                gpu!(scope, trigo_arg = two_pi * unit_1);

                let normal_0 = scope.create_local(item);
                let normal_1 = scope.create_local(item);
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

    fn args_length() -> usize {
        2
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: Runtime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    mean: E,
    std: E,
) -> JitTensor<R, E, D> {
    random(shape, device, Normal { mean, std })
}
