use cubecl::{
    cpa,
    ir::{Elem, FloatKind, Scope, Variable},
};
use std::f32::consts::PI;

use burn_tensor::Shape;

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::JitTensor,
    JitElement, JitRuntime,
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
        let float_elem = Elem::Float(FloatKind::F32);
        let item = output.item();
        let mean = args[0];
        let std = args[1];
        let two_pi = scope.create_with_value(2. * PI, float_elem);
        let t_neg = scope.create_with_value(-2.0, item);
        let two: Variable = 2u32.into();

        cpa!(
            scope,
            range(0u32, n_values_per_thread / 2).for_each(|i, scope| {
                let int_random = scope.create_local(Elem::UInt);

                // First random uniform integer
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
                lcg_step(scope, state_3);

                cpa!(scope, int_random = state_0 ^ state_1);
                cpa!(scope, int_random = int_random ^ state_2);
                cpa!(scope, int_random = int_random ^ state_3);

                let unit_0 = scope.create_local(float_elem);
                cast_uint_to_float(scope, int_random, unit_0);

                // Second random uniform integer
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
                lcg_step(scope, state_3);

                cpa!(scope, int_random = state_0 ^ state_1);
                cpa!(scope, int_random = int_random ^ state_2);
                cpa!(scope, int_random = int_random ^ state_3);

                let unit_1 = scope.create_local(float_elem);
                cast_uint_to_float(scope, int_random, unit_1);

                // Box-Muller transform
                let coeff = scope.create_local(item);
                cpa!(scope, coeff = log(unit_0));
                cpa!(scope, coeff *= t_neg);
                cpa!(scope, coeff = sqrt(coeff));
                cpa!(scope, coeff *= std);

                let trigo_arg = scope.create_local(item);
                cpa!(scope, trigo_arg = two_pi * unit_1);

                let normal_0 = scope.create_local(item);
                let normal_1 = scope.create_local(item);
                cpa!(scope, normal_0 = cos(trigo_arg));
                cpa!(scope, normal_0 *= coeff);
                cpa!(scope, normal_0 += mean);
                cpa!(scope, normal_1 = sin(trigo_arg));
                cpa!(scope, normal_1 *= coeff);
                cpa!(scope, normal_1 += mean);

                // Write to output
                let write_index_0 = scope.create_local(Elem::UInt);
                let write_index_1 = scope.create_local(Elem::UInt);
                let iteration_offset = scope.create_local(Elem::UInt);
                cpa!(scope, write_index_0 = write_index_base);
                cpa!(scope, iteration_offset = two * i);
                cpa!(scope, iteration_offset *= n_invocations);
                cpa!(scope, write_index_0 += iteration_offset);
                cpa!(scope, write_index_1 = write_index_0 + n_invocations);

                cpa!(scope, output[write_index_0] = normal_0);
                cpa!(scope, output[write_index_1] = normal_1);
            })
        );
    }

    fn args_length() -> usize {
        2
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    mean: E,
    std: E,
) -> JitTensor<R, E> {
    random(shape, device, Normal { mean, std })
}
