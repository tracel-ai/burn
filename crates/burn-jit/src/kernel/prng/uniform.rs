use burn_tensor::Shape;
use cubecl::{
    cpa,
    ir::{Elem, FloatKind, Scope, Variable},
};

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::JitTensor,
    JitElement, JitRuntime,
};

use super::{random, Prng};

pub(crate) struct Uniform<E> {
    lower_bound: E,
    upper_bound: E,
}

impl<E: JitElement> Prng<E> for Uniform<E> {
    fn args(self) -> Vec<E> {
        vec![self.lower_bound, self.upper_bound]
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
        let lower_bound = args[0];
        let upper_bound = args[1];
        let scale = scope.create_local(item);
        cpa!(scope, scale = upper_bound - lower_bound);

        cpa!(
            scope,
            range(0u32, n_values_per_thread).for_each(|i, scope| {
                taus_step_0(scope, state_0);
                taus_step_1(scope, state_1);
                taus_step_2(scope, state_2);
                lcg_step(scope, state_3);

                let int_random = scope.create_local(Elem::UInt);
                cpa!(scope, int_random = state_0 ^ state_1);
                cpa!(scope, int_random = int_random ^ state_2);
                cpa!(scope, int_random = int_random ^ state_3);

                let float_random = scope.create_local(float_elem);
                let float_scale = scope.create_local(float_elem);
                cast_uint_to_float(scope, int_random, float_random);
                cpa!(scope, float_scale = cast(scale));

                let uniform_float = scope.create_local(float_elem);
                let uniform = scope.create_local(item);
                cpa!(scope, uniform_float = float_random * float_scale);
                cpa!(scope, uniform = cast(uniform_float));
                cpa!(scope, uniform += lower_bound);

                let write_index = scope.create_local(Elem::UInt);
                cpa!(scope, write_index = i * n_invocations);
                cpa!(scope, write_index += write_index_base);
                cpa!(scope, output[write_index] = uniform);
            })
        );
    }

    fn args_length() -> usize {
        2
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    lower_bound: E,
    upper_bound: E,
) -> JitTensor<R, E> {
    random(
        shape,
        device,
        Uniform {
            lower_bound,
            upper_bound,
        },
    )
}
/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform<R: JitRuntime, E: JitElement>(
    tensor: &JitTensor<R, E>,
    lower_bound: E,
    upper_bound: E,
) -> JitTensor<R, E> {
    random_uniform(
        tensor.shape.clone(),
        &tensor.device,
        lower_bound,
        upper_bound,
    )
}
