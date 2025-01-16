use burn_tensor::{Element, ElementConversion};
use cubecl::tune::{local_tuner, tune_with, LocalTuner, TunableSet};

use crate::{
    element::JitElement,
    kernel::{
        prng::random_like_uniform,
        reduce::{
            naive::kernel::reduce_dim_naive, shared::kernel::reduce_dim_shared,
            subcube::kernel::reduce_dim_subcube, ReduceDimAlgorithm,
        },
    },
    tensor::JitTensor,
    tune_key::JitAutotuneKey,
    JitRuntime, JitTuneId,
};

use super::create_key;

pub fn reduce_dim_input_gen<R: JitRuntime, EI: JitElement + Element>(
    _key: &JitAutotuneKey,
    input: &JitTensor<R>,
    reduce_dim: &usize,
) -> (JitTensor<R>, usize) {
    let random_bounds: (EI, EI) = ((-10.0).elem::<EI>(), (10.0).elem::<EI>());
    let input = random_like_uniform(input, random_bounds.0, random_bounds.1);

    tune_with!(input, *reduce_dim)
}

/// Executes autotune on reduce_dim operation
pub(crate) fn reduce_dim_autotune<
    RD: ReduceDimAlgorithm<EI, EO>,
    R: JitRuntime,
    EI: JitElement + Element,
    EO: JitElement + Element,
>(
    input: JitTensor<R>,
    reduce_dim: usize,
) -> JitTensor<R> {
    let client = input.client.clone();

    let id = JitTuneId::new::<R>(&input.device);

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R, EI>, reduce_dim_input_gen::<R, EI>)
        .with_tunable(reduce_dim_naive::<RD, R, EI, EO>)
        .with_tunable(reduce_dim_shared::<RD, R, EI, EO>)
        .with_tunable(reduce_dim_subcube::<RD, R, EI, EO>);

    TUNER.execute(&id, &client, &tunables, (input, reduce_dim))
}
