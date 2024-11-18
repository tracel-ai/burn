use burn_tensor::{Element, ElementConversion};
use cubecl::tune::{local_tuner, tune_with, LocalTuner};
use cubecl::{tune, Feature};

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

/// Set of reduce_dim implementations available for autotune
/// Autotune key is given by concatenating the closest upper power of 2 of
/// dim to reduce, and product of others
#[tune(
    operations(reduce_dim_naive, reduce_dim_shared, reduce_dim_subcube),
    create_key = create_key::<R, EI>,
    should_run = should_run
)]
pub fn reduce_dim_operations<
    RD: ReduceDimAlgorithm<EI, EO>,
    R: JitRuntime,
    EI: JitElement + Element,
    EO: JitElement + Element,
>(
    key: JitAutotuneKey,
    input: JitTensor<R>,
    reduce_dim: usize,
) -> JitTensor<R> {
    let random_bounds: (EI, EI) = ((-10.0).elem::<EI>(), (10.0).elem::<EI>());
    let input = random_like_uniform(input, random_bounds.0, random_bounds.1);

    tune_with!(input, reduce_dim)
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

    let operation_set = Box::new(ReduceDimOperations::<RD, R, EI, EO>::new(input, reduce_dim));

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(&id, &client, operation_set)
}

fn should_run<
    RD: ReduceDimAlgorithm<EI, EO>,
    R: JitRuntime,
    EI: JitElement + Element,
    EO: JitElement + Element,
>(
    op: &ReduceDimOperations<RD, R, EI, EO>,
    key: &JitAutotuneKey,
    index: usize,
) -> bool {
    let JitAutotuneKey::ReduceDim(key) = key else {
        unreachable!()
    };

    match index {
        // Naive
        0 => key.reduce_dim_length <= 8192,
        // Shared
        1 => key.reduce_dim_length >= 16,
        // Subcube
        2 => {
            let props = op.input.client.properties();
            let hardware = props.hardware_properties();
            props.feature_enabled(Feature::Plane)
                && hardware.plane_size_min == hardware.plane_size_max
        }
        _ => true,
    }
}
