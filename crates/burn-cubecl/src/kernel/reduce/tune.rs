#![allow(missing_docs)]

use super::SumAutotuneKey;
use crate::{CubeAutotuneKey, CubeRuntime, CubeTuneId, tensor::CubeTensor};
use cubecl::{
    client::ComputeClient,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::tune_key::ReduceAutotuneKey,
    routines::{RoutineStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

/// Executes autotune on reduce operations.
pub fn autotune_reduce<R: CubeRuntime>(
    client: &ComputeClient<R>,
    input: CubeTensor<R>,
    output: CubeTensor<R>,
    axis: usize,
    config: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<ReduceAutotuneKey, CubeTuneId> = local_tuner!("reduce-dim");

    let tunables = TUNER.init(|| {
        let mut set = TunableSet::new(create_key::<R>, reduce_input_gen::<R>);

        for strategy in [
            ReduceStrategy::FullUnit(RoutineStrategy::Strategy(UnitStrategy)),
            ReduceStrategy::FullCube(RoutineStrategy::Strategy(CubeStrategy { use_planes: true })),
            ReduceStrategy::FullPlane(RoutineStrategy::Strategy(PlaneStrategy {
                independant: true,
            })),
        ] {
            let name = format!("{strategy:?}");
            set = set.with(Tunable::new(
                name,
                move |(input, output, axis, config, dtypes): (
                    CubeTensor<R>,
                    CubeTensor<R>,
                    usize,
                    ReduceOperationConfig,
                    ReduceDtypes,
                )| {
                    cubek::reduce::reduce::<R>(
                        &input.client,
                        input.as_handle_ref(),
                        output.as_handle_ref(),
                        axis,
                        strategy.clone(),
                        config,
                        dtypes,
                    )
                    .map_err(|e| format!("{e}"))
                },
            ));
        }

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        client,
        tunables,
        (input, output, axis, config, dtypes),
    );
}

pub(crate) fn create_key<Run: CubeRuntime>(
    input: &CubeTensor<Run>,
    output: &CubeTensor<Run>,
    axis: &usize,
    _config: &ReduceOperationConfig,
    dtypes: &ReduceDtypes,
) -> ReduceAutotuneKey {
    let elem_input = input.dtype.into();
    let elem_output = output.dtype.into();
    let elem_acc = dtypes.accumulation.elem_type();

    ReduceAutotuneKey::generate(
        elem_input,
        elem_output,
        elem_acc,
        &input.shape.dims,
        input.strides[*axis] == 1,
        *axis,
    )
}

mod reduce_ops {
    #![allow(missing_docs)]

    use cubek::reduce::ReduceDtypes;

    use super::*;

    pub(crate) fn reduce_input_gen<Run: CubeRuntime>(
        _key: &ReduceAutotuneKey,
        input: &CubeTensor<Run>,
        output: &CubeTensor<Run>,
        dim: &usize,
        config: &ReduceOperationConfig,
        dtypes: &ReduceDtypes,
    ) -> (
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ) {
        (input.clone(), output.copy(), *dim, *config, *dtypes)
    }

    pub(crate) fn reduce<Run: CubeRuntime>(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        strategy: ReduceStrategy,
        config: ReduceOperationConfig,
        dtypes: ReduceDtypes,
    ) -> Result<(), String> {
        cubek::reduce::reduce::<Run>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            strategy,
            config,
            dtypes,
        )
        .map_err(|e| format!("{e}"))
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "autotune")]
pub fn autotune_sum<R: CubeRuntime>(
    client: &ComputeClient<R>,
    input: CubeTensor<R>,
) -> CubeTensor<R> {
    use sum_ops::*;

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!("autotune-sum");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key_sum::<R>, sum_input_gen::<R>)
            .with(Tunable::new("sum_chained", sum_chained::<R>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 1>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 2>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 4>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 8>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 16>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 32>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 64>))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        client,
        tunables,
        input,
    )
}

pub(crate) fn create_key_sum<Run: CubeRuntime>(input: &CubeTensor<Run>) -> CubeAutotuneKey {
    CubeAutotuneKey::Sum(SumAutotuneKey::generate(input))
}

impl SumAutotuneKey {
    #[allow(unused)]
    pub(crate) fn generate<Run: CubeRuntime>(input: &CubeTensor<Run>) -> Self {
        let dtype = input.dtype;
        let length = input.shape.num_elements();
        Self { dtype, length }
    }
}
mod sum_ops {
    #![allow(missing_docs)]
    use crate::ops::numeric::zeros_client;

    use super::*;

    pub(crate) fn sum_input_gen<Run: CubeRuntime>(
        _key: &CubeAutotuneKey,
        input: &CubeTensor<Run>,
    ) -> CubeTensor<Run> {
        input.clone()
    }

    pub(crate) fn sum_one_shot<Run: CubeRuntime, const C: u32>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        let client = input.client.clone();
        let device = input.device.clone();
        let output = zeros_client(client.clone(), device, [1].into(), input.dtype);

        cubek::reduce::shared_sum::<Run>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            C,
            input.dtype.into(),
        )
        .map_err(|e| e.to_string())
        .map(|_| output)
    }

    #[cfg(feature = "autotune")]
    pub(crate) fn sum_chained<Run: CubeRuntime>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        crate::kernel::reduce::reduce::<Run>(
            input,
            crate::kernel::reduce::KernelReduceStrategy::Autotune,
            cubek::reduce::components::instructions::ReduceOperationConfig::Sum,
        )
        .map_err(|e| e.to_string())
    }
}
