#![allow(missing_docs)]

use super::SumAutotuneKey;
use crate::{CubeAutotuneKey, CubeRuntime, CubeTuneId, tensor::CubeTensor};
use cubecl::{
    client::ComputeClient,
    reduce::{ReduceDtypes, ReduceFamily, tune_key::ReduceAutotuneKey},
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};

/// Executes autotune on reduce operations.
pub fn autotune_reduce<R: CubeRuntime, Rd: cubecl::reduce::ReduceFamily>(
    client: &ComputeClient<R::Server>,
    input: CubeTensor<R>,
    output: CubeTensor<R>,
    dim: usize,
    config: Rd::Config,
    dtypes: ReduceDtypes,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<ReduceAutotuneKey, CubeTuneId> = local_tuner!("reduce-dim");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R, Rd>, reduce_input_gen::<R, Rd>)
            .with(Tunable::new("reduce", reduce::<R, Rd>))
            .with(Tunable::new("reduce_shared", reduce_shared::<R, Rd>))
            .with(Tunable::new("reduce_plane", reduce_plane::<R, Rd>))
            .with(Tunable::new(
                "reduce_shared_plane",
                reduce_shared_plane::<R, Rd>,
            ))
    });

    TUNER.execute(
        &CubeTuneId::new::<R>(&input.client, &input.device),
        client,
        tunables,
        (input, output, dim, config, dtypes),
    );
}

pub(crate) fn create_key<Run: CubeRuntime, Rd: ReduceFamily>(
    input: &CubeTensor<Run>,
    output: &CubeTensor<Run>,
    axis: &usize,
    _config: &Rd::Config,
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

    use cubecl::reduce::{ReduceDtypes, ReduceFamily};

    use super::*;

    pub(crate) fn reduce_input_gen<Run: CubeRuntime, Rd: ReduceFamily>(
        _key: &ReduceAutotuneKey,
        input: &CubeTensor<Run>,
        output: &CubeTensor<Run>,
        dim: &usize,
        config: &Rd::Config,
        dtypes: &ReduceDtypes,
    ) -> (
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        Rd::Config,
        ReduceDtypes,
    ) {
        (input.clone(), output.copy(), *dim, *config, *dtypes)
    }

    pub(crate) fn reduce<Run: CubeRuntime, Rd: cubecl::reduce::ReduceFamily>(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
        dtypes: ReduceDtypes,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: false,
                use_planes: false,
            }),
            config,
            dtypes,
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_shared<Run: CubeRuntime, Rd: cubecl::reduce::ReduceFamily>(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
        dtypes: ReduceDtypes,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: true,
                use_planes: false,
            }),
            config,
            dtypes,
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_plane<Run: CubeRuntime, Rd: cubecl::reduce::ReduceFamily>(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
        dtypes: ReduceDtypes,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: false,
                use_planes: true,
            }),
            config,
            dtypes,
        )
        .map_err(|e| format!("{e}"))
    }

    pub(crate) fn reduce_shared_plane<Run: CubeRuntime, Rd: cubecl::reduce::ReduceFamily>(
        input: CubeTensor<Run>,
        output: CubeTensor<Run>,
        axis: usize,
        config: Rd::Config,
        dtypes: ReduceDtypes,
    ) -> Result<(), String> {
        cubecl::reduce::reduce::<Run, Rd>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            axis,
            Some(cubecl::reduce::ReduceStrategy {
                shared: true,
                use_planes: true,
            }),
            config,
            dtypes,
        )
        .map_err(|e| format!("{e}"))
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "autotune")]
pub fn autotune_sum<R: CubeRuntime>(
    client: &ComputeClient<R::Server>,
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
        &CubeTuneId::new::<R>(&input.client, &input.device),
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

        cubecl::reduce::shared_sum::<Run>(
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
            crate::kernel::reduce::ReduceStrategy::Autotune,
            cubecl::reduce::instructions::ReduceFnConfig::Sum,
        )
        .map_err(|e| e.to_string())
    }
}
