#![allow(missing_docs)]

use super::{
    SumAutotuneKey,
    base::{reduce_dims_copy, reduce_dims_folded},
};
use crate::{
    CubeAutotuneKey, CubeTuneId, kernel::reduce::bounds::with_sum_bounds, tensor::CubeTensor,
};
use burn_backend::{DType, cubecl::dtype_to_elem_type};
use cubecl::{
    AutotuneKey,
    client::Client,
    tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner},
};
use cubek::reduce::components::instructions::ReduceOperationConfig;
use serde::{Deserialize, Serialize};

/// Anchors shapes and strides to reuse plan choices across nearby workloads.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
struct ReduceDimsPlanAutotuneKey {
    shape: Vec<usize>,
    strides: Vec<usize>,
    dims: Vec<usize>,
    dtype: DType,
    mean: bool,
}

impl ReduceDimsPlanAutotuneKey {
    fn generate(
        shape: &[usize],
        strides: &[usize],
        dims: &[usize],
        dtype: DType,
        config: ReduceOperationConfig,
    ) -> Self {
        Self::new(
            shape
                .iter()
                .map(|&size| anchor(size, None, None, None))
                .collect(),
            strides
                .iter()
                .map(|&stride| anchor(stride, None, None, None))
                .collect(),
            dims.to_vec(),
            dtype,
            matches!(config, ReduceOperationConfig::Mean),
        )
    }
}

/// Selects a sum or mean plan. The output preserves rank and sets reduced axes to one.
pub(super) fn autotune_reduce_dims_plan(
    input: CubeTensor,
    dims: &[usize],
    config: ReduceOperationConfig,
) -> CubeTensor {
    use reduce_dims_ops::*;

    static TUNER: LocalTuner<ReduceDimsPlanAutotuneKey, CubeTuneId> =
        local_tuner!("reduce-dims-plan");

    let client = input.client.clone();
    let tune_id = CubeTuneId::new(&client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        TunableSet::new(create_key, input_gen)
            .with(Tunable::new("folded", folded))
            .with(Tunable::new("copy_single", copy_single))
    });

    let dims = (0..input.meta.num_dims())
        .filter(|dim| dims.contains(dim))
        .collect();
    TUNER.execute(&tune_id, &client, tunables, (input, dims, config))
}

mod reduce_dims_ops {
    use super::*;

    pub(super) fn create_key(
        (input, dims, config): &(CubeTensor, Vec<usize>, ReduceOperationConfig),
    ) -> ReduceDimsPlanAutotuneKey {
        ReduceDimsPlanAutotuneKey::generate(
            input.meta.shape(),
            input.meta.strides(),
            dims,
            input.dtype,
            *config,
        )
    }

    pub(super) fn input_gen(
        _key: &ReduceDimsPlanAutotuneKey,
        (input, dims, config): &(CubeTensor, Vec<usize>, ReduceOperationConfig),
    ) -> (CubeTensor, Vec<usize>, ReduceOperationConfig) {
        (input.clone(), dims.clone(), *config)
    }

    pub(super) fn folded(
        (input, dims, config): (CubeTensor, Vec<usize>, ReduceOperationConfig),
    ) -> Result<CubeTensor, String> {
        reduce_dims_folded(
            input,
            None,
            &dims,
            crate::kernel::reduce::KernelReduceStrategy::Autotune,
            config,
        )
        .map_err(|error| error.to_string())
    }

    pub(super) fn copy_single(
        (input, dims, config): (CubeTensor, Vec<usize>, ReduceOperationConfig),
    ) -> Result<CubeTensor, String> {
        reduce_dims_copy(input, &dims, config).map_err(|error| error.to_string())
    }
}

/// Executes autotune on a whole-tensor sum.
pub fn autotune_sum(client: &Client, input: CubeTensor) -> CubeTensor {
    use sum_ops::*;

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!("autotune-sum");

    let tune_id = CubeTuneId::new(&input.client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        with_sum_bounds(TunableSet::new(create_key_sum, sum_input_gen))
            .with(Tunable::new("sum_chained", sum_chained))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<1>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<2>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<4>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<8>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<16>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<32>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<64>))
    });

    TUNER.execute(&tune_id, client, tunables, input)
}

pub(crate) fn create_key_sum(input: &CubeTensor) -> CubeAutotuneKey {
    CubeAutotuneKey::Sum(SumAutotuneKey::generate(input))
}

impl SumAutotuneKey {
    #[allow(unused)]
    pub(crate) fn generate(input: &CubeTensor) -> Self {
        let dtype = input.dtype;
        let length = input.meta.num_elements();
        Self::new(dtype, length)
    }
}
mod sum_ops {
    #![allow(missing_docs)]
    use crate::ops::numeric::zeros_client;

    use super::*;

    pub(crate) fn sum_input_gen(_key: &CubeAutotuneKey, input: &CubeTensor) -> CubeTensor {
        input.clone()
    }

    pub(crate) fn sum_one_shot<const C: u32>(input: CubeTensor) -> Result<CubeTensor, String> {
        let client = input.client.clone();
        let device = input.device.clone();
        let output = zeros_client(client.clone(), device, [1].into(), input.dtype);
        let dtype = input.dtype;

        cubek::reduce::shared_sum(
            &output.client,
            input.binding(),
            output.clone().binding(),
            C,
            dtype_to_elem_type(dtype),
        )
        .map_err(|e| e.to_string())
        .map(|_| output)
    }

    #[cfg(feature = "autotune")]
    pub(crate) fn sum_chained(input: CubeTensor) -> Result<CubeTensor, String> {
        crate::kernel::reduce::reduce(
            input,
            None,
            crate::kernel::reduce::KernelReduceStrategy::Autotune,
            cubek::reduce::components::instructions::ReduceOperationConfig::Sum,
        )
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod reduce_dims_plan_tests {
    use super::*;

    #[test]
    fn nearby_lengths_share_plan_keys_unless_full_tuning_is_requested() {
        let [first, second] = [513, 514].map(|n| {
            ReduceDimsPlanAutotuneKey::generate(
                &[2, 3, n],
                &[3 * n, n, 1],
                &[0, 2],
                DType::F32,
                ReduceOperationConfig::Sum,
            )
        });

        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
        if matches!(
            CubeClRuntimeConfig::get().autotune.level,
            cubecl::config::autotune::AutotuneLevel::Full
        ) {
            assert_ne!(first, second);
        } else {
            assert_eq!(first, second);
        }
    }
}
