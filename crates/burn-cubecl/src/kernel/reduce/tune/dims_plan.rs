#![allow(missing_docs)]

use crate::kernel::reduce::{reduce_dims_copy, reduce_dims_folded};
use crate::{CubeTuneId, tensor::CubeTensor};
use burn_backend::DType;
use burn_backend::cubecl::autotune::{RACED_PRIORITY, raced_group};
use cubecl::{
    AutotuneKeyable,
    tune::{Anchor, LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::reduce::ReduceOperation;
use serde::{Deserialize, Serialize};

/// Anchors shapes and strides to reuse plan choices across nearby workloads.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKeyable)]
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
        config: ReduceOperation,
    ) -> Self {
        Self::new(
            shape
                .iter()
                .map(|&size| Anchor::default().round(size))
                .collect(),
            strides
                .iter()
                .map(|&stride| Anchor::default().round(stride))
                .collect(),
            dims.to_vec(),
            dtype,
            matches!(config, ReduceOperation::Mean),
        )
    }
}

/// Selects a sum or mean plan. The output preserves rank and sets reduced axes to one.
pub(crate) fn autotune_reduce_dims_plan(
    input: CubeTensor,
    dims: &[usize],
    config: ReduceOperation,
) -> CubeTensor {
    use reduce_dims_ops::*;

    static TUNER: LocalTuner<ReduceDimsPlanAutotuneKey, CubeTuneId> =
        local_tuner!("reduce-dims-plan");

    let client = input.client.clone();
    let tune_id = CubeTuneId::new(&client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        let raced = raced_group("kernels");
        TunableSet::new(create_key, input_gen)
            .with(Tunable::new("folded", folded).group(&raced, |_| RACED_PRIORITY))
            .with(Tunable::new("copy_single", copy_single).group(&raced, |_| RACED_PRIORITY))
    });

    let dims = (0..input.meta.num_dims())
        .filter(|dim| dims.contains(dim))
        .collect();
    TUNER.execute(&tune_id, &client, tunables, (input, dims, config))
}

mod reduce_dims_ops {
    use super::*;

    pub(super) fn create_key(
        (input, dims, config): &(CubeTensor, Vec<usize>, ReduceOperation),
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
        (input, dims, config): &(CubeTensor, Vec<usize>, ReduceOperation),
    ) -> (CubeTensor, Vec<usize>, ReduceOperation) {
        (input.clone(), dims.clone(), *config)
    }

    pub(super) fn folded(
        (input, dims, config): (CubeTensor, Vec<usize>, ReduceOperation),
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
        (input, dims, config): (CubeTensor, Vec<usize>, ReduceOperation),
    ) -> Result<CubeTensor, String> {
        reduce_dims_copy(input, &dims, config).map_err(|error| error.to_string())
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
                ReduceOperation::Sum,
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
