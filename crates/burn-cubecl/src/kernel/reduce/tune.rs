#![allow(missing_docs)]

use super::{
    SumAutotuneKey, accumulator_len,
    base::{reduce_dims_copy, reduce_dims_folded},
};
use crate::{
    CubeAutotuneKey, CubeTuneId,
    kernel::reduce::bounds::{
        with_reduce_bounds, with_reduce_with_indices_bounds, with_sum_bounds,
    },
    tensor::CubeTensor,
};
use burn_backend::{DType, cubecl::dtype_to_elem_type};
use cubecl::{
    AutotuneKey,
    client::Client,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, anchor, local_tuner},
};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy, ReduceWithIndicesDtypes,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy, tune_key::ReduceAutotuneKey},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};
use serde::{Deserialize, Serialize};

/// Autotune key for both dim-reduce tuners, values-only and fused values+indices.
///
/// [`ReduceAutotuneKey`] alone would put max/min and every top-k on the same entry: it
/// carries the input shape and dtypes, but not the accumulator length. Since shared memory
/// scales with that length, a routine benchmarked at one length can be infeasible at
/// another, and the tuner's cached fast path panics rather than falling back.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ReduceDimAutotuneKey {
    /// The accumulator slots per reduction (`k` for top-k, `1` otherwise).
    accumulator_len: usize,
    /// The shape and dtypes of the reduction itself.
    reduce: ReduceAutotuneKey,
}

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

/// The configured autotune level, read once — [`ReduceStrategy`] carries it
/// so the kernel blueprints know whether raw shapes are their own keys.
fn autotune_level() -> cubecl::config::autotune::AutotuneLevel {
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    static LEVEL: std::sync::OnceLock<cubecl::config::autotune::AutotuneLevel> =
        std::sync::OnceLock::new();
    LEVEL
        .get_or_init(|| CubeClRuntimeConfig::get().autotune.level.clone())
        .clone()
}

/// Registers the routine-selection tunables shared by [`autotune_reduce`] and
/// [`autotune_reduce_with_indices`]: one tunable per (routine, vectorization) pair,
/// in the same groups with the same priorities. Only the launched kernel differs,
/// injected through `launch`; keeping the grid in one place keeps the two tuners
/// from drifting apart when routines or priorities are adjusted.
fn with_routine_tunables<In, Launch>(
    mut set: TunableSet<ReduceDimAutotuneKey, In, ()>,
    group_suffix: &str,
    launch: Launch,
) -> TunableSet<ReduceDimAutotuneKey, In, ()>
where
    In: Clone + Send + Sync + 'static,
    Launch: Fn(ReduceStrategy, In) -> Result<(), String> + Clone + Send + Sync + 'static,
{
    const PRIORITY_MAX: i8 = 2;
    const PRIORITY_MIN: i8 = 1;
    const PRIORITY_SKIP: i8 = -1;

    let default_group =
        TuneGroup::<ReduceDimAutotuneKey>::new(&format!("default_{group_suffix}"), |_key| {
            PRIORITY_MAX
        });
    let vectorized_parallel_group = TuneGroup::<ReduceDimAutotuneKey>::new(
        &format!("vectorized_parallel_{group_suffix}"),
        |key| {
            if key.reduce.axis_is_contiguous {
                PRIORITY_MAX
            } else {
                // We disable the tunable with the setting [vector_size.parallel_output_vectorization]
                // when the reduce isn't parallel, since it would duplicate tunables.
                PRIORITY_SKIP
            }
        },
    );

    enum ReduceProps {
        GreatWithLowReduceCount,
        GreatWithHighReduceCount,
        Balanced,
    }

    for (vectorization, vector_size_ident) in [
        (
            VectorizationStrategy {
                parallel_output_vectorization: true,
            },
            "_vectorized_parallel_reduce",
        ),
        (
            VectorizationStrategy {
                parallel_output_vectorization: false,
            },
            "",
        ),
    ] {
        for (name, routine, props) in [
            (
                "unit",
                RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                ReduceProps::GreatWithHighReduceCount,
            ),
            (
                "plane",
                RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                    independent: true,
                })),
                ReduceProps::Balanced,
            ),
            (
                "cube",
                RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                    use_planes: true,
                })),
                ReduceProps::GreatWithLowReduceCount,
            ),
        ] {
            let name = format!("{name}{vector_size_ident}");
            let launch = launch.clone();
            let mut tunable = Tunable::new(&name, move |inputs| {
                let strategy = ReduceStrategy {
                    routine: routine.clone(),
                    vectorization,
                    // Routes the configured level into the blueprint:
                    // unchecked comptime fast paths are only stable
                    // (and only taken) when every raw shape is its
                    // own key.
                    autotune_level: autotune_level(),
                };
                launch(strategy, inputs)
            });
            if vectorization.parallel_output_vectorization {
                tunable = tunable.group(&vectorized_parallel_group, |_| PRIORITY_MAX);
            }

            tunable = tunable.group(&default_group, move |key| match props {
                ReduceProps::GreatWithLowReduceCount => {
                    if key.reduce.vector_count < 128 {
                        PRIORITY_MAX
                    } else {
                        // When you have a high level of vector to reduce, it is normally
                        // better to use another routine.
                        PRIORITY_MIN
                    }
                }
                ReduceProps::GreatWithHighReduceCount => {
                    if key.reduce.vector_count > 64 {
                        PRIORITY_MAX
                    } else {
                        // Bellow 64 it is normally better to use another routine
                        PRIORITY_MIN
                    }
                }
                ReduceProps::Balanced => PRIORITY_MAX,
            });
            set = set.with(tunable);
        }
    }

    set
}

/// Executes autotune on reduce operations.
pub fn autotune_reduce(
    client: &Client,
    input: CubeTensor,
    output: CubeTensor,
    axis: usize,
    config: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<ReduceDimAutotuneKey, CubeTuneId> = local_tuner!("reduce-dim");

    let tune_id = CubeTuneId::new(&input.client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        with_routine_tunables(
            with_reduce_bounds(TunableSet::new(create_key, reduce_input_gen)),
            "reduce",
            |strategy,
             (input, output, axis, config, dtypes): (
                CubeTensor,
                CubeTensor,
                usize,
                ReduceOperationConfig,
                ReduceDtypes,
            )| {
                cubek::reduce::reduce(
                    &output.client,
                    input.binding(),
                    output.clone().binding(),
                    axis,
                    strategy,
                    config,
                    dtypes,
                )
                .map_err(|e| format!("{e}"))
            },
        )
    });

    TUNER.execute(
        &tune_id,
        client,
        tunables,
        (input, output, axis, config, dtypes),
    );
}

pub(crate) fn create_key(
    (input, output, axis, config, dtypes): &(
        CubeTensor,
        CubeTensor,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ),
) -> ReduceDimAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(output.dtype);
    let elem_acc = dtypes.accumulation;

    ReduceDimAutotuneKey::new(
        accumulator_len(*config),
        ReduceAutotuneKey::generate(
            elem_input,
            elem_output,
            elem_acc,
            input.meta.shape(),
            input.meta.strides()[*axis] == 1,
            *axis,
        ),
    )
}

/// Executes autotune on the fused top-k that writes values and indices at once.
///
/// This needs its own tuner rather than reusing [`autotune_reduce`]: the tunables launch a
/// different kernel with an extra output, so their timings are not interchangeable with the
/// single-output ones. Without it the fused path would have to hardcode a routine, and
/// picking wrong costs more than fusing saves: the unit routine is several times slower
/// than plane/cube on a large reduce, which would make one fused launch lose to the two
/// autotuned launches it replaces.
#[allow(clippy::too_many_arguments)]
pub fn autotune_reduce_with_indices(
    client: &Client,
    input: CubeTensor,
    values: CubeTensor,
    indices: CubeTensor,
    axis: usize,
    config: ReduceOperationConfig,
    dtypes: ReduceWithIndicesDtypes,
) {
    use reduce_with_indices_ops::*;

    static TUNER: LocalTuner<ReduceDimAutotuneKey, CubeTuneId> =
        local_tuner!("reduce-dim-with-indices");

    let tune_id = CubeTuneId::new(&input.client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        with_routine_tunables(
            with_reduce_with_indices_bounds(TunableSet::new(
                create_key_with_indices,
                reduce_with_indices_input_gen,
            )),
            "reduce_with_indices",
            |strategy,
             (input, values, indices, axis, config, dtypes): (
                CubeTensor,
                CubeTensor,
                CubeTensor,
                usize,
                ReduceOperationConfig,
                ReduceWithIndicesDtypes,
            )| {
                cubek::reduce::reduce_with_indices(
                    &values.client,
                    input.binding(),
                    values.clone().binding(),
                    indices.clone().binding(),
                    axis,
                    strategy,
                    config,
                    dtypes,
                )
                .map_err(|e| format!("{e}"))
            },
        )
    });

    TUNER.execute(
        &tune_id,
        client,
        tunables,
        (input, values, indices, axis, config, dtypes),
    );
}

pub(crate) fn create_key_with_indices(
    (input, values, _indices, axis, config, dtypes): &(
        CubeTensor,
        CubeTensor,
        CubeTensor,
        usize,
        ReduceOperationConfig,
        ReduceWithIndicesDtypes,
    ),
) -> ReduceDimAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(values.dtype);
    let elem_acc = dtypes.accumulation;

    ReduceDimAutotuneKey::new(
        accumulator_len(*config),
        ReduceAutotuneKey::generate(
            elem_input,
            elem_output,
            elem_acc,
            input.meta.shape(),
            input.meta.strides()[*axis] == 1,
            *axis,
        ),
    )
}

mod reduce_with_indices_ops {
    #![allow(missing_docs)]

    use super::*;

    pub(crate) fn reduce_with_indices_input_gen(
        _key: &ReduceDimAutotuneKey,
        (input, values, indices, dim, config, dtypes): &(
            CubeTensor,
            CubeTensor,
            CubeTensor,
            usize,
            ReduceOperationConfig,
            ReduceWithIndicesDtypes,
        ),
    ) -> (
        CubeTensor,
        CubeTensor,
        CubeTensor,
        usize,
        ReduceOperationConfig,
        ReduceWithIndicesDtypes,
    ) {
        (
            input.clone(),
            values.copy(),
            indices.copy(),
            *dim,
            *config,
            *dtypes,
        )
    }
}

mod reduce_ops {
    #![allow(missing_docs)]

    use cubek::reduce::ReduceDtypes;

    use super::*;

    pub(crate) fn reduce_input_gen(
        _key: &ReduceDimAutotuneKey,
        (input, output, dim, config, dtypes): &(
            CubeTensor,
            CubeTensor,
            usize,
            ReduceOperationConfig,
            ReduceDtypes,
        ),
    ) -> (
        CubeTensor,
        CubeTensor,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ) {
        (input.clone(), output.copy(), *dim, *config, *dtypes)
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "autotune")]
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

        if matches!(
            autotune_level(),
            cubecl::config::autotune::AutotuneLevel::Full
        ) {
            assert_ne!(first, second);
        } else {
            assert_eq!(first, second);
        }
    }
}
