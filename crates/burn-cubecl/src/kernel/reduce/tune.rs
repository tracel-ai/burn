#![allow(missing_docs)]

use super::SumAutotuneKey;
use crate::{CubeAutotuneKey, CubeRuntime, CubeTuneId, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use cubecl::{
    client::ComputeClient,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy, ReduceWithIndicesDtypes,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy, tune_key::ReduceAutotuneKey},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

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
    mut set: TunableSet<ReduceAutotuneKey, In, ()>,
    group_suffix: &str,
    launch: Launch,
) -> TunableSet<ReduceAutotuneKey, In, ()>
where
    In: Clone + Send + Sync + 'static,
    Launch: Fn(ReduceStrategy, In) -> Result<(), String> + Clone + Send + Sync + 'static,
{
    const PRIORITY_MAX: i8 = 2;
    const PRIORITY_MIN: i8 = 1;
    const PRIORITY_SKIP: i8 = -1;

    let default_group =
        TuneGroup::<ReduceAutotuneKey>::new(&format!("default_{group_suffix}"), |_key| {
            PRIORITY_MAX
        });
    let vectorized_parallel_group = TuneGroup::<ReduceAutotuneKey>::new(
        &format!("vectorized_parallel_{group_suffix}"),
        |key| {
            if key.axis_is_contiguous {
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
                    if key.vector_count < 128 {
                        PRIORITY_MAX
                    } else {
                        // When you have a high level of vector to reduce, it is normally
                        // better to use another routine.
                        PRIORITY_MIN
                    }
                }
                ReduceProps::GreatWithHighReduceCount => {
                    if key.vector_count > 64 {
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
        with_routine_tunables(
            TunableSet::new(create_key::<R>, reduce_input_gen::<R>),
            "reduce",
            |strategy,
             (input, output, axis, config, dtypes): (
                CubeTensor<R>,
                CubeTensor<R>,
                usize,
                ReduceOperationConfig,
                ReduceDtypes,
            )| {
                cubek::reduce::reduce::<R>(
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
        &CubeTuneId::new(&input.client, &input.device),
        client,
        tunables,
        (input, output, axis, config, dtypes),
    );
}

pub(crate) fn create_key<Run: CubeRuntime>(
    (input, output, axis, _config, dtypes): &(
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ),
) -> ReduceAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(output.dtype);
    let elem_acc = dtypes.accumulation.elem_type();

    ReduceAutotuneKey::generate(
        elem_input,
        elem_output,
        elem_acc,
        input.meta.shape(),
        input.meta.strides()[*axis] == 1,
        *axis,
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
pub fn autotune_reduce_with_indices<R: CubeRuntime>(
    client: &ComputeClient<R>,
    input: CubeTensor<R>,
    values: CubeTensor<R>,
    indices: CubeTensor<R>,
    axis: usize,
    k: usize,
    dtypes: ReduceWithIndicesDtypes,
) {
    use reduce_with_indices_ops::*;

    static TUNER: LocalTuner<ReduceAutotuneKey, CubeTuneId> =
        local_tuner!("reduce-dim-with-indices");

    let tunables = TUNER.init(|| {
        with_routine_tunables(
            TunableSet::new(
                create_key_with_indices::<R>,
                reduce_with_indices_input_gen::<R>,
            ),
            "reduce_with_indices",
            |strategy,
             (input, values, indices, axis, k, dtypes): (
                CubeTensor<R>,
                CubeTensor<R>,
                CubeTensor<R>,
                usize,
                usize,
                ReduceWithIndicesDtypes,
            )| {
                cubek::reduce::reduce_with_indices::<R>(
                    &values.client,
                    input.binding(),
                    values.clone().binding(),
                    indices.clone().binding(),
                    axis,
                    strategy,
                    ReduceOperationConfig::TopK(k),
                    dtypes,
                )
                .map_err(|e| format!("{e}"))
            },
        )
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        client,
        tunables,
        (input, values, indices, axis, k, dtypes),
    );
}

pub(crate) fn create_key_with_indices<Run: CubeRuntime>(
    (input, values, _indices, axis, _k, dtypes): &(
        CubeTensor<Run>,
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        usize,
        ReduceWithIndicesDtypes,
    ),
) -> ReduceAutotuneKey {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(values.dtype);
    let elem_acc = dtypes.accumulation.elem_type();

    ReduceAutotuneKey::generate(
        elem_input,
        elem_output,
        elem_acc,
        input.meta.shape(),
        input.meta.strides()[*axis] == 1,
        *axis,
    )
}

mod reduce_with_indices_ops {
    #![allow(missing_docs)]

    use super::*;

    pub(crate) fn reduce_with_indices_input_gen<Run: CubeRuntime>(
        _key: &ReduceAutotuneKey,
        (input, values, indices, dim, k, dtypes): &(
            CubeTensor<Run>,
            CubeTensor<Run>,
            CubeTensor<Run>,
            usize,
            usize,
            ReduceWithIndicesDtypes,
        ),
    ) -> (
        CubeTensor<Run>,
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        usize,
        ReduceWithIndicesDtypes,
    ) {
        (
            input.clone(),
            values.copy(),
            indices.copy(),
            *dim,
            *k,
            *dtypes,
        )
    }
}

mod reduce_ops {
    #![allow(missing_docs)]

    use cubek::reduce::ReduceDtypes;

    use super::*;

    pub(crate) fn reduce_input_gen<Run: CubeRuntime>(
        _key: &ReduceAutotuneKey,
        (input, output, dim, config, dtypes): &(
            CubeTensor<Run>,
            CubeTensor<Run>,
            usize,
            ReduceOperationConfig,
            ReduceDtypes,
        ),
    ) -> (
        CubeTensor<Run>,
        CubeTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ) {
        (input.clone(), output.copy(), *dim, *config, *dtypes)
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
        let length = input.meta.num_elements();
        Self::new(dtype, length)
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
        let dtype = input.dtype;

        cubek::reduce::shared_sum::<Run>(
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
    pub(crate) fn sum_chained<Run: CubeRuntime>(
        input: CubeTensor<Run>,
    ) -> Result<CubeTensor<Run>, String> {
        crate::kernel::reduce::reduce::<Run>(
            input,
            None,
            crate::kernel::reduce::KernelReduceStrategy::Autotune,
            cubek::reduce::components::instructions::ReduceOperationConfig::Sum,
        )
        .map_err(|e| e.to_string())
    }
}
