//! Matmul autotune over cubek's tile-DSL routines alone ([`cubek::matmul::kernels`]): what the
//! `tile-kernels-only` feature races instead of the multi-level candidates. A product none of
//! them serves fails rather than falling back, so a test run lists what they don't serve yet.

use crate::{CubeTuneId, kernel::matmul::utils::init_matmul_output, tensor::CubeTensor};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, HypercubeLastArrivalMatmulRoutine,
    MatmulRoutine, MatmulRoutineCandidate, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
    UnitWholeKMatmulRoutine,
    select::problem::OperandKind,
    tensor::PlainTensor,
    tune::key::{MatmulTuneKey, create_key},
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::{
    Hardware,
    autotune::{CandidatePriority, priority_of},
};
use std::sync::Arc;

type Inputs = (CubeTensor, CubeTensor, CubeTensor);
type MatmulTunables = TunableSet<MatmulTuneKey, Inputs, ()>;

/// Executes autotune on matmul operations, over cubek's tile-DSL routines alone.
pub fn matmul_autotune(
    lhs: CubeTensor,
    rhs: CubeTensor,
    out: Option<CubeTensor>,
    out_dtype: DType,
) -> CubeTensor {
    let output = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    if lhs.meta.shape().iter().any(|&d| d == 0) || rhs.meta.shape().iter().any(|&d| d == 0) {
        return output;
    }

    static TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("tile-kernels");

    let client = lhs.client.clone();
    let tune_client = client.clone();
    let tune_id = CubeTuneId::new(&lhs.client, &lhs.device);
    let tunables = TUNER.init(&tune_id, move || {
        let hw = Hardware::new(
            tune_client.properties_shared(),
            tune_client.target_properties(),
        );
        let routines = TuneGroup::<MatmulTuneKey>::new("tile-kernels", |_| 0);
        let set = TunableSet::new(key, |_key: &MatmulTuneKey, (lhs, rhs, out): &Inputs| {
            (lhs.clone(), rhs.clone(), out.copy())
        });
        // Not the relay hypercube routine: its cubes wait on each other.
        let set = with_routine(set, &routines, PlaneShuffleMatmulRoutine, &hw);
        let set = with_routine(set, &routines, CubeShuffleMatmulRoutine, &hw);
        let set = with_routine(set, &routines, UnitWholeKMatmulRoutine, &hw);
        let set = with_routine(set, &routines, CubeStagedMatmulRoutine, &hw);
        let set = with_routine(set, &routines, HypercubeLastArrivalMatmulRoutine, &hw);
        with_routine(set, &routines, UnitRegistersMatmulRoutine, &hw)
    });

    TUNER.execute(&tune_id, &client, tunables, (lhs, rhs, output.clone()));

    output
}

/// Every setting `routine` lists on `hw`, each a candidate of `group` placed per key by
/// [`priority_of`]: a setting compiling the launch an earlier one compiles sits out the round.
fn with_routine<R, M: MatmulMicrokernel, SM: PartialEq + Clone + 'static>(
    mut set: MatmulTunables,
    group: &TuneGroup<MatmulTuneKey>,
    routine: R,
    hw: &Hardware,
) -> MatmulTunables
where
    R: MatmulRoutine<M, SM> + Copy + Send + Sync + 'static,
    R::Settings: Send + Sync,
{
    let candidates: Arc<[MatmulRoutineCandidate<R>]> =
        MatmulRoutineCandidate::list(routine, hw).into();
    for position in 0..candidates.len() {
        let (running, ranking, hw) = (candidates.clone(), candidates.clone(), hw.clone());
        let tunable = Tunable::new(
            &candidates[position].settings().to_string(),
            move |(lhs, rhs, out): Inputs| launch(&running[position], &hw, lhs, rhs, out),
        )
        .group(group, move |key| {
            priority_of(&ranking, position, key).map_or(-1, CandidatePriority::get)
        });
        set = set.with(tunable);
    }
    set
}

/// `lhs · rhs` written into `out` by `candidate`'s routine at its setting, or why it did not run.
fn launch<R, M: MatmulMicrokernel, SM: PartialEq + Clone + 'static>(
    candidate: &MatmulRoutineCandidate<R>,
    hw: &Hardware,
    lhs: CubeTensor,
    rhs: CubeTensor,
    out: CubeTensor,
) -> Result<(), String>
where
    R: MatmulRoutine<M, SM> + Copy,
{
    if lhs.try_scheme().is_some() || rhs.try_scheme().is_some() {
        return Err("the tile-DSL routines read no quantized operand through burn".into());
    }
    let client = out.client.clone();
    let plain = |tensor: CubeTensor| {
        let elem = dtype_to_storage_type(tensor.dtype);
        PlainTensor::new(tensor.binding(), elem)
    };
    let settings = candidate.settings();
    candidate
        .routine()
        .launch_into(hw, &client, plain(lhs), plain(rhs), plain(out), settings)
        .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
}

fn key((lhs, rhs, out): &Inputs) -> MatmulTuneKey {
    let kind = |tensor: &CubeTensor| match tensor.try_scheme() {
        Some(scheme) => OperandKind::Quantized(*scheme),
        None => OperandKind::Float,
    };
    create_key(
        dtype_to_storage_type(out.dtype),
        (&lhs.meta, kind(lhs)),
        (&rhs.meta, kind(rhs)),
    )
}
