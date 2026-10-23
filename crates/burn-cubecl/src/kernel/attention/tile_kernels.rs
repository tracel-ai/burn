//! Attention autotune over cubek's tile-DSL routines alone ([`cubek::attention::kernels`]): what
//! the `tile-kernels-only` feature races instead of the flash routines and the fallback. An
//! attention none of them serves fails rather than falling back, so a test run lists what they
//! don't serve yet.

use crate::{
    CubeTuneId,
    ops::{numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::AttentionModuleOptions;
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use cubecl::prelude::InputScalar;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::attention::kernels::{
    AttentionBlueprint, AttentionRoutine, AttentionRoutineCandidate, AttentionScoring,
    AttentionTensors, AttentionVisibility, CubeShuffleAttentionRoutine, CubeStagedAttentionRoutine,
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeSecondLaunchFragmentAttentionRoutine,
    HypercubeSecondLaunchShuffleAttentionRoutine, ScoreScale, UnitAttentionRoutine,
    tune::key::AttentionKey,
};
use cubek::matmul::kernels::tensor::PlainTensor;
use cubek::microkernel::attention::AttentionMicrokernel;
use cubek::microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneUnitGroupAttentionMicrokernel,
};
use cubek::std::{
    Hardware,
    autotune::{CandidatePriority, TuneCandidate, priority_of},
    routine::Routine,
};
use std::sync::Arc;

type Inputs = (
    CubeTensor,
    CubeTensor,
    CubeTensor,
    Option<CubeTensor>,
    Option<CubeTensor>,
    AttentionModuleOptions,
);
type AttentionTunables = TunableSet<AttentionKey, Inputs, CubeTensor>;

/// Executes autotune on attention operations, over cubek's tile-DSL routines alone.
pub fn attention_autotune(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    attn_bias: Option<CubeTensor>,
    options: AttentionModuleOptions,
) -> CubeTensor {
    static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("tile-kernels-attention");

    let client = query.client.clone();
    let tune_client = client.clone();
    let tune_id = CubeTuneId::new(&query.client, &query.device);
    let tunables = TUNER.init(&tune_id, move || {
        let hw = Hardware::new(
            tune_client.properties_shared(),
            tune_client.target_properties(),
        );
        let routines = TuneGroup::<AttentionKey>::new("tile-kernels", |_| 1);
        // Under the routines, the unit routine's lead, which asks nothing of the hardware: what
        // runs where none of them serves the problem.
        let unit_lead = TuneGroup::<AttentionKey>::new("unit-lead", |_| 0);
        let set = TunableSet::new(attention_key, |_key: &AttentionKey, inputs: &Inputs| {
            inputs.clone()
        });
        let set = with_routine(
            set,
            &routines,
            CubeStagedAttentionRoutine::<PlaneFragmentAttentionMicrokernel>::new(),
            &hw,
        );
        let set = with_routine(
            set,
            &routines,
            CubeStagedAttentionRoutine::<PlaneUnitGroupAttentionMicrokernel>::new(),
            &hw,
        );
        let set = with_routine(set, &routines, CubeShuffleAttentionRoutine, &hw);
        let set = with_routine(
            set,
            &routines,
            HypercubeLastArrivalShuffleAttentionRoutine::new(),
            &hw,
        );
        let set = with_routine(
            set,
            &routines,
            HypercubeSecondLaunchShuffleAttentionRoutine::new(),
            &hw,
        );
        let set = with_routine(
            set,
            &routines,
            HypercubeSecondLaunchFragmentAttentionRoutine,
            &hw,
        );
        let mut unit = AttentionRoutineCandidate::list(UnitAttentionRoutine, &hw);
        unit.truncate(1);
        with_candidates(set, &unit_lead, unit)
    });

    TUNER.execute(
        &tune_id,
        &client,
        tunables,
        (query, key, value, mask, attn_bias, options),
    )
}

/// Every setting `routine` lists on `hw`, each a candidate of `group`.
fn with_routine<R, M: AttentionMicrokernel, SM: PartialEq + 'static>(
    set: AttentionTunables,
    group: &TuneGroup<AttentionKey>,
    routine: R,
    hw: &Hardware,
) -> AttentionTunables
where
    R: AttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
        + Copy
        + Send
        + Sync
        + 'static,
    R::Settings: Send + Sync,
{
    with_candidates(set, group, AttentionRoutineCandidate::list(routine, hw))
}

/// `candidates`, each in `group`, placed per key by [`priority_of`]: a setting compiling the
/// launch an earlier one compiles sits out the round.
fn with_candidates<R, M: AttentionMicrokernel, SM: PartialEq + 'static>(
    mut set: AttentionTunables,
    group: &TuneGroup<AttentionKey>,
    candidates: Vec<AttentionRoutineCandidate<R>>,
) -> AttentionTunables
where
    R: AttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
        + Copy
        + Send
        + Sync
        + 'static,
    R::Settings: Send + Sync,
{
    let candidates: Arc<[AttentionRoutineCandidate<R>]> = candidates.into();
    for position in 0..candidates.len() {
        let (running, ranking) = (candidates.clone(), candidates.clone());
        let tunable = Tunable::new(
            &candidates[position].strategy().to_string(),
            move |inputs: Inputs| launch(&running[position], inputs),
        )
        .group(group, move |key| {
            priority_of(&ranking, position, key).map_or(-1, CandidatePriority::get)
        });
        set = set.with(tunable);
    }
    set
}

/// The attention of `inputs` by `candidate`'s routine at its setting, `[b, h, q, dv]`, or why it
/// did not run.
fn launch<R, M: AttentionMicrokernel, SM: PartialEq>(
    candidate: &AttentionRoutineCandidate<R>,
    (query, key, value, mask, attn_bias, options): Inputs,
) -> Result<CubeTensor, String>
where
    R: AttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
        + Copy,
{
    if mask.is_some() {
        return Err("the tile-DSL attention takes no mask tensor".into());
    }
    if attn_bias.is_some() || options.softcap.is_some() {
        return Err("the tile-DSL attention takes no additive bias or softcap".into());
    }
    let (client, device) = (query.client.clone(), query.device.clone());
    let hw = Hardware::new(client.properties_shared(), client.target_properties());
    let [batch, seq_kv] = [key.meta.shape()[0], key.meta.shape()[2]];
    let lengths = full_device_dtype(
        client.clone(),
        Shape::new([batch]),
        device.clone(),
        InputScalar::new(seq_kv as u32, dtype_to_storage_type(DType::U32)),
        DType::U32,
    );
    let scoring = scoring(&query, &options);
    let dtype = query.dtype;
    let plain = |tensor: CubeTensor| {
        let elem = dtype_to_storage_type(tensor.dtype);
        PlainTensor::new(tensor.binding(), elem)
    };
    let settings = candidate.strategy();
    let out = AttentionTensors::new(
        &client,
        plain(query),
        plain(key),
        plain(value),
        plain(lengths),
        scoring,
    )
    .and_then(|tensors| candidate.routine().launch_on(&hw, tensors, settings))
    .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))?;
    // The routines write `[b, q, h, dv]`, a token's heads adjacent; burn's attention is
    // `[b, h, q, dv]`, viewed without a copy.
    let out = CubeTensor::new(client, out.handle, *out.metadata, device, dtype);
    Ok(swap_dims(out, 1, 2))
}

/// Which keys a query sees and the scale its scores take, as `options` state them.
fn scoring(query: &CubeTensor, options: &AttentionModuleOptions) -> AttentionScoring {
    let head_dim = query.meta.shape()[3];
    AttentionScoring {
        visibility: AttentionVisibility::new(options.is_causal, None),
        scale: match options.scale {
            Some(scale) => ScoreScale::Custom(scale as f32),
            None => ScoreScale::standard(head_dim),
        },
    }
}

fn attention_key((query, key, _value, _mask, _attn_bias, options): &Inputs) -> AttentionKey {
    let shape = query.meta.shape();
    let scoring = scoring(query, options);
    AttentionKey::new(
        dtype_to_storage_type(query.dtype),
        shape[1],
        key.meta.shape()[1],
        shape[3],
        shape[2],
        key.meta.shape()[2],
        shape[0],
        scoring.visibility,
        scoring.scale,
        None,
    )
}
