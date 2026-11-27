//! Choosing an attention's routine by measurement, in one of two autotunes: a call of many query
//! positions, and a call of one.
//!
//! The routines serve GQA natively, each key head read once per box. They attend the first
//! `len` positions of a cache with the queries bottom-right aligned, so a call is raced at the
//! depth its caller states ([`AttentionKey`]), never at how many positions a cache holds.

use crate::tune::{
    Arithmetic, Autotune, CandidatePriority, Ceiling, GroupPriority, Predictor, Roofline,
    TuneCandidate,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::throughput::MemoryAccess;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::attention::kernels::plan::{AttentionCall, AttentionPlan};
use cubek::attention::kernels::select::problem::QueryPositions;
use cubek::attention::kernels::{
    AttentionBlueprint, AttentionKey, AttentionLaunch, AttentionRoutine, AttentionRoutineCandidate,
    AttentionRoutinePrepared, AttentionRoutineProblem, AttentionTuneInputs,
    CubeShuffleAttentionRoutine, CubeStagedAttentionRoutine, DirectAttention,
    HypercubeFragmentPartialAttentionRoutine, HypercubeLastArrivalShuffleAttentionRoutine,
    HypercubeShufflePartialAttentionRoutine, HypercubeStreamFragmentAttentionRoutine,
    PartialAttentionRoutine, UnitAttentionRoutine,
};
use cubek::microkernel::attention::AttentionMicrokernel;
use cubek::std::Hardware;
use cubek::std::plan::Fastest;
use cubek::std::routine::Routine;

/// The priority each autotune's routines ship at: a caller's to set, as in the matmul family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionAutotuneGroups {
    /// The many-positions autotune's.
    pub prefill: GroupPriority,
    /// The one-position autotune's.
    pub decode: GroupPriority,
}

impl AttentionAutotuneGroups {
    /// The priority of the autotune of calls of `positions` query positions.
    pub fn of(self, positions: QueryPositions) -> GroupPriority {
        match positions {
            QueryPositions::One => self.decode,
            QueryPositions::Many => self.prefill,
        }
    }

    /// The autotune of calls of `positions` query positions as a plan, its routines' group at the
    /// priority [`of`](Self::of) gives `positions`: the decode autotune at one position, the
    /// prefill one at many. What a caller of cubek's
    /// [`select`](cubek::attention::kernels::select::selector::select) hands it as the measured
    /// choice. It measures a float call; a packed cache is rebuilt before it.
    pub fn fastest(self, positions: QueryPositions) -> AttentionPlan {
        let routines = self.of(positions);
        let autotune = match positions {
            QueryPositions::One => "decode",
            QueryPositions::Many => "prefill",
        };
        Box::new(Fastest::new(
            autotune,
            move |call: &AttentionCall| match call {
                AttentionCall::Float(float) => {
                    launch_tuned(positions, &float.device, float.inputs.clone(), routines)
                }
                AttentionCall::Packed(_) => {
                    unreachable!(
                        "an autotune is planned over a float call, the packed cache rebuilt first"
                    )
                }
            },
        ))
    }
}

impl Default for AttentionAutotuneGroups {
    fn default() -> Self {
        Self {
            prefill: GroupPriority::DEFAULT,
            decode: GroupPriority::DEFAULT,
        }
    }
}

/// The attention of `query` over `key` and `value`, `[b, q, h, dv]`, through whichever routine
/// and setting measures fastest for this problem on `device`: among the routines that serve many
/// query positions, or those that serve one, as `positions` says. `routines` is the priority of
/// the routines' group; under it sits the unit routine's lead, which serves any problem on any
/// device.
pub(crate) fn launch_tuned(
    positions: QueryPositions,
    device: &Device,
    inputs: AttentionTuneInputs,
    routines: GroupPriority,
) -> TensorHandle {
    let client = inputs.client().clone();
    let id = CubeTuneId::new(&client, device);
    let predictor = Predictor::on_device(&client);
    match positions {
        QueryPositions::Many => {
            // Named rather than `module_path!()`, so the cached winners stay this autotune's
            // wherever the code lives; a cached winner is an index into the candidate list.
            static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("prefill");
            let tunables = TUNER.init(&id, move || many_queries(&predictor, routines));
            TUNER.execute(&id, &client, tunables, inputs.clone());
        }
        QueryPositions::One => {
            static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("decode");
            let tunables = TUNER.init(&id, move || one_query(&predictor, routines));
            TUNER.execute(&id, &client, tunables, inputs.clone());
        }
    }
    inputs.tensors().clone().into_out()
}

type AttentionAutotune = Autotune<AttentionKey, AttentionTuneInputs, ()>;

/// The routines that serve many query positions: the cube routines, and the hypercube routines for
/// a prompt's or a chunk's few boxes over many keys, a partial one merged by a second launch. The
/// unit routine is listed only where none of them can run, for want of a plane width the device
/// commits to: elsewhere measuring its candidates takes longer than the rest of the autotune. The
/// plane routine is not listed.
///
/// Its candidates are timed as the cache leaves them, unlike one query's: a prompt's queries, keys
/// and values are the activations the layer before it just wrote, so production reads them warm,
/// and a round timing them cold ranks the candidates on a read they never meet (an f32 head dim
/// of 128 on an L4: a winner 14% behind the multi-level kernels warm, against 28% chosen cold).
fn many_queries(
    predictor: &Predictor,
    priority: GroupPriority,
) -> TunableSet<AttentionKey, AttentionTuneInputs, ()> {
    let mut autotune = AttentionAutotune::new(AttentionTuneInputs::many_queries_key);
    let routines = autotune.group("microkernel", priority);
    autotune = autotune
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(CubeStagedAttentionRoutine, predictor),
            AttentionTuneCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(CubeShuffleAttentionRoutine, predictor),
            AttentionTuneCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(
                HypercubeLastArrivalShuffleAttentionRoutine::new(),
                predictor,
            ),
            AttentionTuneCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(HypercubeShufflePartialAttentionRoutine::new(), predictor),
            AttentionTuneCandidate::launch_split_keys,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(HypercubeFragmentPartialAttentionRoutine, predictor),
            AttentionTuneCandidate::launch_split_keys,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(HypercubeStreamFragmentAttentionRoutine, predictor),
            AttentionTuneCandidate::launch,
        );
    if predictor.hardware().plane_dim().is_err() {
        autotune = autotune.prioritized(
            &routines,
            AttentionTuneCandidate::list(UnitAttentionRoutine, predictor),
            AttentionTuneCandidate::launch,
        );
    }
    autotune.floor(unit_lead(predictor.hardware())).finish()
}

/// The routines that serve a single position, reading its group of heads as rows. The
/// hypercube's cubes of plane shuffles share each key head's cache, merged on the last arrival or
/// by a second launch after the partial routine; the cube's plane shuffles split a cube's keys. No
/// routine whose cubes stage is listed: a cube staging a step for a decode step's few rows measured
/// twelve times the hypercube's time. As for many positions, the unit routine is listed only where
/// neither can run, and the plane routine not at all.
fn one_query(
    predictor: &Predictor,
    priority: GroupPriority,
) -> TunableSet<AttentionKey, AttentionTuneInputs, ()> {
    let mut autotune = AttentionAutotune::new(AttentionTuneInputs::one_query_key);
    let routines = autotune.group("microkernel", priority);
    autotune = autotune
        .read_cold(one_query_roofline)
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(
                HypercubeLastArrivalShuffleAttentionRoutine::new(),
                predictor,
            ),
            AttentionTuneCandidate::launch,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(HypercubeShufflePartialAttentionRoutine::new(), predictor),
            AttentionTuneCandidate::launch_split_keys,
        )
        .prioritized(
            &routines,
            AttentionTuneCandidate::list(CubeShuffleAttentionRoutine, predictor),
            AttentionTuneCandidate::launch,
        );
    if predictor.hardware().plane_dim().is_err() {
        autotune = autotune.prioritized(
            &routines,
            AttentionTuneCandidate::list(UnitAttentionRoutine, predictor),
            AttentionTuneCandidate::launch,
        );
    }
    autotune.floor(unit_lead(predictor.hardware())).finish()
}

/// The one-position roofline: the bus, in the direction the traffic goes. One query against the
/// whole cache reuses nothing — every key and value byte is read once and the write is a single
/// row — so the ceiling is the **read** bandwidth.
fn one_query_roofline(_key: &AttentionKey, inputs: &AttentionTuneInputs) -> Roofline {
    Roofline {
        client: inputs.client().clone(),
        ceiling: Ceiling::Memory(MemoryAccess::Read),
        work: inputs.one_query_work(),
    }
}

/// Both autotunes' floor: the unit routine's lead, which asks nothing of the hardware. Nobody
/// wants it to win, but a plan cubecl cannot fill is a panic, and a configuration may take every
/// group out of the autotune.
fn unit_lead(hw: &Hardware) -> Tunable<AttentionKey, AttentionTuneInputs, ()> {
    let hw = hw.clone();
    Tunable::new("unit-attention-lead", move |inputs: AttentionTuneInputs| {
        DirectAttention(UnitAttentionRoutine)
            .launch_lead_on(&hw, inputs.tensors().clone())
            .map(|_| ())
            .map_err(|why| format!("the unit routine's lead does not serve this problem: {why:?}"))
    })
}

/// One routine at one of its settings, as an attention autotune races it, on the device it was
/// listed for.
///
/// Its priority is its routine's estimate timed at the device's peaks, against the attention's
/// roofline ([`Predictor::predict_priority`]).
#[derive(Debug, Clone)]
struct AttentionTuneCandidate<R: Routine> {
    candidate: AttentionRoutineCandidate<R>,
    predictor: Predictor,
}

impl<R: Routine + Copy> AttentionTuneCandidate<R> {
    /// Every setting `routine` lists on `predictor`'s device, its default first.
    fn list(routine: R, predictor: &Predictor) -> Vec<Self> {
        AttentionRoutineCandidate::list(routine, predictor.hardware())
            .into_iter()
            .map(|candidate| Self {
                candidate,
                predictor: predictor.clone(),
            })
            .collect()
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionTuneCandidate<R>
where
    R: AttentionRoutine<M, SM> + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>,
{
    /// The attention `inputs` state, written by the routine at its setting
    /// ([`AttentionRoutineCandidate::launch`]).
    fn launch(&self, inputs: AttentionTuneInputs) -> Result<(), String> {
        self.candidate.launch(inputs)
    }
}

impl<R, M: AttentionMicrokernel, SM> AttentionTuneCandidate<R>
where
    R: PartialAttentionRoutine<M, SM>
        + Routine<Microkernel = M, Blueprint = AttentionBlueprint<M, SM>>
        + Copy,
{
    /// The attention `inputs` state, written by the partial routine at its setting and merged
    /// ([`AttentionRoutineCandidate::launch_split_keys`]).
    fn launch_split_keys(&self, inputs: AttentionTuneInputs) -> Result<(), String> {
        self.candidate.launch_split_keys(inputs)
    }
}

impl<R, M: AttentionMicrokernel, SM: PartialEq> TuneCandidate for AttentionTuneCandidate<R>
where
    R: Routine<
            Problem = AttentionRoutineProblem,
            Microkernel = M,
            Blueprint = AttentionBlueprint<M, SM>,
        >,
{
    type Strategy = R::Settings;
    type Key = AttentionKey;
    type Blueprint = AttentionRoutinePrepared<M, SM>;

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// Its estimate timed at the device's peaks, its products on the largest matrix instruction
    /// the device offers the query's element; last where the key states no problem a routine
    /// serves.
    fn priority(&self, key: &AttentionKey) -> CandidatePriority {
        let Ok(problem) = AttentionRoutineProblem::try_from(key) else {
            return CandidatePriority::LAST;
        };
        let arithmetic = Arithmetic::product(self.predictor.hardware(), problem.elements.query);
        self.predictor
            .predict_priority::<R>(&problem, self.candidate.settings(), arithmetic)
    }

    fn blueprint(&self, key: &AttentionKey) -> Option<AttentionRoutinePrepared<M, SM>> {
        self.candidate.prepared(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::features::MmaConfig;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek::attention::kernels::{
        AttentionKeyElements, AttentionVisibility, QueryPlacement, ScoreScale,
    };
    use cubek::std::devices;
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    use cubek::std::routine::EstimatedProblem;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// `hw` offering the square `16×16×16` instruction for `f16` and for `f32`, summed in `f32`.
    fn offering(hw: Hardware) -> Hardware {
        let square = |element| MmaConfig {
            a_type: element,
            b_type: element,
            cd_type: F32,
            m: 16,
            n: 16,
            k: 16,
        };
        devices::offering_cmma(hw, [square(F16), square(F32)])
    }

    /// A causal `f16` prompt of `seq` positions, `heads` query heads over `kv_heads` key heads of
    /// dim 128, one batch.
    fn key(heads: usize, kv_heads: usize, seq: usize) -> AttentionKey {
        AttentionKey::new(
            AttentionKeyElements::one(F16),
            heads,
            kv_heads,
            128,
            seq,
            seq,
            1,
            AttentionVisibility::Causal {
                queries: QueryPlacement::Last,
            },
            ScoreScale::standard(128),
            None,
            false,
            false,
        )
    }

    /// A key whose query heads do not split over its key heads is declined by every candidate:
    /// tried after every candidate that runs, and claiming no launch.
    #[test]
    fn an_uneven_group_is_tried_last_and_claims_no_launch() {
        let uneven = key(6, 4, 2048);
        for candidate in AttentionTuneCandidate::list(
            UnitAttentionRoutine,
            &Predictor::described(devices::m2_pro()),
        ) {
            let settings = candidate.strategy();
            assert_eq!(
                candidate.priority(&uneven),
                CandidatePriority::LAST,
                "{settings}"
            );
            assert!(candidate.blueprint(&uneven).is_none(), "{settings}");
        }
    }

    /// A setting that prepares on the device is placed by its prediction, which never puts it
    /// with the declines: a prompt's and a decode step's keys alike.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn a_setting_that_prepares_is_placed_above_the_declines() {
        let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
        let lead = AttentionTuneCandidate::list(UnitAttentionRoutine, &predictor).swap_remove(0);
        for key in [key(32, 8, 1), key(32, 8, 4096)] {
            assert!(lead.blueprint(&key).is_some());
            assert!(lead.priority(&key) > CandidatePriority::LAST);
        }
    }

    /// On devices a dev machine cannot show, a cube routine's setting is tried last exactly where
    /// it claims no launch.
    #[test]
    fn a_decline_is_tried_last_and_claims_nothing() {
        let key = key(32, 8, 2048);
        for hw in [
            offering(devices::rdna_range()),
            offering(devices::cdna_wave64()),
            offering(devices::h100()),
            devices::wgpu_no_subgroups(),
            devices::cpu_12(),
        ] {
            for cube in
                AttentionTuneCandidate::list(CubeStagedAttentionRoutine, &Predictor::described(hw))
            {
                let declined = cube.blueprint(&key).is_none();
                let last = cube.priority(&key) == CandidatePriority::LAST;
                assert_eq!(declined, last, "{}", cube.strategy());
            }
        }
    }

    /// **A race runs on the call's buffers.** A prompt's and a decode step's autotunes hand their
    /// candidates the call's own operands, and the output every candidate and the winner write is
    /// the call's.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn a_race_runs_on_the_calls_buffers() {
        use crate::CubeDevice;
        use cubecl::server::Overlap;
        use cubecl::zspace::striding::row_major_contiguous_strides;
        use cubek::attention::kernels::AttentionScoring;
        use cubek::matmul::kernels::tensor::PlainTensor;
        let client = CubeDevice::default().client();
        let (heads, kv_heads, head_dim) = (8, 2, 64);
        let zeros = |shape: [usize; 4]| {
            PlainTensor::zeros(&client, &shape, row_major_contiguous_strides(shape), F32)
        };
        let predictor = Predictor::on_device(&client);
        for queries in [1, 37] {
            let cache = [1, kv_heads, 256, head_dim];
            let visibility = AttentionVisibility::new(queries > 1, None, QueryPlacement::Last);
            let call = AttentionTuneInputs::new(
                &client,
                zeros([1, heads, queries, head_dim]),
                zeros(cache),
                zeros(cache),
                None,
                PlainTensor::of_u32s(&client, &[1], &[queries as u32]),
                queries,
                None,
                AttentionScoring::standard(visibility, head_dim),
            )
            .expect("an even group of a float element");
            let raced = match queries {
                1 => one_query(&predictor, GroupPriority::DEFAULT)
                    .generate_inputs(&call.one_query_key(), &call),
                _ => many_queries(&predictor, GroupPriority::DEFAULT)
                    .generate_inputs(&call.many_queries_key(), &call),
            };
            let out = |inputs: &AttentionTuneInputs| inputs.tensors().clone().into_out().handle;
            assert!(
                matches!(out(&raced).overlap(&out(&call)), Overlap::Same),
                "{queries} queries"
            );
        }
    }

    /// Not a check but a measurement, run by hand: every distinct prefill candidate's priority,
    /// predicted time and warm device time on this machine, as CSV on stdout.
    #[test]
    #[ignore = "a measurement of this machine, run by hand"]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn probe_prefill_priorities() {
        use cubecl::zspace::striding::row_major_contiguous_strides;
        use cubek::attention::kernels::AttentionScoring;
        use cubek::matmul::kernels::tensor::PlainTensor;
        let client = crate::CubeDevice::default().client();
        let predictor = Predictor::on_device(&client);
        let (heads, kv_heads, head_dim) = (32, 8, 128);
        println!("seq,routine,setting,priority,pred_us,floor_us,us");
        for seq in [512usize, 2048] {
            let zeros = |shape: [usize; 4]| {
                PlainTensor::zeros(&client, &shape, row_major_contiguous_strides(shape), F16)
            };
            let cache = [1, kv_heads, seq, head_dim];
            let visibility = AttentionVisibility::new(true, None, QueryPlacement::Last);
            let inputs = AttentionTuneInputs::new(
                &client,
                zeros([1, heads, seq, head_dim]),
                zeros(cache),
                zeros(cache),
                None,
                PlainTensor::of_u32s(&client, &[1], &[seq as u32]),
                seq,
                None,
                AttentionScoring::standard(visibility, head_dim),
            )
            .expect("an even group of a float element");
            let key = inputs.many_queries_key();
            let probe = Probe {
                seq,
                inputs: &inputs,
                key: &key,
                predictor: &predictor,
            };
            probe.sweep(
                "cube_staged",
                AttentionTuneCandidate::list(CubeStagedAttentionRoutine, &predictor),
                AttentionTuneCandidate::launch,
            );
            probe.sweep(
                "cube_shuffle",
                AttentionTuneCandidate::list(CubeShuffleAttentionRoutine, &predictor),
                AttentionTuneCandidate::launch,
            );
            probe.sweep(
                "hypercube_last_arrival_shuffle",
                AttentionTuneCandidate::list(
                    HypercubeLastArrivalShuffleAttentionRoutine::new(),
                    &predictor,
                ),
                AttentionTuneCandidate::launch,
            );
            probe.sweep(
                "hypercube_shuffle_partial",
                AttentionTuneCandidate::list(
                    HypercubeShufflePartialAttentionRoutine::new(),
                    &predictor,
                ),
                AttentionTuneCandidate::launch_split_keys,
            );
            probe.sweep(
                "hypercube_fragment_partial",
                AttentionTuneCandidate::list(HypercubeFragmentPartialAttentionRoutine, &predictor),
                AttentionTuneCandidate::launch_split_keys,
            );
            probe.sweep(
                "hypercube_stream_fragment",
                AttentionTuneCandidate::list(HypercubeStreamFragmentAttentionRoutine, &predictor),
                AttentionTuneCandidate::launch,
            );
        }
    }

    /// One prefill problem the probe times every candidate on.
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    struct Probe<'a> {
        seq: usize,
        inputs: &'a AttentionTuneInputs,
        key: &'a AttentionKey,
        predictor: &'a Predictor,
    }

    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    impl Probe<'_> {
        /// Every distinct candidate of `candidates` that prepares, timed through `launch`.
        fn sweep<R, M, SM>(
            &self,
            name: &str,
            candidates: Vec<AttentionTuneCandidate<R>>,
            launch: impl Fn(&AttentionTuneCandidate<R>, AttentionTuneInputs) -> Result<(), String>,
        ) where
            M: AttentionMicrokernel,
            SM: PartialEq,
            R: Routine<
                    Problem = AttentionRoutineProblem,
                    Microkernel = M,
                    Blueprint = AttentionBlueprint<M, SM>,
                >,
        {
            let problem = AttentionRoutineProblem::try_from(self.key).expect("a problem");
            let demand = problem.demand();
            let client = self.inputs.client().clone();
            let hw = self.predictor.hardware();
            let arithmetic = Arithmetic::product(hw, problem.elements.query);
            let mut seen = Vec::new();
            for candidate in &candidates {
                let Some(prepared) = candidate.blueprint(self.key) else {
                    continue;
                };
                if seen.contains(&prepared) {
                    continue;
                }
                seen.push(prepared);
                let Some(estimate) = self.predictor.estimate::<R>(&problem, candidate.strategy())
                else {
                    continue;
                };
                let peaks =
                    crate::tune::Peaks::measured(&client, hw, &estimate, demand, arithmetic);
                let us = |time: Option<core::time::Duration>| {
                    time.map_or(f64::NAN, |time| time.as_secs_f64() * 1e6)
                };
                let predicted = us(peaks.time_of(&estimate, demand, hw));
                let floor = us(peaks.floor_of(demand));
                let run = || launch(candidate, self.inputs.clone()).is_ok();
                if !(run() && run()) {
                    continue;
                }
                let measured = (0..3)
                    .filter_map(|_| {
                        let window = client.profile_start().ok()?;
                        run();
                        let profile = client.profile_end(window).ok()?;
                        let ticks = cubecl::future::block_on(profile.resolve())?;
                        Some(ticks.duration().as_secs_f64() * 1e6)
                    })
                    .reduce(f64::min)
                    .unwrap_or(f64::NAN);
                println!(
                    "{},{name},\"{}\",{},{predicted:.2},{floor:.2},{measured:.2}",
                    self.seq,
                    candidate.strategy(),
                    candidate.priority(self.key).get(),
                );
            }
        }
    }
}
