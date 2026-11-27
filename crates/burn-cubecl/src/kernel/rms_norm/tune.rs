//! The RMS-norm autotune: every routine's settings, raced per key, over the cube groups' floor.
//!
//! The autotune is not bounded: a norm reads the activations the step just wrote, so a caller
//! meets it warm, and a round read cold would time a kernel nobody runs.

use crate::tune::{
    Arithmetic, Autotune, CandidateIdentity, CandidatePriority, GroupPriority, Identity, Predictor,
    TuneCandidate,
};
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::tune::{AutotuneOutput, LocalTuner, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::matmul::kernels::tensor::{LhsOutput, PlainTensor};
use cubek::rms_norm::kernels::{
    CubeGroupsRmsNormRoutine, CubeRmsNormRoutine, PlaneRmsNormRoutine, RmsNormBlueprint,
    RmsNormCall, RmsNormCandidate, RmsNormKey, RmsNormRoutine, RmsNormRoutineCandidate,
    RmsNormRoutinePrepared, RmsNormRoutineProblem,
};
use cubek::std::Hardware;
use cubek::std::routine::Routine;
use std::sync::Arc;

/// What an autotune's candidates run on: the call, and the output every one of them writes into,
/// allocated once before the race.
#[derive(Clone)]
struct RmsNormRace {
    pub call: RmsNormCall,
    pub out: TensorHandle,
}

/// What an autotune's candidate hands back: the norm, and the client it lives on, which the
/// autotune checks read two candidates' outputs through.
#[derive(Clone)]
struct RmsNormOutput {
    /// Read only by the autotune checks.
    #[cfg_attr(not(feature = "autotune-checks"), allow(dead_code))]
    pub client: Client,
    pub handle: TensorHandle,
}

impl AutotuneOutput for RmsNormOutput {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        crate::tune::assert_equivalent(&self.client, self.handle.clone(), other.handle);
    }
}

/// What a candidate of the autotune is identified by: the candidate itself, or the floor by name.
type RmsNormIdentity = CandidateIdentity<RmsNormCandidate>;

/// The autotune's candidates, as the tuner holds them.
type RmsNormTunables = Arc<TunableSet<RmsNormKey, RmsNormRace, RmsNormOutput, RmsNormIdentity>>;

static TUNER: LocalTuner<RmsNormKey, CubeTuneId> = local_tuner!("norm");

/// The norm of `call` through the autotune, its candidates grouped at priority `tile` on `device`.
///
/// # Guarantees
///
/// - The output is allocated once, before the race: every candidate of a round writes into it,
///   and so does the winner's run on the call.
/// - An input with no elements launches nothing and is not tuned: it hands back an empty output of
///   the input's shape.
pub fn launch_tuned(call: RmsNormCall, device: &Device, tile: GroupPriority) -> TensorHandle {
    let (out, _) = PlainTensor::empty_contiguous(
        &call.client,
        call.input.binding.shape.clone(),
        call.input.elem,
    );
    // Nothing to normalize is nothing to tune: an empty input shares the key of a single row,
    // and a round measured on it would time no launch and keep a winner for the decode bucket.
    if call.input.binding.shape.contains(&0) {
        return out;
    }
    let client = call.client.clone();
    // Keyed by the device: the set is built from what this device supports.
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client, tile);
    TUNER
        .execute(&id, &client, tunables, RmsNormRace { call, out })
        .handle
}

/// The norm of `call` written into `out`, a matrix of its rows as a product reads its lhs
/// ([`RmsNormRoutineCandidate::launch_into`]).
///
/// # Guarantees
///
/// - The norm is written by the routine the autotune elected for the rows, so the output inherits
///   the kernel the unfused round chose, or by the floor where no round has settled or the elected
///   routine cannot write it — the plane routine, whose planes no cube holds, never encodes, and a
///   setting's stage can outgrow the device.
/// - A decline is a launch's before it dispatches, so the floor writes into the same `out`.
///
/// # Errors
///
/// What the floor refuses: a scheme the store does not write, a device without plane
/// instructions, rows in batch entries, a row too long for a cube's stage, an output that is not
/// the rows in tiles.
pub fn launch_into_tuned(
    call: &RmsNormCall,
    device: &Device,
    tile: GroupPriority,
    out: LhsOutput,
) -> Result<(), Decline> {
    if let Some(elected) = fastest_rms_norm_routine(&call.client, device, &call.input, tile)
        && elected.launch_into(call, out.clone()).is_ok()
    {
        return Ok(());
    }
    let hw = Hardware::from_client(&call.client);
    RmsNormRoutineCandidate::floor(&hw).launch_into(call, out)
}

/// The routine, at its setting, that the autotune found fastest for the norm of `input`'s rows on
/// `client`'s `device`, its candidates grouped at priority `tile`: what a fused tail runs the norm
/// on, so it inherits the kernel the unfused round chose.
///
/// # Guarantees
///
/// `None` where no round in this process has settled the key, or the floor named apart won it.
pub fn fastest_rms_norm_routine(
    client: &Client,
    device: &Device,
    input: &PlainTensor,
    tile: GroupPriority,
) -> Option<RmsNormCandidate> {
    let id = CubeTuneId::new(client, device);
    let tunables = tunables(&id, client, tile);
    let key = RmsNormKey::from(input);
    TUNER.fastest_identity(&id, &tunables, &key)?.routine()
}

/// The autotune for the device `id` names, built on first touch, its candidates grouped at
/// priority `tile`. The one place it is built: a tuner caches a set per initializer *type*, so a
/// second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client, tile: GroupPriority) -> RmsNormTunables {
    let predictor = Predictor::on_device(client);
    TUNER.init(id, move || candidates(predictor.clone(), tile))
}

/// Every routine's candidates on `hw`, identified by their routine and setting, over the cube
/// groups' lead as the floor.
fn candidates(
    predictor: Predictor,
    priority: GroupPriority,
) -> TunableSet<RmsNormKey, RmsNormRace, RmsNormOutput, RmsNormIdentity> {
    let mut autotune = Autotune::<RmsNormKey, RmsNormRace, RmsNormOutput, RmsNormIdentity>::new(
        |race: &RmsNormRace| RmsNormKey::from(&race.call.input),
    );
    let tile = autotune.group("tile", priority);
    let floor = RmsNormTuneCandidate::floor(&predictor);
    autotune
        .identified(
            &tile,
            RmsNormTuneCandidate::list(CubeRmsNormRoutine, &predictor),
            RmsNormTuneCandidate::identity,
            RmsNormTuneCandidate::run,
        )
        .identified(
            &tile,
            RmsNormTuneCandidate::list(PlaneRmsNormRoutine, &predictor),
            RmsNormTuneCandidate::identity,
            RmsNormTuneCandidate::run,
        )
        .identified(
            &tile,
            RmsNormTuneCandidate::list(CubeGroupsRmsNormRoutine, &predictor),
            RmsNormTuneCandidate::identity,
            RmsNormTuneCandidate::run,
        )
        // The cube groups' lead, which runs everywhere: taking `tile` out of the autotune stops
        // the family being measured, not being served.
        .floor(RmsNormIdentity::named(
            "cube-groups-rms-norm-floor",
            move |race: RmsNormRace| floor.run(race),
        ))
        .finish()
}

/// One routine at one of its settings, as the RMS-norm autotune races it, on the device it was
/// listed for.
///
/// Its priority is its routine's estimate timed at the device's peaks, against the norm's
/// roofline ([`Predictor::predict_priority`]).
#[derive(Debug, Clone)]
struct RmsNormTuneCandidate<R: Routine> {
    candidate: RmsNormRoutineCandidate<R>,
    predictor: Predictor,
}

impl RmsNormTuneCandidate<CubeGroupsRmsNormRoutine> {
    /// The cube groups at their default: the autotune's floor.
    fn floor(predictor: &Predictor) -> Self {
        Self {
            candidate: RmsNormRoutineCandidate::floor(predictor.hardware()),
            predictor: predictor.clone(),
        }
    }
}

impl<R: Routine + Copy> RmsNormTuneCandidate<R> {
    /// Every setting `routine` lists on `predictor`'s device, its default first.
    fn list(routine: R, predictor: &Predictor) -> Vec<Self> {
        RmsNormRoutineCandidate::list(routine, predictor.hardware())
            .into_iter()
            .map(|candidate| Self {
                candidate,
                predictor: predictor.clone(),
            })
            .collect()
    }

    /// What the autotune identifies this candidate by: its routine and setting.
    fn identity(&self) -> RmsNormIdentity
    where
        RmsNormRoutineCandidate<R>: Into<RmsNormCandidate>,
    {
        RmsNormIdentity::Routine(self.candidate.clone().into())
    }
}

impl<R, M> RmsNormTuneCandidate<R>
where
    R: RmsNormRoutine<M> + Routine<Microkernel = M, Blueprint = RmsNormBlueprint<M>>,
{
    /// The norm of `race`'s call at this candidate's setting, written into the race's output.
    fn run(&self, race: RmsNormRace) -> Result<RmsNormOutput, String> {
        let RmsNormRace { call, out } = race;
        self.candidate
            .launch_contiguous(&call, out)
            .map(|handle| RmsNormOutput {
                client: call.client.clone(),
                handle,
            })
            .map_err(|error| error.to_string())
    }
}

impl<R, M: PartialEq> TuneCandidate for RmsNormTuneCandidate<R>
where
    R: RmsNormRoutine<M>
        + Routine<Problem = RmsNormRoutineProblem, Microkernel = M, Blueprint = RmsNormBlueprint<M>>,
{
    type Strategy = R::Settings;
    type Key = RmsNormKey;
    type Blueprint = RmsNormRoutinePrepared<M>;

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// The routine's estimate at this setting, timed at the device's peaks, against the norm's
    /// roofline, the arithmetic counted in `f32`, which the norm computes in.
    fn priority(&self, key: &RmsNormKey) -> CandidatePriority {
        self.predictor.predict_priority::<R>(
            &RmsNormRoutineProblem::from(key),
            self.candidate.settings(),
            Arithmetic::plain(ElemType::Float(FloatKind::F32)),
        )
    }

    fn blueprint(&self, key: &RmsNormKey) -> Option<RmsNormRoutinePrepared<M>> {
        self.candidate.prepared(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek::std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// The memory `output`'s buffer lies in.
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn output_memory(output: &TensorHandle) -> impl PartialEq + core::fmt::Debug + use<> {
        PlainTensor::of_handle(output.clone())
            .binding
            .handle
            .memory
            .id()
    }

    /// A race of three rows over 64, its output allocated once.
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn race_of_three_rows() -> RmsNormRace {
        let client = crate::CubeDevice::default().client();
        let f32 = ElemType::Float(FloatKind::F32);
        let call = RmsNormCall {
            client: client.clone(),
            input: PlainTensor::zeros(&client, &[3, 64], [64, 1], f32),
            weight: PlainTensor::zeros(&client, &[64], [1], f32),
            epsilon: 1e-6,
        };
        let (out, _) = PlainTensor::empty_contiguous(&client, [3, 64].into(), f32);
        RmsNormRace { call, out }
    }

    /// **A race runs on the call's buffers.** The autotune hands its candidates the call's own
    /// rows, weight and output, whatever bucket the rows' count is keyed in.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn a_race_runs_on_the_calls_rows() {
        let race = race_of_three_rows();
        let tunables = candidates(
            Predictor::on_device(&race.call.client),
            GroupPriority::DEFAULT,
        );
        let raced = tunables.generate_inputs(&RmsNormKey::from(&race.call.input), &race);
        let memory = |tensor: &PlainTensor| tensor.binding.handle.memory.id();
        assert_eq!(memory(&raced.call.input), memory(&race.call.input));
        assert_eq!(memory(&raced.call.weight), memory(&race.call.weight));
        assert_eq!(output_memory(&raced.out), output_memory(&race.out));
    }

    /// **A race allocates no output.** Every candidate that runs writes the norm into the race's
    /// one output.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn every_candidate_writes_the_races_output() {
        let race = race_of_three_rows();
        let hw = Predictor::on_device(&race.call.client);
        let outputs = (RmsNormTuneCandidate::list(CubeGroupsRmsNormRoutine, &hw).iter())
            .map(|candidate| candidate.run(race.clone()))
            .chain(
                RmsNormTuneCandidate::list(PlaneRmsNormRoutine, &hw)
                    .iter()
                    .map(|candidate| candidate.run(race.clone())),
            )
            .chain(
                RmsNormTuneCandidate::list(CubeRmsNormRoutine, &hw)
                    .iter()
                    .map(|candidate| candidate.run(race.clone())),
            )
            .filter_map(Result::ok)
            .map(|output| output_memory(&output.handle))
            .collect::<Vec<_>>();
        assert!(!outputs.is_empty(), "the floor runs everywhere");
        for output in outputs {
            assert_eq!(output, output_memory(&race.out));
        }
    }

    /// On devices a dev machine cannot show: a plane or cube setting is tried last exactly where
    /// it claims no launch.
    #[test]
    fn a_decline_is_tried_last_and_claims_nothing() {
        for key in [
            RmsNormKey::new(F16, 2560, 1),
            RmsNormKey::new(F16, 128, 512 * 24),
            RmsNormKey::new(F16, 8192, 1),
        ] {
            for hw in [
                devices::rdna_range(),
                devices::cdna_wave64(),
                devices::h100(),
                devices::wgpu_no_subgroups(),
                devices::cpu_12(),
            ] {
                let hw = Predictor::described(hw);
                for plane in RmsNormTuneCandidate::list(PlaneRmsNormRoutine, &hw) {
                    let declined = plane.blueprint(&key).is_none();
                    let last = plane.priority(&key) == CandidatePriority::LAST;
                    assert_eq!(declined, last, "{}", plane.strategy());
                }
                for cube in RmsNormTuneCandidate::list(CubeRmsNormRoutine, &hw) {
                    let declined = cube.blueprint(&key).is_none();
                    let last = cube.priority(&key) == CandidatePriority::LAST;
                    assert_eq!(declined, last, "{}", cube.strategy());
                }
            }
        }
    }

    /// A decode step's single row by cubes of one row leaves the device idle, so it is predicted
    /// further above its roofline than a prompt's rows, which fill it.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn a_grid_that_leaves_the_device_idle_is_tried_later() {
        let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
        let lead = RmsNormTuneCandidate::list(PlaneRmsNormRoutine, &predictor).swap_remove(0);
        let idle = lead.priority(&RmsNormKey::new(F16, 128, 1));
        let full = lead.priority(&RmsNormKey::new(F16, 128, 512 * 24));
        if !crate::tune::predicted_on_device(&[idle, full]) {
            return;
        }
        assert!(idle < full, "{idle:?} against {full:?}");
    }
}
