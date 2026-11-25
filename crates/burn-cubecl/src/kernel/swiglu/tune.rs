//! The SwiGLU autotune: the cube routine's settings, raced per key, over its default as the floor.
//!
//! The autotune is not bounded: the gate reads the projection the step just wrote, so a caller
//! meets it warm, and a round read cold would time a kernel nobody runs.

use crate::tune::{
    Autotune, CandidateIdentity, CandidatePriority, GroupPriority, Identity, TuneCandidate,
};
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::tune::{AutotuneOutput, LocalTuner, TunableSet, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::matmul::kernels::tensor::LhsOutput;
use cubek::std::Hardware;
use cubek::std::routine::Routine;
use cubek::swiglu::kernels::{
    CubeSwigluRoutine, SwigluBlueprint, SwigluCall, SwigluKey, SwigluRoutine,
    SwigluRoutineCandidate, SwigluRoutinePrepared, SwigluTensors,
};
use std::sync::Arc;

/// What an autotune's candidates run on: the call, and the output every one of them writes into,
/// allocated once before the race.
#[derive(Clone)]
struct SwigluRace {
    pub call: SwigluCall,
    pub out: TensorHandle,
}

/// What an autotune's candidate hands back: the gate, and the client it lives on, which the
/// autotune checks read two candidates' outputs through.
#[derive(Clone)]
struct SwigluOutput {
    /// Read only by the autotune checks.
    #[cfg_attr(not(feature = "autotune-checks"), allow(dead_code))]
    pub client: Client,
    pub handle: TensorHandle,
}

impl AutotuneOutput for SwigluOutput {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        crate::tune::assert_equivalent(&self.client, self.handle.clone(), other.handle);
    }
}

/// What a candidate of the autotune is identified by: the candidate itself, so a launch outside a
/// round reads the launches the winner prepared in it, or the floor by name.
type SwigluIdentity = CandidateIdentity<SwigluRoutineCandidate<CubeSwigluRoutine>>;

/// The autotune's candidates, as the tuner holds them.
type SwigluTunables = Arc<TunableSet<SwigluKey, SwigluRace, SwigluOutput, SwigluIdentity>>;

static TUNER: LocalTuner<SwigluKey, CubeTuneId> = local_tuner!("activation");

/// The gate of `call` through the autotune, its candidates grouped at priority `tile` on `device`.
///
/// # Guarantees
///
/// - The output is allocated once, before the race: every candidate of a round writes into it,
///   and so does the winner's run on the call.
/// - Fused rows with no elements are not tuned: the floor hands back an empty output of the
///   gate's shape.
pub fn launch_tuned(call: SwigluCall, device: &Device, tile: GroupPriority) -> TensorHandle {
    // Nothing to gate is nothing to tune: an empty input shares the key of a single row, and a
    // round measured on it would time no launch and keep a winner for the decode bucket.
    if call.both.binding.shape.contains(&0) {
        let hw = Hardware::from_client(&call.client);
        return SwigluRoutineCandidate::floor(&hw)
            .launch(&call)
            .expect("the floor gates every float row");
    }
    let client = call.client.clone();
    let out = SwigluTensors::allocate_out(&client, &call.both);
    // Keyed by the device: the set is built from what this device supports.
    let id = CubeTuneId::new(&client, device);
    let tunables = tunables(&id, &client, tile);
    TUNER
        .execute(&id, &client, tunables, SwigluRace { call, out })
        .handle
}

/// The gate of `call`'s rows written into `out`, a matrix of its rows as a product reads its lhs
/// ([`SwigluRoutineCandidate::launch_into`]).
///
/// # Guarantees
///
/// - The gate is written at the setting the autotune elected for the rows, so the output inherits
///   the kernel the unfused round chose, or at the default, the floor, where no round has settled
///   or the elected setting cannot write it — a stage that outgrows the device.
/// - A decline is a launch's before it dispatches, so the floor writes into the same `out`.
///
/// # Errors
///
/// What the floor refuses: a scheme the store does not write, a device without plane
/// instructions, a run too long for a cube's stage, an output that is not the rows in tiles.
pub fn launch_into_tuned(
    call: &SwigluCall,
    device: &Device,
    tile: GroupPriority,
    out: LhsOutput,
) -> Result<(), Decline> {
    if let Some(elected) = fastest_swiglu_routine(call, device, tile)
        && elected.launch_into(call, out.clone()).is_ok()
    {
        return Ok(());
    }
    let hw = Hardware::from_client(&call.client);
    SwigluRoutineCandidate::floor(&hw).launch_into(call, out)
}

/// The candidate the autotune found fastest for the gate of `call`'s rows on `device`, or `None`
/// where no round has settled the key, or the floor named apart won it.
fn fastest_swiglu_routine(
    call: &SwigluCall,
    device: &Device,
    tile: GroupPriority,
) -> Option<SwigluRoutineCandidate<CubeSwigluRoutine>> {
    let id = CubeTuneId::new(&call.client, device);
    let tunables = tunables(&id, &call.client, tile);
    TUNER
        .fastest_identity(&id, &tunables, &SwigluKey::from(&call.both))?
        .routine()
}

/// The autotune for the device `id` names, built on first touch, its candidates grouped at
/// priority `tile`. The one place it is built: a tuner caches a set per initializer *type*, so a
/// second `init` site would build a second set.
fn tunables(id: &CubeTuneId, client: &Client, tile: GroupPriority) -> SwigluTunables {
    let hw = Hardware::from_client(client);
    TUNER.init(id, move || candidates(hw.clone(), tile))
}

/// The routine's candidates on `hw`, identified by their setting, over its default as the floor.
fn candidates(
    hw: Hardware,
    priority: GroupPriority,
) -> TunableSet<SwigluKey, SwigluRace, SwigluOutput, SwigluIdentity> {
    let mut autotune = Autotune::<SwigluKey, SwigluRace, SwigluOutput, SwigluIdentity>::new(
        |race: &SwigluRace| SwigluKey::from(&race.call.both),
    );
    let tile = autotune.group("tile", priority);
    let floor = SwigluTuneCandidate::floor(&hw);
    autotune
        .identified(
            &tile,
            SwigluTuneCandidate::list(CubeSwigluRoutine, &hw),
            |raced| SwigluIdentity::Routine(raced.candidate.clone()),
            SwigluTuneCandidate::run,
        )
        // The default, which runs everywhere: taking `tile` out of the autotune stops the family
        // being measured, not being served.
        .floor(SwigluIdentity::named(
            "cube-swiglu-floor",
            move |race: SwigluRace| floor.run(race),
        ))
        .finish()
}

/// One setting of a SwiGLU routine, as the autotune races it, on the device it was listed for.
///
/// Its priority is how far the prepared launch's grid falls short of filling the device
/// ([`Hardware::halvings_short_of_filling`]): a decode step's single row by a run of a few lines
/// is the shape that leaves it idle.
#[derive(Debug, Clone)]
struct SwigluTuneCandidate<R: Routine> {
    candidate: SwigluRoutineCandidate<R>,
    hw: Hardware,
}

impl SwigluTuneCandidate<CubeSwigluRoutine> {
    /// The cube routine at its default: the autotune's floor.
    fn floor(hw: &Hardware) -> Self {
        Self {
            candidate: SwigluRoutineCandidate::floor(hw),
            hw: hw.clone(),
        }
    }
}

impl<R: Routine + Copy> SwigluTuneCandidate<R> {
    /// Every setting `routine` lists on `hw`, its default first.
    fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        SwigluRoutineCandidate::list(routine, hw)
            .into_iter()
            .map(|candidate| Self {
                candidate,
                hw: hw.clone(),
            })
            .collect()
    }
}

impl<R, M> SwigluTuneCandidate<R>
where
    R: SwigluRoutine<M> + Routine<Microkernel = M, Blueprint = SwigluBlueprint<M>>,
{
    /// The gate of `race`'s call at this candidate's setting, written into the race's output.
    fn run(&self, race: SwigluRace) -> Result<SwigluOutput, String> {
        let SwigluRace { call, out } = race;
        self.candidate
            .launch_contiguous(&call, out)
            .map(|handle| SwigluOutput {
                client: call.client.clone(),
                handle,
            })
            .map_err(|error| error.to_string())
    }
}

impl<R, M: PartialEq> TuneCandidate for SwigluTuneCandidate<R>
where
    R: SwigluRoutine<M> + Routine<Microkernel = M, Blueprint = SwigluBlueprint<M>>,
{
    type Strategy = R::Settings;
    type Key = SwigluKey;
    type Blueprint = SwigluRoutinePrepared<M>;

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &SwigluKey) -> CandidatePriority {
        match self.candidate.prepared(key) {
            Some(prepared) => CandidatePriority::FIRST
                .lowered(self.hw.halvings_short_of_filling(prepared.launch())),
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &SwigluKey) -> Option<SwigluRoutinePrepared<M>> {
        self.candidate.prepared(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    use cubek::matmul::kernels::tensor::PlainTensor;
    use cubek::std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// The memory `output`'s buffer lies in.
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    fn output_memory(output: &TensorHandle) -> impl PartialEq + core::fmt::Debug + use<> {
        PlainTensor::of_handle(output.clone())
            .binding
            .handle
            .memory
            .id()
    }

    /// A race of three fused rows of 128, its output allocated once.
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    fn race_of_three_rows() -> SwigluRace {
        let client = crate::CubeDevice::default().client();
        let f32 = ElemType::Float(FloatKind::F32);
        let call = SwigluCall {
            client: client.clone(),
            both: PlainTensor::zeros(&client, &[3, 128], [128, 1], f32),
        };
        let out = SwigluTensors::allocate_out(&client, &call.both);
        SwigluRace { call, out }
    }

    /// **A race runs on the call's buffers.** The autotune hands its candidates the call's own
    /// fused rows and output, whatever bucket the rows' count is keyed in.
    #[test]
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    fn a_race_runs_on_the_calls_rows() {
        let race = race_of_three_rows();
        let tunables = candidates(
            Hardware::from_client(&race.call.client),
            GroupPriority::DEFAULT,
        );
        let raced = tunables.generate_inputs(&SwigluKey::from(&race.call.both), &race);
        let memory = |tensor: &PlainTensor| tensor.binding.handle.memory.id();
        assert_eq!(memory(&raced.call.both), memory(&race.call.both));
        assert_eq!(output_memory(&raced.out), output_memory(&race.out));
    }

    /// **A race allocates no output.** Every candidate that runs writes the gate into the race's
    /// one output.
    #[test]
    #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
    fn every_candidate_writes_the_races_output() {
        let race = race_of_three_rows();
        let hw = Hardware::from_client(&race.call.client);
        let outputs = SwigluTuneCandidate::list(CubeSwigluRoutine, &hw)
            .iter()
            .filter_map(|candidate| candidate.run(race.clone()).ok())
            .map(|output| output_memory(&output.handle))
            .collect::<Vec<_>>();
        assert!(!outputs.is_empty(), "the floor runs everywhere");
        for output in outputs {
            assert_eq!(output, output_memory(&race.out));
        }
    }

    /// A decode step's single row by the default's run falls short of the device, and a prompt's
    /// rows fill it.
    #[test]
    fn the_priority_reads_how_far_the_grid_falls_short() {
        let hw = devices::strix_halo();
        let lead = SwigluTuneCandidate::list(CubeSwigluRoutine, &hw).swap_remove(0);
        assert_eq!(
            lead.priority(&SwigluKey::new(F16, 2816, 512 * 4)),
            CandidatePriority::FIRST
        );
        assert!(lead.priority(&SwigluKey::new(F16, 2816, 1)) < CandidatePriority::FIRST);
    }
}
