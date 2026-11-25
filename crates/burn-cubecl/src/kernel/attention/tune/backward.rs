//! Choosing a backward pass's strategy by measurement.

use crate::tune::{Autotune, CandidatePriority, GroupPriority, TuneCandidate};
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::attention_backward::kernels::{
    AttentionBackwardCandidate, AttentionBackwardKey, AttentionBackwardLaunch,
    CubeStagedAttentionBackward, HypercubeAttentionBackward, PlaneFragmentAttentionBackward,
    UnitAttentionBackward,
};
use cubek::attention_backward::{AttentionBackwardGradients, AttentionBackwardTensors};
use cubek::std::Hardware;

type AttentionBackwardAutotune = Autotune<AttentionBackwardKey, AttentionBackwardTensors, ()>;

/// The gradients of the attention `tensors` state, through whichever strategy and setting
/// measures fastest for this problem on `device`.
///
/// Autotune races the plane, the cube staged and the hypercube strategies where the device commits
/// to a plane width, the unit strategy where it does not: elsewhere measuring the unit strategy's
/// settings takes longer than the others'. Under them all sits the unit strategy's lead, which
/// asks nothing of the hardware and serves any problem on any device.
pub(crate) fn launch_backward_tuned(
    device: &Device,
    tensors: AttentionBackwardTensors,
) -> AttentionBackwardGradients {
    let client = tensors.client().clone();
    let id = CubeTuneId::new(&client, device);
    let hw = Hardware::from_client(&client);
    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<AttentionBackwardKey, CubeTuneId> = local_tuner!("attention-backward");
    let tunables = TUNER.init(&id, move || {
        let mut autotune = AttentionBackwardAutotune::new(|tensors: &AttentionBackwardTensors| {
            AttentionBackwardKey::from(&tensors.problem())
        });
        let strategies = autotune.group("microkernel", GroupPriority::DEFAULT);
        autotune = match hw.plane_dim() {
            Ok(_) => autotune
                .prioritized(
                    &strategies,
                    AttentionBackwardTuneCandidate::list(PlaneFragmentAttentionBackward, &hw),
                    AttentionBackwardTuneCandidate::launch,
                )
                .prioritized(
                    &strategies,
                    AttentionBackwardTuneCandidate::list(CubeStagedAttentionBackward, &hw),
                    AttentionBackwardTuneCandidate::launch,
                )
                .prioritized(
                    &strategies,
                    AttentionBackwardTuneCandidate::list(HypercubeAttentionBackward, &hw),
                    AttentionBackwardTuneCandidate::launch,
                ),
            Err(_) => autotune.prioritized(
                &strategies,
                AttentionBackwardTuneCandidate::list(UnitAttentionBackward, &hw),
                AttentionBackwardTuneCandidate::launch,
            ),
        };
        autotune.floor(unit_lead(&hw)).finish()
    });
    TUNER.execute(&id, &client, tunables, tensors.clone());
    tensors.into_gradients()
}

/// The autotune's floor: the unit strategy's lead, which asks nothing of the hardware and serves
/// any problem on any device.
fn unit_lead(hw: &Hardware) -> Tunable<AttentionBackwardKey, AttentionBackwardTensors, ()> {
    let hw = hw.clone();
    Tunable::new(
        "unit-attention-backward-lead",
        move |tensors: AttentionBackwardTensors| {
            let strategy = UnitAttentionBackward;
            let lead = strategy.lead(&hw);
            strategy
                .launch_on(&hw, tensors, &lead)
                .map(|_| ())
                .map_err(|why| {
                    format!("the unit strategy's lead does not serve this problem: {why:?}")
                })
        },
    )
}

/// One strategy at one of its settings, as the backward pass's autotune races it, on the device it
/// was listed for.
///
/// Its priority is how far the strategy's widest launch, the walk owned by the keys, falls short
/// of filling the device ([`Hardware::halvings_short_of_filling`]).
#[derive(Debug, Clone)]
struct AttentionBackwardTuneCandidate<S: AttentionBackwardLaunch> {
    candidate: AttentionBackwardCandidate<S>,
    hw: Hardware,
}

impl<S: AttentionBackwardLaunch + Copy> AttentionBackwardTuneCandidate<S> {
    /// Every setting `strategy` lists on `hw`, its default first.
    fn list(strategy: S, hw: &Hardware) -> Vec<Self> {
        AttentionBackwardCandidate::list(strategy, hw)
            .into_iter()
            .map(|candidate| Self {
                candidate,
                hw: hw.clone(),
            })
            .collect()
    }

    /// `tensors`' gradients written by the strategy at its setting
    /// ([`AttentionBackwardCandidate::launch`]).
    fn launch(&self, tensors: AttentionBackwardTensors) -> Result<(), String> {
        self.candidate.launch(tensors)
    }
}

impl<S: AttentionBackwardLaunch> TuneCandidate for AttentionBackwardTuneCandidate<S> {
    type Strategy = S::Settings;
    type Key = AttentionBackwardKey;
    type Blueprint = ();

    fn strategy(&self) -> &S::Settings {
        self.candidate.settings()
    }

    /// First, lowered a level for each halving the widest launch's grid falls short of filling
    /// the device; last where the strategy declines the problem, which it then does at launch
    /// before compiling anything.
    fn priority(&self, key: &AttentionBackwardKey) -> CandidatePriority {
        match self.candidate.widest_launch(key) {
            Some(launch) => {
                CandidatePriority::FIRST.lowered(self.hw.halvings_short_of_filling(&launch))
            }
            None => CandidatePriority::LAST,
        }
    }
}
