//! Choosing a backward pass's strategy by measurement.

use crate::tune::{
    Arithmetic, Autotune, CandidatePriority, GroupPriority, Predictor, TuneCandidate,
};
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::attention_backward::kernels::{
    AttentionBackwardCandidate, AttentionBackwardKey, AttentionBackwardLaunch,
    AttentionBackwardProblem, CubeStagedAttentionBackward, HypercubeAttentionBackward,
    PlaneFragmentAttentionBackward, UnitAttentionBackward,
};
use cubek::attention_backward::{AttentionBackwardGradients, AttentionBackwardTensors};
use cubek::std::Hardware;
use cubek::std::routine::EstimatedProblem;

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
    let predictor = Predictor::on_device(&client);
    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<AttentionBackwardKey, CubeTuneId> = local_tuner!("attention-backward");
    let tunables = TUNER.init(&id, move || {
        let mut autotune = AttentionBackwardAutotune::new(|tensors: &AttentionBackwardTensors| {
            AttentionBackwardKey::from(&tensors.problem())
        });
        let strategies = autotune.group("microkernel", GroupPriority::DEFAULT);
        autotune = match predictor.hardware().plane_dim() {
            Ok(_) => autotune
                .prioritized(
                    &strategies,
                    AttentionBackwardTuneCandidate::list(
                        PlaneFragmentAttentionBackward,
                        &predictor,
                    ),
                    AttentionBackwardTuneCandidate::launch,
                )
                .prioritized(
                    &strategies,
                    AttentionBackwardTuneCandidate::list(CubeStagedAttentionBackward, &predictor),
                    AttentionBackwardTuneCandidate::launch,
                )
                .prioritized(
                    &strategies,
                    AttentionBackwardTuneCandidate::list(HypercubeAttentionBackward, &predictor),
                    AttentionBackwardTuneCandidate::launch,
                ),
            Err(_) => autotune.prioritized(
                &strategies,
                AttentionBackwardTuneCandidate::list(UnitAttentionBackward, &predictor),
                AttentionBackwardTuneCandidate::launch,
            ),
        };
        autotune.floor(unit_lead(predictor.hardware())).finish()
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
/// Its priority is the time its launches' estimates predict, one after the other, at the device's
/// peaks, against the backward problem's roofline ([`Predictor::priority_of_launches`]).
#[derive(Debug, Clone)]
struct AttentionBackwardTuneCandidate<S: AttentionBackwardLaunch> {
    candidate: AttentionBackwardCandidate<S>,
    predictor: Predictor,
}

impl<S: AttentionBackwardLaunch + Copy> AttentionBackwardTuneCandidate<S> {
    /// Every setting `strategy` lists on `predictor`'s device, its default first.
    fn list(strategy: S, predictor: &Predictor) -> Vec<Self> {
        AttentionBackwardCandidate::list(strategy, predictor.hardware())
            .into_iter()
            .map(|candidate| Self {
                candidate,
                predictor: predictor.clone(),
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

    /// Its launches' estimates timed at the device's peaks and added up, their products on the
    /// largest matrix instruction the device offers the problem's element; last where the
    /// strategy declines the problem, which it then does at launch before compiling anything.
    fn priority(&self, key: &AttentionBackwardKey) -> CandidatePriority {
        let (Ok(problem), Some(estimates)) = (
            AttentionBackwardProblem::try_from(key),
            self.candidate.estimates(key),
        ) else {
            return CandidatePriority::LAST;
        };
        let arithmetic = Arithmetic::product(self.predictor.hardware(), problem.element);
        self.predictor
            .priority_of_launches(&estimates, problem.demand(), arithmetic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek::microkernel::attention::AttentionVisibility;
    use cubek::std::devices;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// An `f32` key of `heads` query heads over `kv_heads` key heads of dim 64, 1024 queries over
    /// as many keys, one batch.
    fn key(heads: usize, kv_heads: usize) -> AttentionBackwardKey {
        AttentionBackwardKey::new(
            F32,
            heads,
            kv_heads,
            64,
            64,
            1024,
            1024,
            1,
            AttentionVisibility::Attended,
        )
    }

    /// A key whose query heads do not split over its key heads is declined by every strategy:
    /// tried after every candidate that runs.
    #[test]
    fn an_uneven_group_is_tried_last() {
        let predictor = Predictor::described(devices::m2_pro());
        for candidate in AttentionBackwardTuneCandidate::list(UnitAttentionBackward, &predictor) {
            assert_eq!(
                candidate.priority(&key(6, 4)),
                CandidatePriority::LAST,
                "{}",
                candidate.strategy()
            );
        }
    }

    /// A strategy that prepares on the device is placed by its launches' prediction, which never
    /// puts it with the declines.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn a_strategy_that_prepares_is_placed_above_the_declines() {
        let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
        let lead =
            AttentionBackwardTuneCandidate::list(UnitAttentionBackward, &predictor).swap_remove(0);
        assert!(lead.priority(&key(8, 2)) > CandidatePriority::LAST);
    }

    /// The prediction times every launch of a strategy, one after the other: the unit
    /// strategy's walk owned by the keys, after the one owned by the queries, places it below
    /// its first launch alone.
    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
    fn every_launch_lowers_the_priority() {
        let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
        let lead =
            AttentionBackwardTuneCandidate::list(UnitAttentionBackward, &predictor).swap_remove(0);
        let key = key(8, 2);
        let problem = AttentionBackwardProblem::try_from(&key).unwrap();
        let estimates = lead
            .candidate
            .estimates(&key)
            .expect("the unit strategy's lead prepares");
        assert_eq!(
            estimates.len(),
            2,
            "the walks owned by the queries and by the keys"
        );
        let arithmetic = Arithmetic::product(predictor.hardware(), problem.element);
        let first = predictor.priority_of_launches(&estimates[..1], problem.demand(), arithmetic);
        let both = lead.priority(&key);
        if !crate::tune::predicted_on_device(&[first, both]) {
            return;
        }
        assert!(both < first, "{both:?} against {first:?}");
    }
}
