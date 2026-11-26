//! What a family places its routine candidates with ([`Predictor`]): their estimates timed at
//! the device's peaks, against the problem's roofline.

use super::{Arithmetic, CandidatePriority, Peaks};
use core::time::Duration;
use cubecl::client::Client;
use cubecl::tune::Work;
use cubek::std::Hardware;
use cubek::std::microkernel::KernelEstimate;
use cubek::std::routine::{EstimatedProblem, Routine};

/// The device a family's candidates are listed for, and the client its peaks are probed on
/// where there is a device to probe.
///
/// A candidate list is also built on devices no process holds — the descriptions a test lists a
/// family's candidates on — and there nothing can be timed, so every candidate that prepares is
/// first and a round is its list.
#[derive(Clone)]
pub struct Predictor {
    hw: Hardware,
    client: Option<Client>,
}

/// The device, and whether its peaks can be probed: a client has nothing to show.
impl core::fmt::Debug for Predictor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Predictor")
            .field("hw", &self.hw)
            .field("probed", &self.client.is_some())
            .finish()
    }
}

impl Predictor {
    /// The device `client` runs on, its peaks probed on it.
    pub fn on_device(client: &Client) -> Self {
        Self {
            hw: Hardware::from_client(client),
            client: Some(client.clone()),
        }
    }

    /// The device `hw` describes, with nothing to probe its peaks on: every candidate that prepares
    /// is first.
    pub fn described(hw: Hardware) -> Self {
        Self { hw, client: None }
    }

    /// The device the candidates are listed for.
    pub fn hardware(&self) -> &Hardware {
        &self.hw
    }

    /// Where `R` at `settings` goes in a round for `problem`, computing on `arithmetic`: how far
    /// the time its estimate predicts at the device's peaks sits above the problem's roofline
    /// ([`Peaks::priority_of`]). Last where the routine declines the problem, which it then does
    /// at launch before compiling anything; first where there is no device to probe.
    pub fn predict_priority<R: Routine>(
        &self,
        problem: &R::Problem,
        settings: &R::Settings,
        arithmetic: Arithmetic,
    ) -> CandidatePriority {
        let estimate = self.estimate::<R>(problem, settings).map(|mut estimate| {
            estimate.demand = Some(problem.demand());
            estimate
        });
        self.priority_of(estimate, arithmetic)
    }

    /// Where a candidate whose launch `estimate` describes goes in a round, computing on
    /// `arithmetic`: how far its predicted time sits above the roofline of the demand it states
    /// ([`Peaks::priority_of`]). Last where there is no estimate, its routine declining the
    /// problem; first where there is no device to probe, or the estimate states no demand.
    pub fn priority_of(
        &self,
        estimate: Option<KernelEstimate>,
        arithmetic: Arithmetic,
    ) -> CandidatePriority {
        let Some(estimate) = estimate else {
            return CandidatePriority::LAST;
        };
        let Some(demand) = estimate.demand else {
            return CandidatePriority::FIRST;
        };
        self.priority_of_launches(&[estimate], demand, arithmetic)
    }

    /// Where a candidate that runs the launches `estimates` describe one after the other goes in
    /// a round for a problem of `demand`, computing on `arithmetic`: how far their predicted
    /// times added up sit above the problem's roofline ([`CandidatePriority::above_floor`]), each
    /// launch timed at the peaks its own estimate reads. Last where there is no launch; first
    /// where there is no device to probe, or a launch's time cannot be predicted.
    pub fn priority_of_launches(
        &self,
        estimates: &[KernelEstimate],
        demand: Work,
        arithmetic: Arithmetic,
    ) -> CandidatePriority {
        let Some(first) = estimates.first() else {
            return CandidatePriority::LAST;
        };
        let Some(client) = &self.client else {
            return CandidatePriority::FIRST;
        };
        let peaks = |estimate: &KernelEstimate| {
            Peaks::measured(client, &self.hw, estimate, demand, arithmetic)
        };
        let predicted = estimates
            .iter()
            .try_fold(Duration::ZERO, |total, estimate| {
                let launch = estimate.demand.unwrap_or(demand);
                Some(total + peaks(estimate).time_of(estimate, launch, &self.hw)?)
            });
        match (predicted, peaks(first).floor_of(demand)) {
            (Some(predicted), Some(floor)) => CandidatePriority::above_floor(predicted, floor),
            _ => CandidatePriority::FIRST,
        }
    }

    /// The estimate of `R` at `settings` for `problem` on this device: its grid, then every part
    /// its owners set. `None` where the routine declines the problem.
    pub fn estimate<R: Routine>(
        &self,
        problem: &R::Problem,
        settings: &R::Settings,
    ) -> Option<KernelEstimate> {
        let (blueprint, grid) = R::prepare(&self.hw, problem, settings).ok()?;
        let mut estimate = KernelEstimate::from(&grid);
        R::estimate(problem, &blueprint, &grid, &mut estimate);
        Some(estimate)
    }
}

#[cfg(all(test, any(feature = "cuda", feature = "wgpu", feature = "metal")))]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek::rms_norm::kernels::{PlaneRmsNormRoutine, RmsNormKey, RmsNormRoutineProblem};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// An RMS norm over a prompt's rows of 4096 columns.
    fn norm() -> RmsNormRoutineProblem {
        RmsNormRoutineProblem::from(&RmsNormKey::new(F32, 4096, 512))
    }

    /// The estimate of the first setting that prepares [`norm`] on the device, and the norm's
    /// demand.
    fn lead_estimate(predictor: &Predictor) -> (KernelEstimate, Work) {
        let problem = norm();
        let mut estimate = PlaneRmsNormRoutine::settings(predictor.hardware())
            .iter()
            .find_map(|settings| predictor.estimate::<PlaneRmsNormRoutine>(&problem, settings))
            .expect("a setting the device lists prepares a norm over 4096 columns");
        estimate.demand = Some(problem.demand());
        (estimate, problem.demand())
    }

    /// A candidate that runs a launch after the first is predicted to take both: placed below
    /// the same first launch alone, rather than by its first or its slowest.
    #[test]
    fn a_second_launch_is_predicted_on_top_of_the_first() {
        let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
        let (estimate, demand) = lead_estimate(&predictor);
        let arithmetic = Arithmetic::plain(F32);
        let alone =
            predictor.priority_of_launches(core::slice::from_ref(&estimate), demand, arithmetic);
        let twice =
            predictor.priority_of_launches(&[estimate.clone(), estimate], demand, arithmetic);
        assert!(twice < alone, "{twice:?} against {alone:?}");
        assert!(twice > CandidatePriority::LAST, "{twice:?}");
    }

    /// A candidate that states no launch is tried last.
    #[test]
    fn no_launch_is_tried_last() {
        let predictor = Predictor::on_device(&crate::CubeDevice::default().client());
        let priority = predictor.priority_of_launches(&[], norm().demand(), Arithmetic::plain(F32));
        assert_eq!(priority, CandidatePriority::LAST);
    }
}
