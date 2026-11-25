//! What a family places its routine candidates with ([`Predictor`]): their estimates timed at
//! the device's peaks, against the problem's roofline.

use super::{Arithmetic, CandidatePriority, Peaks};
use cubecl::client::Client;
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
pub(crate) struct Predictor {
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
    pub(crate) fn on_device(client: &Client) -> Self {
        Self {
            hw: Hardware::from_client(client),
            client: Some(client.clone()),
        }
    }

    /// The device `hw` describes, with nothing to probe its peaks on.
    #[cfg(test)]
    pub(crate) fn described(hw: Hardware) -> Self {
        Self { hw, client: None }
    }

    /// The device the candidates are listed for.
    pub(crate) fn hardware(&self) -> &Hardware {
        &self.hw
    }

    /// Where `R` at `settings` goes in a round for `problem`, computing on `arithmetic`: how far
    /// the time its estimate predicts at the device's peaks sits above the problem's roofline
    /// ([`Peaks::priority_of`]). Last where the routine declines the problem, which it then does
    /// at launch before compiling anything; first where there is no device to probe.
    pub(crate) fn predict_priority<R: Routine>(
        &self,
        problem: &R::Problem,
        settings: &R::Settings,
        arithmetic: Arithmetic,
    ) -> CandidatePriority {
        let Some(estimate) = self.estimate::<R>(problem, settings) else {
            return CandidatePriority::LAST;
        };
        let Some(client) = &self.client else {
            return CandidatePriority::FIRST;
        };
        let demand = problem.demand();
        Peaks::measured(client, &self.hw, &estimate, demand, arithmetic)
            .priority_of(&estimate, demand, &self.hw)
    }

    /// The estimate of `R` at `settings` for `problem` on this device: its grid, then every part
    /// its owners set. `None` where the routine declines the problem.
    pub(crate) fn estimate<R: Routine>(
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
