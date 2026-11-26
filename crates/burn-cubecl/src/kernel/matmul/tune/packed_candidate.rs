//! The candidate the autotunes race a routine as at the settings it lists for a packed input
//! ([`MatmulPackedTuneCandidate`]).

use super::base::{MatmulProduct, RoutineIdentity};
use crate::tune::{Arithmetic, CandidateIdentity, CandidatePriority, Predictor, TuneCandidate};
use cubek::matmul::kernels::plan::MatmulCall;
use cubek::matmul::kernels::{
    MatmulPackedCandidate, MatmulRoutine, MatmulRoutineProblem, MatmulTuneKey,
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::routine::Routine;

/// One routine at one of the settings it lists for a packed input, reading the packed inputs as
/// stored.
///
/// Its priority is the estimate of the launch it prepares for the packed problem a key describes
/// ([`MatmulPackedCandidate::estimate`]), timed at the device's peaks
/// ([`Predictor::priority_of`]); last on a float key, which the routine declines before compiling
/// anything.
#[derive(Debug, Clone)]
pub(super) struct MatmulPackedTuneCandidate<R: Routine> {
    candidate: MatmulPackedCandidate<R>,
    predictor: Predictor,
}

impl<R: Routine + Copy> MatmulPackedTuneCandidate<R> {
    /// `routine` at each of `settings` on `predictor`'s device, in their order.
    pub(super) fn of(
        routine: R,
        settings: impl IntoIterator<Item = R::Settings>,
        predictor: &Predictor,
    ) -> Vec<Self> {
        MatmulPackedCandidate::of(routine, settings, predictor.hardware())
            .into_iter()
            .map(|candidate| Self {
                candidate,
                predictor: predictor.clone(),
            })
            .collect()
    }

    /// `call`'s product by the routine at its setting, the packed inputs read as stored
    /// ([`MatmulPackedCandidate::launch`]).
    pub(super) fn launch<M: MatmulMicrokernel, SM>(
        &self,
        call: MatmulCall,
    ) -> Result<MatmulProduct, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        let client = call.client.clone();
        self.candidate
            .launch(call)
            .map(|handle| MatmulProduct { client, handle })
    }

    /// [`launch`](Self::launch) with a packed lhs decoded first
    /// ([`MatmulPackedCandidate::launch_activation_decoded`]).
    pub(super) fn launch_activation_decoded<M: MatmulMicrokernel, SM>(
        &self,
        call: MatmulCall,
    ) -> Result<MatmulProduct, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        let client = call.client.clone();
        self.candidate
            .launch_activation_decoded(call)
            .map(|handle| MatmulProduct { client, handle })
    }

    /// What a [`launch_activation_decoded`](Self::launch_activation_decoded) candidate is
    /// identified by: its setting, named apart from the same setting reading the lhs as stored,
    /// which a caller may launch again as a routine; this one it may not, since its lhs is decoded
    /// first.
    pub(super) fn activation_decoded_identity(&self) -> RoutineIdentity
    where
        R::Settings: core::fmt::Display,
    {
        CandidateIdentity::Named(format!("{} activation decoded", self.candidate.settings()))
    }
}

impl<R: Routine<Problem = MatmulRoutineProblem>> TuneCandidate for MatmulPackedTuneCandidate<R>
where
    R::Settings: core::fmt::Display,
{
    type Strategy = R::Settings;
    type Key = MatmulTuneKey;
    type Blueprint = ();

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// Its estimate on the packed problem the key describes, timed at the device's peaks, on the
    /// largest matrix instruction the device offers the product's element; first where the key
    /// states no problem the estimate can be read off, there being nothing to claim against it.
    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        if !key.is_packed() {
            return CandidatePriority::LAST;
        }
        let Some(estimate) = self.candidate.estimate(key) else {
            return CandidatePriority::FIRST;
        };
        let arithmetic = Arithmetic::product(self.predictor.hardware(), key.shape().elem);
        self.predictor.priority_of(Some(estimate), arithmetic)
    }
}
