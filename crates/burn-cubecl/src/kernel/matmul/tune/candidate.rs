//! The candidate the many-rows and few-rows autotunes race a float routine as
//! ([`MatmulTuneCandidate`]).

use super::base::MatmulProduct;
use crate::tune::{Arithmetic, CandidatePriority, Predictor, TuneCandidate};
use cubek::matmul::kernels::plan::MatmulCall;
use cubek::matmul::kernels::{
    MatmulBlueprint, MatmulRoutine, MatmulRoutineCandidate, MatmulRoutinePrepared,
    MatmulRoutineProblem, MatmulTuneKey,
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::routine::Routine;

/// One candidate of a matmul autotune: a float routine at one of its settings, prepared for the
/// device the autotune is built for.
///
/// Its priority is its routine's estimate timed at the device's peaks, against the product's
/// roofline ([`Predictor::predict_priority`]).
#[derive(Debug, Clone)]
pub(super) struct MatmulTuneCandidate<R: Routine> {
    candidate: MatmulRoutineCandidate<R>,
    predictor: Predictor,
}

impl<R: Routine + Copy> MatmulTuneCandidate<R> {
    /// Every setting `routine` lists on `predictor`'s device, its default first.
    pub(super) fn list(routine: R, predictor: &Predictor) -> Vec<Self> {
        MatmulRoutineCandidate::list(routine, predictor.hardware())
            .into_iter()
            .map(|candidate| Self {
                candidate,
                predictor: predictor.clone(),
            })
            .collect()
    }

    /// `call`'s product by the routine at its setting ([`MatmulRoutineCandidate::launch`]).
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
}

impl<R, M: MatmulMicrokernel, SM: PartialEq> TuneCandidate for MatmulTuneCandidate<R>
where
    R: Routine<Problem = MatmulRoutineProblem, Microkernel = M, Blueprint = MatmulBlueprint<M, SM>>,
{
    type Strategy = R::Settings;
    type Key = MatmulTuneKey;
    type Blueprint = MatmulRoutinePrepared<M, SM>;

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    /// Its estimate timed at the device's peaks, on the largest matrix instruction the device
    /// offers the product's element where it runs on one; last on a packed key, which a float
    /// routine declines before compiling anything.
    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        if key.is_packed() {
            return CandidatePriority::LAST;
        }
        let shape = key.shape();
        self.predictor.predict_priority::<R>(
            &MatmulRoutineProblem::from(&shape),
            self.candidate.settings(),
            Arithmetic::product(self.predictor.hardware(), shape.elem),
        )
    }

    fn blueprint(&self, key: &MatmulTuneKey) -> Option<MatmulRoutinePrepared<M, SM>> {
        self.candidate.prepared(key)
    }
}
