//! The candidate the autotunes race a routine as at the settings it lists for a packed input
//! ([`MatmulPackedTuneCandidate`]).

use super::base::{MatmulProduct, RoutineIdentity};
use crate::tune::{CandidateIdentity, CandidatePriority, TuneCandidate};
use cubek::matmul::kernels::plan::MatmulCall;
use cubek::matmul::kernels::{
    MatmulPackedCandidate, MatmulRoutine, MatmulTuneKey, PackedContraction,
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::Hardware;
use cubek::std::routine::Routine;

/// One routine at one of the settings it lists for a packed input, reading the packed inputs as
/// stored.
///
/// Its priority says little of the problem: a packed routine's plan reads how a packed input is
/// stored and how its device loads it, which a key does not carry, so a packed key tries the
/// settings in the order the routine lists them, and a float key last, where the routine declines
/// it before compiling anything. What a key does say is its rows: at one, a setting on matrix
/// instructions fills a single row of its fragments, the arithmetic of several rows for the bytes a
/// setting in registers reads once, so it is tried after those rather than beside them.
#[derive(Debug, Clone)]
pub(super) struct MatmulPackedTuneCandidate<R: Routine> {
    candidate: MatmulPackedCandidate<R>,
}

impl<R: Routine + Copy> MatmulPackedTuneCandidate<R> {
    /// `routine` at each of `settings` on `hw`, in their order.
    pub(super) fn of(
        routine: R,
        settings: impl IntoIterator<Item = R::Settings>,
        hw: &Hardware,
    ) -> Vec<Self> {
        MatmulPackedCandidate::of(routine, settings, hw)
            .into_iter()
            .map(|candidate| Self { candidate })
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

impl<R: Routine> TuneCandidate for MatmulPackedTuneCandidate<R>
where
    R::Settings: core::fmt::Display + PackedContraction,
{
    type Strategy = R::Settings;
    type Key = MatmulTuneKey;
    type Blueprint = ();

    fn strategy(&self) -> &R::Settings {
        self.candidate.settings()
    }

    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        let one_row_on_fragments = key.m == 1 && self.candidate.settings().on_matrix_instructions();
        match key.is_packed() && !one_row_on_fragments {
            true => CandidatePriority::FIRST,
            false => CandidatePriority::LAST,
        }
    }
}
