use super::{block_scaled, few_rows, many_rows};
use crate::tune::{Autotune, CandidateIdentity, GroupPriority};
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::std::tensor::TensorHandle;
use cubecl::tune::{AutotuneOutput, LocalTuner, TunableSet, local_tuner};
use cubek::matmul::kernels::plan::{MatmulCall, MatmulPlan};
use cubek::matmul::kernels::select::MatmulAutotune;
use cubek::matmul::kernels::{MatmulRoutineSettings, MatmulTuneKey};
use cubek::std::Hardware;
use cubek::std::plan::Fastest;
use std::sync::Arc;

/// The priority each autotune's routines ship at: a caller's to set, since which group a device
/// measures first is a deployment's call, and fixed for the process once an autotune is first
/// built. The block-scaled autotune races one kernel's settings in a group of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulAutotuneGroups {
    /// The few-rows autotune's.
    pub few_rows: GroupPriority,
    /// The many-rows autotune's.
    pub many_rows: GroupPriority,
}

impl Default for MatmulAutotuneGroups {
    fn default() -> Self {
        Self {
            few_rows: GroupPriority::DEFAULT,
            many_rows: GroupPriority::DEFAULT,
        }
    }
}

impl MatmulAutotuneGroups {
    /// The `autotune` race as a plan, its groups at these priorities: its fastest candidate for
    /// the call's key, measuring first where no winner is settled for it. What a caller of cubek's
    /// [`select`](cubek::matmul::kernels::select::selector::select) hands it as the measured
    /// choice.
    pub fn fastest(self, autotune: MatmulAutotune) -> MatmulPlan {
        Box::new(Fastest::new(autotune.name(), move |call: &MatmulCall| {
            launch_tuned(autotune, call, self)
        }))
    }
}

/// What a candidate of every autotune is identified by: a routine by its setting, which something
/// other than the tuner can launch again ([`fastest_routine`]), every other by its name.
pub(super) type RoutineIdentity = CandidateIdentity<MatmulRoutineSettings>;

/// Any autotune's candidates.
pub(super) type MatmulTunables =
    TunableSet<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>;

/// An autotune being built.
pub(super) type CandidateList = Autotune<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity>;

/// What an autotune's candidate hands back: the product, and the client it lives on, which the
/// autotune checks read two candidates' products through.
#[derive(Clone)]
pub(super) struct MatmulProduct {
    /// Read only by the autotune checks.
    #[cfg_attr(not(feature = "autotune-checks"), allow(dead_code))]
    pub client: Client,
    pub handle: TensorHandle,
}

impl AutotuneOutput for MatmulProduct {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        crate::tune::assert_equivalent(&self.client, self.handle.clone(), other.handle);
    }
}

static FEW_ROWS_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("few-rows");
static MANY_ROWS_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("many-rows");
static BLOCK_SCALED_TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("block-scaled");

/// `call` through `autotune`'s fastest candidate, its groups at `groups`, measuring first where no
/// winner is settled for its key: into the call's destination, allocated once before the race
/// where the caller gave none ([`MatmulCall::with_destination`]).
fn launch_tuned(
    autotune: MatmulAutotune,
    call: &MatmulCall,
    groups: MatmulAutotuneGroups,
) -> TensorHandle {
    let (tuner, tunables) = tunables(autotune, call, groups);
    let call = call.clone().with_destination();
    tuner
        .execute(&call.tune_id, &call.client, tunables, call.clone())
        .handle
}

/// The routine, at its setting, `autotune` settled as the fastest for `call`, its groups at
/// `groups`: `None` until the tuner has settled one for the call's key, and where the winner is no
/// routine. What a caller that launches the routine itself, into a sink of its own, asks: the
/// fused matmul, or a custom operation that writes the product with an epilogue of its own.
///
/// # Guarantees
///
/// Reads the tuner and never races: a key no launch has settled yet answers `None`.
#[cfg(any(feature = "fusion", test))]
pub fn fastest_routine(
    autotune: MatmulAutotune,
    call: &MatmulCall,
    groups: MatmulAutotuneGroups,
) -> Option<MatmulRoutineSettings> {
    let (tuner, tunables) = tunables(autotune, call, groups);
    let key = tunables.generate_key(call);
    tuner
        .fastest_identity(&call.tune_id, &tunables, &key)?
        .routine()
}

/// `autotune`'s tuner and its candidates on `call`'s device, built on first touch.
///
/// The one place each set is built: a tuner caches a set per initializer type, so a second
/// `init` site would build a second set.
fn tunables(
    autotune: MatmulAutotune,
    call: &MatmulCall,
    groups: MatmulAutotuneGroups,
) -> (
    &'static LocalTuner<MatmulTuneKey, CubeTuneId>,
    Arc<MatmulTunables>,
) {
    let hw = call.hw.clone();
    let (tuner, candidates): (_, fn(Hardware, MatmulAutotuneGroups) -> MatmulTunables) =
        match autotune {
            MatmulAutotune::FewRows => (&FEW_ROWS_TUNER, few_rows::candidates),
            MatmulAutotune::ManyRows => (&MANY_ROWS_TUNER, many_rows::candidates),
            MatmulAutotune::BlockScaled => (&BLOCK_SCALED_TUNER, block_scaled::candidates),
        };
    let tunables = tuner.init(&call.tune_id, move || candidates(hw.clone(), groups));
    (tuner, tunables)
}

/// An autotune measured on the call itself, its candidates keyed as the call keys them.
pub(super) fn candidate_list() -> CandidateList {
    CandidateList::new(MatmulCall::key)
}
