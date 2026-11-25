//! The routed product's autotune, `out[r, s] = lhs[r] @ stacked[routes[r, s]]`: cubek's
//! candidates for it ([`RoutedMatmulCandidates`]), the dense plane shuffle, cube staged and unit
//! registers routines, each at the settings listed for the stack's kind, raced on the product's
//! own operands. A packed stack is read packed.
//!
//! The race is measured on the call's rows and stack, each row's slots routed round the experts in
//! turn ([`RoutedMatmulTensors::with_spread_routes`]), and bounded by the product's roofline.

use super::base::MatmulProduct;
use super::roofline;
use crate::tune::{Autotune, GroupPriority};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::tune::{LocalTuner, Tunable, TuneGroup, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::matmul::kernels::MatmulRoutine;
use cubek::matmul::kernels::indexed::routed::{
    RoutedMatmulCandidate, RoutedMatmulCandidates, RoutedMatmulKey, RoutedMatmulTensors,
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::Hardware;
use cubek::std::routine::Routine;

/// An autotune being built.
type CandidateList = Autotune<RoutedMatmulKey, RoutedMatmulTensors, MatmulProduct>;

// Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the code
// lives; a cached winner is an index into the candidate list.
static TUNER: LocalTuner<RoutedMatmulKey, CubeTuneId> = local_tuner!("routed");

/// The routed product `tensors` states, through whichever candidate measures fastest for it on
/// `device`: `[rows, slots, n]` at the rows' element. `routines` is the priority of the routines'
/// group; the floors sit under it.
///
/// # Guarantees
///
/// No row launches nothing and is not tuned: it hands back an empty `[0, slots, n]`.
///
/// # Errors
///
/// A product no floor serves ([`RoutedMatmulTensors::served_on_floor`]), so no candidate of the
/// autotune is known to: the caller names what runs instead.
pub fn launch_tuned(
    device: &Device,
    tensors: RoutedMatmulTensors,
    routines: GroupPriority,
) -> Result<TensorHandle, Decline> {
    if tensors.rows() == 0 {
        return Ok(tensors.output());
    }
    let client = tensors.client().clone();
    let hw = Hardware::from_client(&client);
    // A round must reach a floor that answers.
    tensors.served_on_floor(&hw)?;

    let id = CubeTuneId::new(&client, device);
    let tunables = TUNER.init(&id, move || candidates(&hw, routines).finish());
    Ok(TUNER.execute(&id, &client, tunables, tensors).handle)
}

/// The autotune's candidates on `hw`, its routines' group at `routines`: the plane shuffle's, the
/// cube staged routine's and the unit registers', in that order, over their floors in the same
/// order.
fn candidates(hw: &Hardware, routines: GroupPriority) -> CandidateList {
    let mut autotune =
        CandidateList::measured_on(RoutedMatmulTensors::key, "spread-routes", spread_routes);
    let microkernel = autotune.group("microkernel", routines);
    let autotune = autotune.bounded(roofline::routed);
    let raced = RoutedMatmulCandidates::raced(hw);
    let autotune = in_group(autotune, &microkernel, raced.plane_shuffle);
    let autotune = in_group(autotune, &microkernel, raced.cube_staged);
    let autotune = in_group(autotune, &microkernel, raced.unit_registers);
    let floors = RoutedMatmulCandidates::floors();
    let autotune = floored(autotune, floors.plane_shuffle);
    let autotune = floored(autotune, floors.cube_staged);
    floored(autotune, floors.unit_registers)
}

/// `autotune` with every one of `candidates` added to `group`, in their order.
fn in_group<Mk, SM, R>(
    autotune: CandidateList,
    group: &TuneGroup<RoutedMatmulKey>,
    candidates: Vec<RoutedMatmulCandidate<R>>,
) -> CandidateList
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Routine + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    candidates
        .into_iter()
        .fold(autotune, |autotune, candidate| {
            autotune.candidate(group, tunable(candidate))
        })
}

/// `autotune` with every one of `floors` added as a floor, in their order.
fn floored<Mk, SM, R>(
    autotune: CandidateList,
    floors: Vec<RoutedMatmulCandidate<R>>,
) -> CandidateList
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Routine + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    floors
        .into_iter()
        .fold(autotune, |autotune, floor| autotune.floor(tunable(floor)))
}

/// `candidate` as a tunable, named by its setting.
fn tunable<Mk, SM, R>(
    candidate: RoutedMatmulCandidate<R>,
) -> Tunable<RoutedMatmulKey, RoutedMatmulTensors, MatmulProduct>
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Routine + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    Tunable::new(
        &candidate.settings().to_string(),
        move |tensors: RoutedMatmulTensors| {
            candidate.launch(&tensors).map(|handle| MatmulProduct {
                client: tensors.client().clone(),
                handle,
            })
        },
    )
}

/// The operands a candidate is measured on: the call's rows and stack, each row's slots routed
/// round the experts in turn. A build that discards launches discards those that make the call's
/// routes, so a kernel reading them as indices would run on garbage.
fn spread_routes(_key: &RoutedMatmulKey, tensors: &RoutedMatmulTensors) -> RoutedMatmulTensors {
    tensors.with_spread_routes()
}
