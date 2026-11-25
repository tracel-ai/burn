//! The grouped product's autotune, `out[r] = lhs[r] @ stacked[group(r)]`: cubek's candidates for
//! it ([`MGroupedMatmulCandidates`]), the dense plane shuffle, which holds a chunk of a group's
//! rows against one read of its slab, raced against the dense cube staged routines, which stage
//! them, each launched on the groups padded to its own chunks. A packed stack is read packed.
//!
//! The race is measured on the call's rows spread evenly over its groups
//! ([`MGroupedMatmulTensors::with_spread_rows`]), and is not bounded.

use crate::tune::{Autotune, GroupPriority};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Decline;
use cubecl::tune::{AutotuneOutput, LocalTuner, Tunable, local_tuner};
use cubecl::{CubeTuneId, Device};
use cubek::matmul::kernels::MatmulRoutine;
use cubek::matmul::kernels::indexed::m_grouped::{
    MGroupedMatmulCandidate, MGroupedMatmulCandidates, MGroupedMatmulKey, MGroupedMatmulTensors,
    PaddedRows,
};
use cubek::matmul::kernels::tensor::PlainTensor;
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::Hardware;
use cubek::std::routine::Routine;

/// What a candidate hands back: the product, padded to the candidate's chunks.
#[derive(Clone)]
struct PaddedProduct {
    pub rows: PaddedRows,
}

/// Two candidates may pad to different chunks, so they are compared as the rows they place.
impl AutotuneOutput for PaddedProduct {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        let client = self.rows.layout().client().clone();
        crate::tune::assert_equivalent(&client, self.rows.clone().unpad(), other.rows.unpad());
    }
}

/// An autotune being built.
type CandidateList = Autotune<MGroupedMatmulKey, MGroupedMatmulTensors, PaddedProduct>;

// Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the code
// lives; a cached winner is an index into the candidate list.
static TUNER: LocalTuner<MGroupedMatmulKey, CubeTuneId> = local_tuner!("m-grouped");

/// The grouped product `tensors` states, through whichever candidate measures fastest for it on
/// `device`, unpadded: `[rows, n]` at the rows' element. `routines` is the priority of the
/// routines' group; the floors sit under it.
///
/// # Guarantees
///
/// No row launches nothing and is not tuned: it hands back an empty `[0, n]`.
///
/// # Errors
///
/// As [`launch_tuned_padded`].
pub fn launch_tuned(
    device: &Device,
    tensors: MGroupedMatmulTensors,
    routines: GroupPriority,
) -> Result<TensorHandle, Decline> {
    let rows = tensors.source().rows();
    if rows == 0 {
        let [_, n] = tensors.slab_shape();
        let elem = tensors.source().buffer().elem;
        return Ok(PlainTensor::empty(tensors.client(), [0, n].into(), elem).0);
    }
    launch_tuned_padded(device, tensors, routines).map(PaddedRows::unpad)
}

/// [`launch_tuned`], the output handed back padded to the winner's chunks, for a next step that
/// reads it padded.
///
/// # Assumptions
///
/// The product has a row at least.
///
/// # Errors
///
/// A product no floor serves ([`MGroupedMatmulTensors::served_on_floor`]), so no candidate of the
/// autotune is known to: the caller names what runs instead.
///
/// # Panics
///
/// On no row, which breaks the assumption.
pub fn launch_tuned_padded(
    device: &Device,
    tensors: MGroupedMatmulTensors,
    routines: GroupPriority,
) -> Result<PaddedRows, Decline> {
    let client = tensors.client().clone();
    let hw = Hardware::from_client(&client);
    // A round must reach a floor that answers.
    tensors.served_on_floor(&hw)?;

    let id = CubeTuneId::new(&client, device);
    let tunables = TUNER.init(&id, move || candidates(&hw, routines).finish());
    Ok(TUNER.execute(&id, &client, tunables, tensors).rows)
}

/// The autotune's candidates on `hw`, its routines' group at `routines`: the plane shuffle's, the
/// cube staged routine's and the copy engine's cube staged routine's, in that order, over the
/// plane shuffle's floors, then the cube staged routine's.
fn candidates(hw: &Hardware, routines: GroupPriority) -> CandidateList {
    let mut autotune =
        CandidateList::measured_on(MGroupedMatmulTensors::key, "evenly-spread", spread);
    let microkernel = autotune.group("microkernel", routines);
    let raced = MGroupedMatmulCandidates::raced(hw);
    let autotune = raced
        .plane_shuffle
        .into_iter()
        .fold(autotune, |autotune, raced| {
            autotune.candidate(&microkernel, tunable(raced))
        });
    let autotune = raced
        .cube_staged
        .into_iter()
        .fold(autotune, |autotune, raced| {
            autotune.candidate(&microkernel, tunable(raced))
        });
    let autotune = raced
        .cube_staged_async_copy
        .into_iter()
        .fold(autotune, |autotune, raced| {
            autotune.candidate(&microkernel, tunable(raced))
        });
    let floors = MGroupedMatmulCandidates::floors();
    let autotune = floors
        .plane_shuffle
        .into_iter()
        .fold(autotune, |autotune, floor| autotune.floor(tunable(floor)));
    floors
        .cube_staged
        .into_iter()
        .fold(autotune, |autotune, floor| autotune.floor(tunable(floor)))
}

/// `candidate` as a tunable, named by its setting.
fn tunable<Mk, SM, R>(
    candidate: MGroupedMatmulCandidate<R>,
) -> Tunable<MGroupedMatmulKey, MGroupedMatmulTensors, PaddedProduct>
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Routine + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    Tunable::new(
        &candidate.settings().to_string(),
        move |tensors: MGroupedMatmulTensors| {
            candidate
                .launch(&tensors)
                .map(|rows| PaddedProduct { rows })
        },
    )
}

/// The operands a candidate is measured on: the call's rows spread evenly over its groups. A build
/// that discards launches discards the sort that makes the call's layout, so a kernel reading it
/// would run on garbage.
fn spread(_key: &MGroupedMatmulKey, tensors: &MGroupedMatmulTensors) -> MGroupedMatmulTensors {
    tensors.with_spread_rows()
}
