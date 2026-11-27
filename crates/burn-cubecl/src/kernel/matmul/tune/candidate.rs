//! The candidate the many-rows and few-rows autotunes race a float routine as
//! ([`MatmulTuneCandidate`]), and the matmul family's one priority rule.

use super::base::MatmulProduct;
use crate::tune::{CandidatePriority, TuneCandidate};
use cubecl::ir::{ElemType, FloatKind};
use cubek::matmul::kernels::plan::MatmulCall;
use cubek::matmul::kernels::{
    CubeBlock, GridCoverage, MatmulBlueprint, MatmulRoutine, MatmulRoutineCandidate,
    MatmulRoutinePrepared, MatmulRoutineProblem, MatmulShape, MatmulTuneKey,
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::routine::Routine;
use cubek::std::{Hardware, Instructions, manual_mma};

/// One candidate of a matmul autotune: a float routine at one of its settings, prepared for the
/// device the autotune is built for.
///
/// Its priority is the matmul family's one rule, the output block one cube of its prepared launch
/// covers against the key's extents ([`shape_priority`]).
#[derive(Debug, Clone)]
pub(super) struct MatmulTuneCandidate<R: Routine> {
    candidate: MatmulRoutineCandidate<R>,
    hw: Hardware,
}

impl<R: Routine + Copy> MatmulTuneCandidate<R> {
    /// Every setting `routine` lists on `hw`, its default first.
    pub(super) fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        MatmulRoutineCandidate::list(routine, hw)
            .into_iter()
            .map(|candidate| Self {
                candidate,
                hw: hw.clone(),
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

    /// Where the output block one cube of the prepared launch covers places it on the key's
    /// extents; last where the routine declines the problem, which it then does at launch before
    /// compiling anything.
    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        let Some(prepared) = self.candidate.prepared(key) else {
            return CandidatePriority::LAST;
        };
        // The priority reads the output block alone; the cube's `K` stands in for the step it does
        // not read: the whole of it, or a cube's run of it where several share a box.
        shape_priority(
            &key.shape(),
            prepared.cube_block(),
            masked_cells(&self.hw, key),
        )
    }

    fn blueprint(&self, key: &MatmulTuneKey) -> Option<MatmulRoutinePrepared<M, SM>> {
        self.candidate.prepared(key)
    }
}

/// What the cells a masked grid covers past the problem's edge cost its candidate: the
/// arithmetic a cube spends on them, and nothing else, since a masked cell reads and writes
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MaskedCells {
    /// They take arithmetic from an instruction the problem's bytes leave idle: a product whose
    /// element the device multiplies on a tensor core, whose rate is hundreds of operations per
    /// byte the bus brings. A decode's eight rows padded to a tile of sixty-four cost the bus
    /// nothing, and a staged candidate reads the weight at its rate.
    Free,
    /// They take arithmetic from the instruction the product is bound by, or near enough: a
    /// product on the units' own arithmetic, an integer one among them, whose rate is a few
    /// dozen operations per byte.
    Cost,
}

/// Where a candidate whose cube covers `block` is tried on `shape` — the priority both arms are
/// written in.
///
/// [`Tiled`](GridCoverage::Tiled) candidates go first. A [`Masked`](GridCoverage::Masked)
/// candidate falls one priority level per doubling of what its grid covers against what the
/// problem asks for, where those cells cost ([`MaskedCells::Cost`]); where they are free it goes
/// first with the tiled ones. A [`Whole`](GridCoverage::Whole) candidate goes last: a grid that
/// cannot spread is the one claim about a launch that needs no clock — and even that says only
/// where to try it, since how much it costs is still a measurement.
pub(super) fn shape_priority(
    shape: &MatmulShape,
    block: CubeBlock,
    masked: MaskedCells,
) -> CandidatePriority {
    match shape.coverage(block) {
        GridCoverage::Tiled => CandidatePriority::FIRST,
        GridCoverage::Masked if masked == MaskedCells::Free => CandidatePriority::FIRST,
        GridCoverage::Masked => CandidatePriority::FIRST
            .lowered((shape.covered(block) / shape.work().max(1)).max(1).ilog2()),
        GridCoverage::Whole => CandidatePriority::LAST,
    }
}

/// What the cells a masked grid covers cost a product of `key` on `hw`: free where the device
/// multiplies the key's element on a tensor core, accumulating at `f32`, and a cost otherwise.
fn masked_cells(hw: &Hardware, key: &MatmulTuneKey) -> MaskedCells {
    let elem = key.shape().elem;
    let f32 = ElemType::Float(FloatKind::F32);
    let matmul = &hw.properties.features.matmul;
    let tensor_core = manual_mma(hw)
        .chain(matmul.cmma.iter().copied())
        .for_triple(elem, elem, f32)
        .next()
        .is_some();
    match tensor_core {
        true => MaskedCells::Free,
        false => MaskedCells::Cost,
    }
}
