//! Matmul autotune over cubek's tile-DSL routines alone ([`cubek::matmul::kernels`]): what the
//! `tile-kernels-only` feature races instead of the multi-level candidates. A product no routine
//! serves fails rather than falling back, so a test run lists what they don't serve yet.

use crate::{CubeTuneId, kernel::matmul::utils::init_matmul_output, tensor::CubeTensor};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::throughput::MemoryAccess;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubek::matmul::definition::{MatmulCost, MatmulGlobalElems};
use cubek::matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, HypercubeLastArrivalMatmulRoutine,
    MatmulRoutine, MatmulRoutineCandidate, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
    UnitWholeKMatmulRoutine,
    select::problem::OperandKind,
    tensor::PlainTensor,
    tune::key::{MatmulTuneKey, create_key},
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::{
    Hardware,
    autotune::{Autotune, Ceiling, GroupPriority, Roofline},
};

type Inputs = (CubeTensor, CubeTensor, CubeTensor);

/// Executes autotune on matmul operations, over cubek's tile-DSL routines alone.
pub fn matmul_autotune(
    lhs: CubeTensor,
    rhs: CubeTensor,
    out: Option<CubeTensor>,
    out_dtype: DType,
) -> CubeTensor {
    let output = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    if lhs.meta.shape().iter().any(|&d| d == 0) || rhs.meta.shape().iter().any(|&d| d == 0) {
        return output;
    }

    static TUNER: LocalTuner<MatmulTuneKey, CubeTuneId> = local_tuner!("tile-kernels");

    let client = lhs.client.clone();
    let tune_client = client.clone();
    let tune_id = CubeTuneId::new(&lhs.client, &lhs.device);
    let tunables = TUNER.init(&tune_id, move || {
        candidates(&Hardware::from_client(&tune_client))
    });

    TUNER.execute(&tune_id, &client, tunables, (lhs, rhs, output.clone()));

    output
}

/// Every routine's settings in one group, all of them raced in one round, and the unit routine's
/// lead under them, which serves any product on any device. Not the relay hypercube routine: its
/// cubes wait on each other.
///
/// The round tries the candidates in the order their cubes suit the problem, and stops at the
/// first one close enough to the problem's roofline ([`roofline`]), as the multi-level autotune's
/// does: the candidates after it are never compiled.
fn candidates(hw: &Hardware) -> TunableSet<MatmulTuneKey, Inputs, ()> {
    let mut autotune = Autotune::new(key).bounded(roofline);
    let routines = autotune.group("tile-kernels", GroupPriority::DEFAULT);
    let lead = MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, hw).swap_remove(0);
    autotune
        .prioritized(
            &routines,
            MatmulRoutineCandidate::list(PlaneShuffleMatmulRoutine, hw),
            launch,
        )
        .prioritized(
            &routines,
            MatmulRoutineCandidate::list(CubeShuffleMatmulRoutine, hw),
            launch,
        )
        .prioritized(
            &routines,
            MatmulRoutineCandidate::list(UnitWholeKMatmulRoutine, hw),
            launch,
        )
        .prioritized(
            &routines,
            MatmulRoutineCandidate::list(CubeStagedMatmulRoutine, hw),
            launch,
        )
        .prioritized(
            &routines,
            MatmulRoutineCandidate::list(HypercubeLastArrivalMatmulRoutine, hw),
            launch,
        )
        .prioritized(
            &routines,
            MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, hw),
            launch,
        )
        .floor(Tunable::new(
            "unit-registers-lead",
            move |inputs: Inputs| launch(&lead, inputs),
        ))
        .finish()
}

/// `lhs · rhs` written into `out` by `candidate`'s routine at its setting, or why it did not run.
fn launch<R, M: MatmulMicrokernel, SM: PartialEq + Clone + 'static>(
    candidate: &MatmulRoutineCandidate<R>,
    (lhs, rhs, out): Inputs,
) -> Result<(), String>
where
    R: MatmulRoutine<M, SM> + Copy,
{
    if lhs.try_scheme().is_some() || rhs.try_scheme().is_some() {
        return Err("the tile-DSL routines read no quantized operand through burn".into());
    }
    let client = out.client.clone();
    let plain = |tensor: CubeTensor| {
        let elem = dtype_to_storage_type(tensor.dtype);
        PlainTensor::new(tensor.binding(), elem)
    };
    candidate
        .launch_into(&client, plain(lhs), plain(rhs), plain(out))
        .map_err(|why| {
            let settings = candidate.settings();
            format!("{settings} does not serve this problem: {why:?}")
        })
}

/// What `lhs · rhs` costs at least: its operations against the instruction its elements run on,
/// and its compulsory bytes against a copy's bandwidth, whichever is slower for the problem.
fn roofline(_key: &MatmulTuneKey, (lhs, rhs, out): &Inputs) -> Roofline {
    let (lhs_shape, rhs_shape) = (lhs.meta.shape(), rhs.meta.shape());
    let rank = lhs_shape.len();
    let cost = MatmulCost {
        batches: out.meta.shape()[..rank - 2].iter().product(),
        m: lhs_shape[rank - 2],
        k: lhs_shape[rank - 1],
        n: rhs_shape[rank - 1],
        elems: MatmulGlobalElems {
            lhs: dtype_to_storage_type(lhs.dtype),
            rhs: dtype_to_storage_type(rhs.dtype),
            out: dtype_to_storage_type(out.dtype),
        },
    };
    Roofline {
        client: out.client.clone(),
        ceiling: Ceiling::Either {
            compute: cost.compute_key(&out.client),
            memory: MemoryAccess::Copy,
        },
        work: cost.work(),
    }
}

fn key((lhs, rhs, out): &Inputs) -> MatmulTuneKey {
    let kind = |tensor: &CubeTensor| match tensor.try_scheme() {
        Some(scheme) => OperandKind::Quantized(*scheme),
        None => OperandKind::Float,
    };
    create_key(
        dtype_to_storage_type(out.dtype),
        (&lhs.meta, kind(lhs)),
        (&rhs.meta, kind(rhs)),
    )
}
