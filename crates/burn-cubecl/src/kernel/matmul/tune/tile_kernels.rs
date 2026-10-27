//! Matmul autotune over cubek's tile-DSL routines alone ([`cubek::matmul::kernels`]): what the
//! `tile-kernels-only` feature races instead of the multi-level candidates. A product no routine
//! serves fails rather than falling back, so a test run lists what they don't serve yet.
//!
//! A quantized input races two ways: read as it is stored, packed under its scales, by the
//! routines at the settings they list for packed inputs; and dequantized into the output's
//! element first, by every float routine. A scheme or a layout the packed routines do not read
//! is served by the second.

use crate::{
    CubeTuneId,
    kernel::{matmul::utils::init_matmul_output, quantization::dequantize},
    ops::swap_dims,
    tensor::CubeTensor,
};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::throughput::MemoryAccess;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use cubek::matmul::definition::{MatmulCost, MatmulGlobalElems};
use cubek::matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedMatmulRoutine, HypercubeLastArrivalMatmulRoutine,
    MatmulPackedCandidate, MatmulRoutine, MatmulRoutineCandidate, PlaneShuffleMatmulRoutine,
    UnitRegistersMatmulRoutine, UnitWholeKMatmulRoutine,
    select::problem::OperandKind,
    tensor::StoredTensor,
    tune::key::{MatmulTuneKey, create_key},
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::{
    Hardware,
    autotune::{Autotune, Ceiling, GroupPriority, Roofline},
    matmul_axes::{K, M, N},
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

    TUNER.execute(&tune_id, &client, tunables, run_as((lhs, rhs, output.clone())));

    output
}

/// The product the routines run for `lhs · rhs` into `out`: itself, or, where it writes a single
/// column, its transpose `(rhsᵀ · lhsᵀ)ᵀ`, the same buffers viewed through swapped strides.
///
/// The routines cut the output's columns into blocks a cube or a unit covers, so a single column
/// leaves almost every lane of a block idle, and nothing reads the long lhs faster than its
/// rows; transposed, the product is one row against a long rhs, which the routines that hold
/// every row whole stream at the bus. The output of one column stays contiguous transposed; a
/// wider one would be written across its strides, so it runs as it is.
fn run_as((lhs, rhs, out): Inputs) -> Inputs {
    let rank = out.meta.num_dims();
    let (rows, columns) = (out.meta.shape()[rank - 2], out.meta.shape()[rank - 1]);
    if columns != 1 || rows == 1 {
        return (lhs, rhs, out);
    }
    let transposed = |tensor: CubeTensor| {
        let rank = tensor.meta.num_dims();
        swap_dims(tensor, rank - 2, rank - 1)
    };
    (transposed(rhs), transposed(lhs), transposed(out))
}

/// Every routine's settings in one group, all of them raced in one round, and the unit routine's
/// lead under them, which serves any product on any device. Not the relay hypercube routine: its
/// cubes wait on each other. The routines that read a packed input as it is stored race at the
/// settings they list for one, first on a quantized product; the float settings decode it first.
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
        .prioritized(
            &routines,
            MatmulPackedCandidate::of(
                PlaneShuffleMatmulRoutine,
                PlaneShuffleMatmulRoutine.packed_settings(),
                hw,
            ),
            launch_packed,
        )
        .prioritized(
            &routines,
            MatmulPackedCandidate::of(
                CubeShuffleMatmulRoutine,
                CubeShuffleMatmulRoutine.packed_settings(hw),
                hw,
            ),
            launch_packed,
        )
        .prioritized(
            &routines,
            MatmulPackedCandidate::of(
                CubeStagedMatmulRoutine,
                CubeStagedMatmulRoutine.packed_settings(hw),
                hw,
            ),
            launch_packed,
        )
        .floor(Tunable::new(
            "unit-registers-lead",
            move |inputs: Inputs| launch(&lead, inputs),
        ))
        .finish()
}

/// `lhs · rhs` written into `out` by `candidate`'s float routine at its setting, a quantized input
/// dequantized into the output's element first, or why it did not run.
fn launch<R, M: MatmulMicrokernel, SM: PartialEq + Clone + 'static>(
    candidate: &MatmulRoutineCandidate<R>,
    (lhs, rhs, out): Inputs,
) -> Result<(), String>
where
    R: MatmulRoutine<M, SM> + Copy,
{
    let client = out.client.clone();
    let lhs = StoredTensor::Plain(dequantize(lhs, out.dtype).plain());
    let rhs = StoredTensor::Plain(dequantize(rhs, out.dtype).plain());
    candidate
        .launch_into(&client, lhs, rhs, out.plain())
        .map_err(|why| {
            let settings = candidate.settings();
            format!("{settings} does not serve this problem: {why:?}")
        })
}

/// `lhs · rhs` written into `out` by `candidate`'s routine at a setting it lists for a packed
/// input, each input read as it is stored ([`CubeTensor::stored`]), or why it did not run.
fn launch_packed<R, M: MatmulMicrokernel, SM>(
    candidate: &MatmulPackedCandidate<R>,
    (lhs, rhs, out): Inputs,
) -> Result<(), String>
where
    R: MatmulRoutine<M, SM> + Copy,
{
    let client = out.client.clone();
    let declined = |why| {
        let settings = candidate.settings();
        format!("{settings} does not serve this packed problem: {why:?}")
    };
    let lhs = lhs.stored([M, K]).map_err(declined)?;
    let rhs = rhs.stored([K, N]).map_err(declined)?;
    candidate
        .launch_into(&client, lhs, rhs, out.plain())
        .map_err(declined)
}

/// What `lhs · rhs` costs at least: its operations against the instruction its elements run on,
/// and its compulsory bytes against a copy's bandwidth, whichever is slower for the problem. A
/// quantized input is counted at the output's element, which it is multiplied in.
fn roofline(_key: &MatmulTuneKey, (lhs, rhs, out): &Inputs) -> Roofline {
    let (lhs_shape, rhs_shape) = (lhs.meta.shape(), rhs.meta.shape());
    let elem = |tensor: &CubeTensor| match tensor.try_scheme() {
        Some(_) => dtype_to_storage_type(out.dtype),
        None => dtype_to_storage_type(tensor.dtype),
    };
    let rank = lhs_shape.len();
    let cost = MatmulCost {
        batches: out.meta.shape()[..rank - 2].iter().product(),
        m: lhs_shape[rank - 2],
        k: lhs_shape[rank - 1],
        n: rhs_shape[rank - 1],
        elems: MatmulGlobalElems {
            lhs: elem(lhs),
            rhs: elem(rhs),
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
