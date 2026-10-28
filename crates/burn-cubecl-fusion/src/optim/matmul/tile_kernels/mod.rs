//! The fused matmul on cubek's tile routines (`tile-kernels-only`).
//!
//! A routine stores its product through a destination the caller chooses, so the fused one is
//! the unfused routine's own kernel storing into the trace's element-wise block ([`TraceSink`](sink::TraceSink))
//! instead of a buffer. Which routine runs is the one cubek's matmul tables elected for the
//! product ([`elected`]); the autotune then races it fused against the fallback, the routine
//! storing its product and the block running as its own kernel.

mod carrier;
mod sink;

use super::optimization::MatmulOptimizationTuneArg;
use crate::{
    CubeFusionHandle,
    engine::{
        launch::FuseTraceLauncher,
        trace::{TraceError, TuneOutput},
    },
    tune::{FusionInputGen, TuneInput},
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use burn_ir::TensorIr;
use burn_std::DType;
use carrier::Carrier;
use cubecl::{
    AutotuneKey, CubeTuneId,
    tile::launch::{Refusal, SinkLaunch, SinkLaunchVisitor},
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::matmul::kernels::{
    MatmulOpStrategy,
    plan::{MatmulCall, MatmulCallInput},
    select::{
        problem::{MatmulProblem, OperandKind},
        selector::measured_table,
    },
    tensor::PlainTensor,
    tune::{
        MatmulTableGroups,
        key::{MatmulTuneKey, create_key},
    },
};
use cubek::std::Hardware;
use serde::{Deserialize, Serialize};

/// Why a fused product did not run fused; the fallback runs it instead.
#[derive(Debug)]
pub enum TileFusedMatmulError {
    /// No routine is elected for the product.
    NoRoutineElected,
    /// A quantized operand, which the fused run does not read yet.
    QuantizedOperand,
    /// The elected routine declined the product, or a sink.
    Routine(Refusal),
    /// The launcher ran the trace twice, and the routine's launch is taken by the first.
    RanTwice,
    /// The trace holds other than one block, which the routine stores into.
    NotOneBlock { blocks: usize },
    /// The block stores in lines of another width than the routine's.
    BlockWidth { block: usize, store: usize },
    /// The block covers another count of elements than the routine stores.
    BlockCoverage { block: usize, store: usize },
}

impl core::fmt::Display for TileFusedMatmulError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoRoutineElected => write!(f, "no routine is elected for the product"),
            Self::QuantizedOperand => write!(f, "a quantized operand is not read fused yet"),
            Self::Routine(why) => write!(f, "the routine declined: {why:?}"),
            Self::RanTwice => write!(f, "the launcher ran the trace twice"),
            Self::NotOneBlock { blocks } => write!(f, "the trace holds {blocks} blocks, not one"),
            Self::BlockWidth { block, store } => write!(
                f,
                "the block stores lines of {block}, the routine lines of {store}"
            ),
            Self::BlockCoverage { block, store } => write!(
                f,
                "the block covers {block} elements, the routine stores {store}"
            ),
        }
    }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct TileFusedMatmulAutotuneKey {
    matmul: MatmulTuneKey,
    #[autotune(anchor)]
    num_out_buffers: usize,
    #[autotune(anchor)]
    num_ops: usize,
}

/// Run the fused product: its elected routine storing into the block, or the fallback, whichever
/// the autotune measured faster for its key.
pub(crate) fn execute(arg: MatmulOptimizationTuneArg, context: &mut Context<CubeFusionHandle>) {
    static TUNER: LocalTuner<TileFusedMatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let client = arg.info.client.clone();
    let tune_id = CubeTuneId::new(&client, &arg.info.device);
    let tunables = TUNER.init(&tune_id, || {
        // The fallback first: it always runs.
        TunableSet::new(create_autotune_key, FusionInputGen)
            .with(Tunable::new("fused_matmul_fallback", tune_fallback))
            .with(Tunable::new("fused_matmul_tile", tune_fused))
    });

    TUNER.execute(&tune_id, &client, tunables, TuneInput::new(context, arg));
}

/// The routine cubek's matmul table elected for `call`, prepared for a sink and handed to
/// `visitor`; `None` where the product's plan is no single table, or the table has not measured
/// the product's key yet.
///
/// The table is the one the unfused product races on, under the same device identity, so the
/// fused product launches whichever routine the unfused one settled.
fn elected<V: SinkLaunchVisitor>(call: &MatmulCall, visitor: V) -> Option<V::Output> {
    let problem = MatmulProblem::new(
        (call.lhs.meta.shape(), call.lhs.tensor.kind()),
        (call.rhs.meta.shape(), call.rhs.tensor.kind()),
    );
    let table = measured_table(MatmulOpStrategy::Autotune, &problem, &call.hw)?;
    let (settings, _) = table.fastest_routine(call, MatmulTableGroups::default())?;
    Some(settings.sink_launch(call, visitor))
}

/// The product's routine storing into the trace's block.
fn execute_fused(
    arg: &MatmulOptimizationTuneArg,
    context: &mut Context<CubeFusionHandle>,
) -> Result<TuneOutput, TraceError<TileFusedMatmulError>> {
    let info = &arg.info;
    let op = &info.matmul.op;
    let lhs = operand(context, &op.lhs).map_err(TraceError::RunnerError)?;
    let rhs = operand(context, &op.rhs).map_err(TraceError::RunnerError)?;
    let call = MatmulCall {
        hw: Hardware::from_client(&info.client),
        client: info.client.clone(),
        tune_id: CubeTuneId::new(&info.client, &info.device),
        lhs: MatmulCallInput::plain(lhs),
        rhs: MatmulCallInput::plain(rhs),
        out: dtype_to_storage_type(context.tensors[&op.out.id].dtype),
        routes: None,
        destination: None,
    };
    elected(&call, FusedRun { arg, context }).unwrap_or(Err(TraceError::RunnerError(
        TileFusedMatmulError::NoRoutineElected,
    )))
}

/// The operand `tensor` names in the trace, bound where its handle lies.
fn operand(
    context: &Context<CubeFusionHandle>,
    tensor: &TensorIr,
) -> Result<PlainTensor, TileFusedMatmulError> {
    let global = &context.tensors[&tensor.id];
    if let DType::QFloat(_) = global.dtype {
        return Err(TileFusedMatmulError::QuantizedOperand);
    }
    let handle = context
        .handles
        .get_handle_ref(&global.id)
        .expect("a fused product's operand has a handle")
        .clone();
    Ok(PlainTensor::new(
        handle.binding(global.shape.clone()),
        dtype_to_storage_type(global.dtype),
    ))
}

/// A fused product waiting for its routine's launch: the trace runs with the launch storing into
/// its block.
struct FusedRun<'a> {
    arg: &'a MatmulOptimizationTuneArg,
    context: &'a mut Context<CubeFusionHandle>,
}

impl SinkLaunchVisitor for FusedRun<'_> {
    type Output = Result<TuneOutput, TraceError<TileFusedMatmulError>>;

    fn visit<L: SinkLaunch>(self, prepared: Result<L, Refusal>) -> Self::Output {
        let info = &self.arg.info;
        let launch =
            prepared.map_err(|why| TraceError::RunnerError(TileFusedMatmulError::Routine(why)))?;
        let carrier = Carrier::new(launch, info.matmul.out.clone());
        FuseTraceLauncher::new(&info.trace, &carrier).launch(
            &info.client,
            &info.device,
            self.context,
        )
    }
}

fn create_autotune_key(input: &TuneInput<MatmulOptimizationTuneArg>) -> TileFusedMatmulAutotuneKey {
    let opt = input.optimization();
    let (tensors, handles) = (input.tensors(), input.handles());
    let op = &opt.info.matmul.op;
    let operand = |tensor: &TensorIr| {
        let global = &tensors[&tensor.id];
        let handle = handles
            .get_handle_ref(&global.id)
            .expect("a fused product's operand has a handle");
        let kind = match global.dtype {
            DType::QFloat(scheme) => OperandKind::Quantized(scheme),
            _ => OperandKind::Float,
        };
        (handle.metadata(global.shape.clone()), kind)
    };
    let (lhs, rhs) = (operand(&op.lhs), operand(&op.rhs));
    let out = dtype_to_storage_type(tensors[&op.out.id].dtype);
    TileFusedMatmulAutotuneKey::new(
        create_key(out, (&lhs.0, lhs.1), (&rhs.0, rhs.1)),
        opt.info.num_output_buffers(),
        opt.info.num_ops_fused(),
    )
}

fn tune_fused(input: TuneInput<MatmulOptimizationTuneArg>) -> Result<TuneOutput, String> {
    let is_original = input.is_original();
    input.execute(|context, opt| match execute_fused(opt, context) {
        Ok(output) => Ok(output),
        Err(_) if is_original => Ok(opt.execute_fallback(context)),
        Err(TraceError::RunnerError(error)) => Err(error.to_string()),
        Err(error) => Err(format!("{error:?}")),
    })
}

fn tune_fallback(input: TuneInput<MatmulOptimizationTuneArg>) -> Result<TuneOutput, String> {
    Ok(input.execute(|context, opt| opt.execute_fallback(context)))
}
