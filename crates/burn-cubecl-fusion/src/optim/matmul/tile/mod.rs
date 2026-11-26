//! The fused matmul on cubek's tile routines.
//!
//! A routine stores its product through a destination the caller chooses, so the fused one is
//! the unfused routine's own kernel storing into the trace's element-wise block ([`TraceSink`](sink::TraceSink))
//! instead of a buffer. Which routine runs is the one the backend's matmul autotune elected for
//! the product ([`elected`]); the autotune then races it fused against the fallback, the routine
//! storing its product and the block running as its own kernel.

mod operand;
mod runner;
mod sink;

use super::optimization::{MatmulOptimizationInfo, MatmulOptimizationTuneArg};
use crate::{
    CubeFusionHandle,
    elector::elected_matmul,
    engine::{
        launch::FuseTraceLauncher,
        trace::{TraceError, TuneOutput},
    },
    tune::{FusionInputGen, TuneInput},
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeTuneId,
    tile::launch::{Decline, SinkLaunch, SinkLaunchVisitor},
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::matmul::kernels::{
    MatmulOpStrategy, MatmulRoutineSettings, MatmulTuneKey, create_key,
    plan::MatmulCall,
    select::{problem::MatmulProblem, selector::measured_autotune},
};
use cubek::std::{
    Hardware,
    matmul_dimensions::{K, M, N},
};
use operand::TraceOperand;
use runner::SinkRunner;
use serde::{Deserialize, Serialize};

/// Why a fused product did not run fused; the fallback runs it instead.
#[derive(Debug)]
pub enum TileFusedMatmulError {
    /// No routine is elected for the product.
    NoRoutineElected,
    /// An operand the routines do not read as it is stored.
    Operand(Decline),
    /// The elected routine declined the product, or a sink.
    Routine(Decline),
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
            Self::Operand(why) => write!(f, "an operand is not read as stored: {why:?}"),
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
    /// The routine the plain matmul's race elected for the product, by its name: `None` while
    /// that race has not settled. A fused race run before then measures the tile candidate on no
    /// routine, so its persisted decision must not answer a key whose routine is elected.
    routine: Option<String>,
}

/// Run the fused product: its elected routine storing into the block, or the fallback, whichever
/// the autotune measured faster for its key.
pub(crate) fn execute(mut arg: MatmulOptimizationTuneArg, context: &mut Context<CubeFusionHandle>) {
    static TUNER: LocalTuner<TileFusedMatmulAutotuneKey, CubeTuneId> = local_tuner!();

    arg.routine = matmul_call(&arg.info, context)
        .ok()
        .and_then(|call| elected(&call));

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

/// The routine the backend's matmul autotune elected for `call`; `None` where the product's plan
/// is no single race, or the race has not measured the product's key yet
/// ([`RoutineElector::elect_matmul`](crate::RoutineElector::elect_matmul)).
///
/// The race is the one cubek's selection sends the unfused product to, under the same device
/// identity, so the fused product launches whichever routine the unfused one settled.
fn elected(call: &MatmulCall) -> Option<MatmulRoutineSettings> {
    let (lhs_shape, rhs_shape) = (
        call.lhs.meta.logical_shape().ok()?,
        call.rhs.meta.logical_shape().ok()?,
    );
    let problem = MatmulProblem::new(
        (&lhs_shape, call.lhs.tensor.kind()),
        (&rhs_shape, call.rhs.tensor.kind()),
    );
    let autotune = measured_autotune(MatmulOpStrategy::Autotune, &problem, &call.hw)?;
    elected_matmul(autotune, call)
}

/// The fused product's matmul as a call, read from `context`.
fn matmul_call(
    info: &MatmulOptimizationInfo,
    context: &Context<CubeFusionHandle>,
) -> Result<MatmulCall, TileFusedMatmulError> {
    let op = &info.matmul.op;
    let lhs = TraceOperand::new(context, &op.lhs).call_input([M, K])?;
    let rhs = TraceOperand::new(context, &op.rhs).call_input([K, N])?;
    Ok(MatmulCall {
        hw: Hardware::from_client(&info.client),
        client: info.client.clone(),
        tune_id: CubeTuneId::new(&info.client, &info.device),
        lhs,
        rhs,
        out: dtype_to_storage_type(context.tensors[&op.out.id].dtype),
        destination: None,
    })
}

/// The product's routine storing into the trace's block.
fn execute_fused(
    arg: &MatmulOptimizationTuneArg,
    context: &mut Context<CubeFusionHandle>,
) -> Result<TuneOutput, TraceError<TileFusedMatmulError>> {
    let call = matmul_call(&arg.info, context).map_err(TraceError::RunnerError)?;
    let Some(settings) = arg.routine else {
        return Err(TraceError::RunnerError(
            TileFusedMatmulError::NoRoutineElected,
        ));
    };
    settings.sink_launch(&call, FusedRun { arg, context })
}

/// A fused product waiting for its routine's launch: the trace runs with the launch storing into
/// its block.
struct FusedRun<'a> {
    arg: &'a MatmulOptimizationTuneArg,
    context: &'a mut Context<CubeFusionHandle>,
}

impl SinkLaunchVisitor for FusedRun<'_> {
    type Output = Result<TuneOutput, TraceError<TileFusedMatmulError>>;

    fn visit<L: SinkLaunch>(self, prepared: Result<L, Decline>) -> Self::Output {
        let info = &self.arg.info;
        let launch =
            prepared.map_err(|why| TraceError::RunnerError(TileFusedMatmulError::Routine(why)))?;
        let runner = SinkRunner::new(launch, info.matmul.out.clone());
        FuseTraceLauncher::new(&info.trace, &runner).launch(
            &info.client,
            &info.device,
            self.context,
        )
    }
}

fn create_autotune_key(input: &TuneInput<MatmulOptimizationTuneArg>) -> TileFusedMatmulAutotuneKey {
    let opt = input.optimization();
    let op = &opt.info.matmul.op;
    let routine = opt.routine.map(|settings| settings.to_string());
    let lhs = TraceOperand::new(input.context(), &op.lhs).key();
    let rhs = TraceOperand::new(input.context(), &op.rhs).key();
    let out = dtype_to_storage_type(input.tensors()[&op.out.id].dtype);
    TileFusedMatmulAutotuneKey::new(
        create_key(out, (&lhs.0, lhs.1), (&rhs.0, rhs.1)),
        opt.info.num_output_buffers(),
        opt.info.num_ops_fused(),
        routine,
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
