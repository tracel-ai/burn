use crate::{
    CubeFusionHandle,
    matmul::optimization::{
        DoubleBuffering, Ordered1, Ordered2, Simple, SimpleMultiRows, SimpleUnit, Specialized,
    },
    shared::trace::TuneOutput,
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    matmul::tune_key::{MatmulAutotuneKey, should_tune_double_buffering},
    tune::{LocalTuner, TunableSet, local_tuner},
};
use serde::{Deserialize, Serialize};

use super::optimization::{MatmulOptimization, MatmulVariantSelection};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,
    #[autotune(anchor)]
    num_out_buffers: usize,
    #[autotune(anchor)]
    num_ops: usize,
}

/// Executes autotune on matmul operations
pub fn fused_matmul_autotune<R: Runtime, BT: CubeElement>(
    optimization: &MatmulOptimization<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedMatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R>, input_gen::<R>)
        .with_tunable(tune_fallback::<R, BT>) // First one should always work.
        .with_tunable(tune_fused::<R, BT, SimpleUnit>)
        .with_tunable(tune_fused::<R, BT, Simple>)
        .with_tunable(tune_fused::<R, BT, SimpleMultiRows>)
        .with_tunable(tune_fused::<R, BT, Ordered1>)
        .with_tunable(tune_fused::<R, BT, Ordered2>)
        .with_tunable_optional(tune_fused::<R, BT, DoubleBuffering>, |key| {
            should_tune_double_buffering(key.num_out_buffers > 1, &key.matmul_key)
        })
        .with_tunable_optional(tune_fused::<R, BT, Specialized>, |key| {
            should_tune_double_buffering(key.num_out_buffers > 1, &key.matmul_key)
        });

    TUNER.execute(
        &CubeTuneId::new::<R>(&optimization.client, &optimization.device),
        &optimization.client,
        &tunables,
        TuneInput::new(context, optimization),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, MatmulOptimization<R>>,
) -> FusedMatmulAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let lhs = context.tensors.get(&opt.variants.simple.op.lhs.id).unwrap();
    let rhs = context.tensors.get(&opt.variants.simple.op.rhs.id).unwrap();
    let out = context.tensors.get(&opt.variants.simple.op.out.id).unwrap();

    let lhs_strides = context
        .handles
        .get_handle(&lhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides;
    let rhs_strides = context
        .handles
        .get_handle(&rhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides;

    let key = MatmulAutotuneKey::generate::<R>(
        &opt.client,
        &lhs.shape,
        &rhs.shape,
        &lhs_strides,
        &rhs_strides,
        lhs.dtype.into(),
        rhs.dtype.into(),
        out.dtype.into(),
    );
    FusedMatmulAutotuneKey::new(key, opt.num_output_buffers(), opt.num_ops_fused())
}

fn input_gen<R: Runtime>(
    _key: &FusedMatmulAutotuneKey,
    input: &TuneInput<R, MatmulOptimization<R>>,
) -> TuneInput<R, MatmulOptimization<R>> {
    input.clone()
}

fn tune_fused<R: Runtime, BT: CubeElement, S: MatmulVariantSelection>(
    input: TuneInput<R, MatmulOptimization<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_fused::<BT, S>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT, S>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, MatmulOptimization<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    Ok(match context {
        TuneContext::Original(context) => optimization.execute_fallback::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    })
}
