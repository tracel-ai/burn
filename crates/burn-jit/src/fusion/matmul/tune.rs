use crate::{
    fusion::JitFusionHandle, kernel::matmul::MatmulAutotuneKey, BoolElement, JitRuntime, JitTuneId,
};
use burn_fusion::stream::{Context, ContextOwned};
use cubecl::{
    tune::{local_tuner, LocalTuner, TunableSet},
    AutotuneKey,
};
use serde::{Deserialize, Serialize};

use super::optimization::MatmulOptimization;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,
    #[autotune(anchor)]
    num_ops_fused: usize,
    // #[autotune(anchor)]
    // extra_reads: usize,
    // #[autotune(anchor)]
    // extra_writes: usize,
}

/// Executes autotune on matmul operations
pub fn fused_matmul_autotune<R: JitRuntime, BT: BoolElement>(
    optimization: &MatmulOptimization<R>,
    context: &mut Context<JitFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedMatmulAutotuneKey, JitTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R, BT>, input_gen::<R, BT>)
        .with_tunable(tune_fused::<R, BT>)
        .with_tunable(tune_fallback::<R, BT>);

    TUNER.execute(
        &JitTuneId::new::<R>(&optimization.device),
        &optimization.client,
        &tunables,
        FuseAutotuneInput::new(context, optimization),
    );
}

pub enum FuseAutotuneContext<R: JitRuntime> {
    Original(*mut Context<'static, JitFusionHandle<R>>),
    Fork(ContextOwned<JitFusionHandle<R>>),
}

unsafe impl<R: JitRuntime> Send for FuseAutotuneContext<R> {}
unsafe impl<R: JitRuntime> Send for FuseAutotuneInput<R> {}

impl<R: JitRuntime> FuseAutotuneContext<R> {
    fn new(context: &mut Context<'_, JitFusionHandle<R>>) -> Self {
        Self::Original(core::ptr::from_mut(context) as *mut Context<'static, JitFusionHandle<R>>)
    }

    pub fn get(&self) -> SafeContext<'static, R> {
        match self {
            FuseAutotuneContext::Original(ptr) => {
                SafeContext::Original(unsafe { ptr.as_mut().unwrap() })
            }
            FuseAutotuneContext::Fork(context) => SafeContext::Fork(context.fork()),
        }
    }
}

pub enum SafeContext<'a, R: JitRuntime> {
    Original(&'a mut Context<'a, JitFusionHandle<R>>),
    Fork(ContextOwned<JitFusionHandle<R>>),
}

pub struct FuseAutotuneInput<R: JitRuntime> {
    context: FuseAutotuneContext<R>,
    optimization: *const MatmulOptimization<R>,
}

impl<R: JitRuntime> FuseAutotuneInput<R> {
    pub fn new(
        context: &mut Context<JitFusionHandle<R>>,
        optimization: &MatmulOptimization<R>,
    ) -> Self {
        let context = FuseAutotuneContext::new(context);
        let optimization = core::ptr::from_ref(optimization);

        Self {
            context,
            optimization,
        }
    }

    pub fn context(&self) -> SafeContext<'static, R> {
        self.context.get()
    }

    pub fn optimization(&self) -> &MatmulOptimization<R> {
        unsafe { self.optimization.as_ref().unwrap() }
    }
}

impl<R: JitRuntime> Clone for FuseAutotuneInput<R> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            optimization: self.optimization.clone(),
        }
    }
}

impl<R: JitRuntime> Clone for FuseAutotuneContext<R> {
    fn clone(&self) -> Self {
        let context = match self {
            FuseAutotuneContext::Original(ptr) => {
                let context: &mut Context<'static, JitFusionHandle<R>> =
                    unsafe { ptr.as_mut().unwrap() };
                context.fork()
            }
            FuseAutotuneContext::Fork(context) => context.fork(),
        };
        FuseAutotuneContext::Fork(context)
    }
}

pub(crate) fn create_key<R: JitRuntime, BT: BoolElement>(
    input: &FuseAutotuneInput<R>,
) -> FusedMatmulAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        SafeContext::Original(context) => context,
        SafeContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let lhs = context.tensors.get(&opt.matmul.op.lhs.id).unwrap();
    let rhs = context.tensors.get(&opt.matmul.op.rhs.id).unwrap();
    let out = context.tensors.get(&opt.matmul.op.out.id).unwrap();

    let key = MatmulAutotuneKey::from_shape(
        &lhs.shape.clone().into(),
        &rhs.shape.clone().into(),
        out.dtype,
    );
    FusedMatmulAutotuneKey::new(key, opt.len)
}

fn input_gen<R: JitRuntime, BT: BoolElement>(
    _key: &FusedMatmulAutotuneKey,
    input: &FuseAutotuneInput<R>,
) -> FuseAutotuneInput<R> {
    input.clone()
}

fn tune_fused<R: JitRuntime, BT: BoolElement>(input: FuseAutotuneInput<R>) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        SafeContext::Original(context) => optimization.execute_fused::<BT>(context),
        SafeContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: JitRuntime, BT: BoolElement>(
    input: FuseAutotuneInput<R>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        SafeContext::Original(context) => optimization.execute_fallback::<BT>(context),
        SafeContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    };

    Ok(())
}
