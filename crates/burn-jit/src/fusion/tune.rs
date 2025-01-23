use super::JitFusionHandle;
use crate::JitRuntime;
use burn_fusion::stream::{Context, ContextOwned};

/// Fusion context used when tuning kernels.
///
/// Either the original context is returned or a fork of the original.
/// The fork is only given when performing autotuning, and not when actually performing the
/// operation.
pub enum TuneContext<'a, R: JitRuntime> {
    Original(&'a mut Context<'a, JitFusionHandle<R>>),
    Fork(Box<ContextOwned<JitFusionHandle<R>>>),
}

/// Fusion input wrapper containing the context and the optimization.
///
/// # Safety
///
/// This should only be used with the [tuner](cubecl::tune::LocalTuner), since safety assumptions
/// are made based on its behavior.
pub struct TuneInput<R: JitRuntime, O> {
    context: UnsafeTuneContext<R>,
    optimization: *const O,
}

/// Unsafe wrapper around the context.
///
/// # Safety
///
/// The wrapper removes the context lifetime.
///
/// For it to be correct, the context must not be used after the invocation of the
/// [cubecl::tune::LocalTuner::execute] function. This is the case, since autotune functions are
/// tuned using a cloned version of the input; therefore, a fork of the context will be used to find
/// the best kernel to use, which can be async.
enum UnsafeTuneContext<R: JitRuntime> {
    Original(*mut Context<'static, JitFusionHandle<R>>),
    Fork(Box<ContextOwned<JitFusionHandle<R>>>),
}

unsafe impl<R: JitRuntime> Send for UnsafeTuneContext<R> {}
unsafe impl<R: JitRuntime, O> Send for TuneInput<R, O> {}

impl<R: JitRuntime, O> TuneInput<R, O> {
    /// Create a new autotune input from the [context](Context) and an optimization.
    pub fn new(context: &mut Context<JitFusionHandle<R>>, optimization: &O) -> Self {
        let context = UnsafeTuneContext::new(context);
        // We can erase the lifetime for the same reason we do with the context.
        let optimization = core::ptr::from_ref(optimization);

        Self {
            context,
            optimization,
        }
    }

    /// Retrieve the [autotune context](TuneContext) for the current input.
    pub fn context(&self) -> TuneContext<'static, R> {
        self.context.get()
    }

    /// Retrieve the optimization for the current input.
    pub fn optimization(&self) -> &O {
        unsafe { self.optimization.as_ref().unwrap() }
    }
}

impl<R: JitRuntime> UnsafeTuneContext<R> {
    fn new(context: &mut Context<'_, JitFusionHandle<R>>) -> Self {
        let ptr = core::ptr::from_mut(context);

        // It is necessary for the lifetime.
        #[allow(clippy::unnecessary_cast)]
        Self::Original(ptr as *mut Context<'static, _>)
    }

    fn get(&self) -> TuneContext<'static, R> {
        match self {
            UnsafeTuneContext::Original(ptr) => {
                TuneContext::Original(unsafe { ptr.as_mut().unwrap() })
            }
            UnsafeTuneContext::Fork(context) => TuneContext::Fork(Box::new(context.fork())),
        }
    }
}

impl<R: JitRuntime, O> Clone for TuneInput<R, O> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            optimization: self.optimization,
        }
    }
}

impl<R: JitRuntime> Clone for UnsafeTuneContext<R> {
    fn clone(&self) -> Self {
        let context = match self {
            UnsafeTuneContext::Original(ptr) => {
                let context: &mut Context<'static, JitFusionHandle<R>> =
                    unsafe { ptr.as_mut().unwrap() };
                context.fork()
            }
            UnsafeTuneContext::Fork(context) => context.fork(),
        };
        UnsafeTuneContext::Fork(Box::new(context))
    }
}
