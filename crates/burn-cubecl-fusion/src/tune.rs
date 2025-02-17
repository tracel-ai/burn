use super::CubeFusionHandle;
use burn_fusion::stream::{Context, ContextOwned};
use cubecl::Runtime;

/// Fusion context used when tuning kernels.
///
/// Either the original context is returned or a fork of the original.
/// The fork is only given when performing autotuning, and not when actually performing the
/// operation.
pub enum TuneContext<'a, R: Runtime> {
    Original(&'a mut Context<'a, CubeFusionHandle<R>>),
    Fork(Box<ContextOwned<CubeFusionHandle<R>>>),
}

/// Fusion input wrapper containing the context and the optimization.
///
/// # Safety
///
/// This should only be used with the [tuner](cubecl::tune::LocalTuner), since safety assumptions
/// are made based on its behavior.
pub struct TuneInput<R: Runtime, O> {
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
enum UnsafeTuneContext<R: Runtime> {
    Original(*mut Context<'static, CubeFusionHandle<R>>),
    Fork(Box<ContextOwned<CubeFusionHandle<R>>>),
}

unsafe impl<R: Runtime> Send for UnsafeTuneContext<R> {}
unsafe impl<R: Runtime, O> Send for TuneInput<R, O> {}

impl<R: Runtime, O> TuneInput<R, O> {
    /// Create a new autotune input from the [context](Context) and an optimization.
    pub fn new(context: &mut Context<CubeFusionHandle<R>>, optimization: &O) -> Self {
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

impl<R: Runtime> UnsafeTuneContext<R> {
    fn new(context: &mut Context<'_, CubeFusionHandle<R>>) -> Self {
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

impl<R: Runtime, O> Clone for TuneInput<R, O> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            optimization: self.optimization,
        }
    }
}

impl<R: Runtime> Clone for UnsafeTuneContext<R> {
    fn clone(&self) -> Self {
        let context = match self {
            UnsafeTuneContext::Original(ptr) => {
                let context: &mut Context<'static, CubeFusionHandle<R>> =
                    unsafe { ptr.as_mut().unwrap() };
                context.fork()
            }
            UnsafeTuneContext::Fork(context) => context.fork(),
        };
        UnsafeTuneContext::Fork(Box::new(context))
    }
}
