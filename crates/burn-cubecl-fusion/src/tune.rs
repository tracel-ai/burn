use crate::CubeFusionHandle;
use burn_fusion::stream::{Context, ContextOwned};
use burn_ir::TensorId;
use cubecl::Runtime;
use std::{
    cell::{Cell, UnsafeCell},
    sync::Arc,
    vec::Drain,
};

/// Fusion context used when tuning kernels.
///
/// Either the original context is returned or a fork of the original.
/// The fork is only given when performing autotuning, and not when actually performing the
/// operation.
///
/// # Sequential execution and rollback
///
/// All tune functions (fused and fallback) run on the same thread, sequentially.
/// When a fused optimization fails on the [`Original`](TuneContext::Original) context,
/// the optimization is responsible for rolling back any modifications it made
/// (e.g., restoring input handle strides and re-registering output handles). This
/// guarantees the context is in a clean state before the fallback path executes on it.
///
/// For the [`Fork`](TuneContext::Fork) path (used during benchmarking), failures are
/// simply discarded, the fork is dropped and the original context is untouched.
pub enum TuneContext<'a, R: Runtime> {
    Original(&'a mut Context<'a, CubeFusionHandle<R>>),
    Fork(TuneContextFork<R>),
}

/// A forked context that writes newly created output handles into a shared
/// [`SharedNewHandles`] cell on [`Drop`].
///
/// When execution on a fork produces output handles (tensor IDs not present at
/// fork time), those handles are collected and stored in the shared cell so that
/// the owning [`UnsafeTuneContext::Original`] can persist them on drop if it was
/// never itself executed.
pub struct TuneContextFork<R: Runtime> {
    context: Box<ContextOwned<CubeFusionHandle<R>>>,
    /// Shared with the [`UnsafeTuneContext::Original`] that spawned this fork.
    /// New output handles are pushed here on drop.
    new_handles: Arc<SharedNewHandles<R>>,
    /// Raw pointer to the original [`Context`]. Used at drop time to check
    /// which handle IDs already exist, so only truly new outputs are collected.
    ptr: *mut Context<'static, CubeFusionHandle<R>>,
}

/// Thread-safe shared storage for newly created output handles.
///
/// # Safety
///
/// All access is sequential on the same thread during autotuning the [`Sync`]
/// impl is required for [`Arc`] but concurrent access never occurs.
struct SharedNewHandles<R: Runtime>(UnsafeCell<Vec<(TensorId, CubeFusionHandle<R>)>>);

unsafe impl<R: Runtime> Send for SharedNewHandles<R> {}
unsafe impl<R: Runtime> Sync for SharedNewHandles<R> {}

impl<R: Runtime> SharedNewHandles<R> {
    fn new() -> Self {
        Self(UnsafeCell::new(Vec::new()))
    }

    /// Push a newly created handle. Only called from `TuneContextFork::drop`.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent access (guaranteed by sequential execution).
    unsafe fn push(&self, id: TensorId, handle: CubeFusionHandle<R>) {
        unsafe { &mut *self.0.get() }.push((id, handle));
    }

    /// Read all collected handles. Only called from `UnsafeTuneContext::drop`.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent access and that all writers have finished.
    unsafe fn drain(&self) -> Drain<'_, (TensorId, CubeFusionHandle<R>)> {
        unsafe { &mut *self.0.get() }.drain(..)
    }

    /// Clear all collected handles. Called before each new fork execution in
    /// [`UnsafeTuneContext::get`] to discard outputs from prior benchmark runs.
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent access (guaranteed by sequential execution).
    unsafe fn clear(&self) {
        unsafe { &mut *self.0.get() }.clear();
    }
}

impl<R: Runtime> TuneContextFork<R> {
    /// Convert the forked context into a borrowed [`Context`] for optimization execution.
    pub fn as_context(&mut self) -> Context<'_, CubeFusionHandle<R>> {
        self.context.as_context()
    }
}

impl<R: Runtime> Drop for TuneContextFork<R> {
    fn drop(&mut self) {
        let fork_handles = self.context.handles();

        let original = unsafe { self.ptr.as_ref().unwrap() };
        for id in fork_handles.handle_ids() {
            if !original.handles.has_handle(id)
                && let Some(handle) = fork_handles.get_handle_ref(id)
            {
                // SAFETY: sequential execution no concurrent access.
                unsafe { self.new_handles.push(*id, handle.clone()) };
            }
        }
    }
}

/// Fusion input wrapper containing the context and the optimization.
///
/// # Safety
///
/// This should only be used with the [tuner](cubecl::tune::LocalTuner), since safety assumptions
/// are made based on its behavior.
pub struct TuneInput<R: Runtime, O> {
    context: UnsafeTuneContext<R>,
    optimization: Arc<O>,
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
///
/// # Output handle persistence (clone contract)
///
/// When this context is cloned (for autotuning), the resulting fork shares a
/// [`SharedNewHandles`] cell with the original. Newly created output handles from
/// forked executions are collected via [`TuneContextFork::drop`].
///
/// When the [`Original`](UnsafeTuneContext::Original) is dropped without [`get()`]
/// having been called (i.e., the original context was never used for execution), the
/// collected handles are persisted to the real context. This upholds the [`Clone`]
/// contract: output handles produced by a forked execution are visible in the
/// original context even when the original path was never taken.
enum UnsafeTuneContext<R: Runtime> {
    Original {
        ptr: *mut Context<'static, CubeFusionHandle<R>>,
        /// Tracks whether [`get()`](UnsafeTuneContext::get) was called.
        /// If false at drop time, forked output handles must be persisted.
        executed: Cell<bool>,
        /// Shared with forks cloned from this original.
        new_handles: Arc<SharedNewHandles<R>>,
    },
    Fork {
        context: Box<ContextOwned<CubeFusionHandle<R>>>,
        /// Shared with the original forks write new handles here.
        new_handles: Arc<SharedNewHandles<R>>,
        ptr: *mut Context<'static, CubeFusionHandle<R>>,
    },
}

unsafe impl<R: Runtime> Send for UnsafeTuneContext<R> {}
unsafe impl<R: Runtime> Sync for UnsafeTuneContext<R> {}
unsafe impl<R: Runtime, O> Send for TuneInput<R, O> {}
unsafe impl<R: Runtime, O> Sync for TuneInput<R, O> {}

impl<R: Runtime, O> TuneInput<R, O> {
    /// Create a new autotune input from the [context](Context) and an optimization.
    pub fn new(context: &mut Context<CubeFusionHandle<R>>, optimization: O) -> Self {
        let context = UnsafeTuneContext::new(context);

        Self {
            context,
            optimization: Arc::new(optimization),
        }
    }

    /// Retrieve the [autotune context](TuneContext) for the current input.
    pub fn context(&self) -> TuneContext<'static, R> {
        self.context.get()
    }

    /// Retrieve the optimization for the current input.
    pub fn optimization(&self) -> &O {
        &self.optimization
    }
}

impl<R: Runtime> UnsafeTuneContext<R> {
    fn new(context: &mut Context<'_, CubeFusionHandle<R>>) -> Self {
        let ptr = core::ptr::from_mut(context);

        // It is necessary for the lifetime.
        #[allow(clippy::unnecessary_cast)]
        Self::Original {
            ptr: ptr as *mut Context<'static, _>,
            executed: Cell::new(false),
            new_handles: Arc::new(SharedNewHandles::new()),
        }
    }

    fn get(&self) -> TuneContext<'static, R> {
        match self {
            UnsafeTuneContext::Original { ptr, executed, .. } => {
                executed.set(true);
                TuneContext::Original(unsafe { ptr.as_mut().unwrap() })
            }
            UnsafeTuneContext::Fork {
                context,
                new_handles,
                ptr,
            } => {
                let fork = context.fork();

                // Each new fork execution resets the handles saved by the previous execution,
                // making sure no memory leak is created by keeping handles that were discarded.
                unsafe { new_handles.clear() };
                TuneContext::Fork(TuneContextFork {
                    context: Box::new(fork),
                    new_handles: new_handles.clone(),
                    ptr: *ptr,
                })
            }
        }
    }
}

impl<R: Runtime> Drop for UnsafeTuneContext<R> {
    fn drop(&mut self) {
        if let UnsafeTuneContext::Original {
            ptr,
            executed,
            new_handles,
        } = self
            && !executed.get()
        {
            // The original context was never used for execution persist
            // output handles that were produced by forked executions.
            let context = unsafe { ptr.as_mut().unwrap() };
            // SAFETY: all forks have been dropped (sequential execution),
            // so no concurrent writers.
            let handles = unsafe { new_handles.drain() };
            for (id, handle) in handles {
                context.handles.register_handle(id, handle);
            }
        }
    }
}

impl<R: Runtime, O> Clone for TuneInput<R, O> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            optimization: self.optimization.clone(),
        }
    }
}

impl<R: Runtime> Clone for UnsafeTuneContext<R> {
    fn clone(&self) -> Self {
        match self {
            UnsafeTuneContext::Original {
                ptr, new_handles, ..
            } => {
                let context: &mut Context<'static, CubeFusionHandle<R>> =
                    unsafe { ptr.as_mut().unwrap() };
                let forked = context.fork();

                UnsafeTuneContext::Fork {
                    context: Box::new(forked),
                    new_handles: new_handles.clone(),
                    ptr: *ptr,
                }
            }
            UnsafeTuneContext::Fork {
                context,
                ptr,
                new_handles,
            } => {
                // Fork-of-fork: they modify the same new_handles.
                UnsafeTuneContext::Fork {
                    context: Box::new(context.fork()),
                    new_handles: new_handles.clone(),
                    ptr: *ptr,
                }
            }
        }
    }
}
