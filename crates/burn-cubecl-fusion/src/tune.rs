use crate::CubeFusionHandle;
use burn_fusion::stream::Context;
use burn_ir::{HandleContainer, TensorId, TensorIr};
use cubecl::Runtime;
use cubecl::tune::{InputGenerator, TuneInputs};
use hashbrown::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

/// [`TuneInputs`] marker for [`TuneInput`]: a `'static` type whose GAT `At<'a>` resolves
/// to `TuneInput<'a, R, O>`. This is the indirection that lets a
/// `TunableSet<_, FusionTuneInputs<R, O>, _>` live `'static` inside `LocalTuner::init`'s
/// cache while still accepting a borrowing `TuneInput<'a, …>` at `execute` time, via HRTB
/// over `'a`.
pub(crate) struct FusionTuneInputs<R: Runtime, O>(PhantomData<(fn() -> R, fn() -> O)>);

impl<R: Runtime, O: Send + Sync + 'static> TuneInputs for FusionTuneInputs<R, O> {
    type At<'a> = TuneInput<'a, R, O>;
}

/// [`InputGenerator`] for [`TuneInput`]: produces a [`TuneState::Bench`] fork that's
/// used only for timing measurements. Benchmarks discard their outputs, so the returned
/// input skips the handle-tracking machinery that [`TuneState::Fork`] carries for the
/// wasm try-all fallback.
pub(crate) struct FusionInputGen;

impl<K, R, O> InputGenerator<K, FusionTuneInputs<R, O>> for FusionInputGen
where
    K: 'static,
    R: Runtime,
    O: Send + Sync + 'static,
{
    fn generate<'a>(
        &self,
        _key: &K,
        inputs: &<FusionTuneInputs<R, O> as TuneInputs>::At<'a>,
    ) -> <FusionTuneInputs<R, O> as TuneInputs>::At<'a> {
        inputs.for_benchmark()
    }
}

/// Shared staging area for handles produced by a wasm try-all fork.
///
/// [`Fork`](TuneState::Fork)'s `Drop` dumps all of the fork's handles here (replacing any
/// previous contents), and [`Original`](TuneState::Original)'s `Drop` drains this and
/// filters each entry against the real context's current handles — anything the real
/// context doesn't already have is a genuinely new output and gets registered.
///
/// The pipeline is strictly serial (wasm try-all runs on the caller's single thread), so
/// the lock is always uncontended; `spin::Mutex` is there purely to give
/// `Arc<HandleCollector<R>>` `Send + Sync`.
pub(crate) struct HandleCollector<R: Runtime>(spin::Mutex<HashMap<TensorId, CubeFusionHandle<R>>>);

impl<R: Runtime> HandleCollector<R> {
    fn new() -> Self {
        Self(spin::Mutex::new(HashMap::new()))
    }

    fn capture(&self, handles: &HandleContainer<CubeFusionHandle<R>>) {
        let mut bag = self.0.lock();
        bag.clear();
        for id in handles.handle_ids() {
            if let Some(h) = handles.get_handle_ref(id) {
                bag.insert(*id, h.clone());
            }
        }
    }

    fn take(&self) -> HashMap<TensorId, CubeFusionHandle<R>> {
        core::mem::take(&mut *self.0.lock())
    }
}

/// Fusion input for autotuning. Thread the caller's `&mut Context` through the tuning
/// pipeline without `unsafe` by riding cubecl's `'a` lifetime parameter.
pub(crate) struct TuneInput<'a, R: Runtime, O> {
    optimization: Arc<O>,
    state: TuneState<'a, R>,
}

enum TuneState<'a, R: Runtime> {
    /// The real context. The winning candidate executes here directly.
    Original {
        context: &'a mut Context<CubeFusionHandle<R>>,
        /// Shared with any `Fork` spawned from this `Original`. `Drop` drains from here
        /// if the cache-hit path never ran (e.g. wasm fallback succeeded on a fork).
        new_handles: Arc<HandleCollector<R>>,
        /// Set by [`TuneInput::execute`] when the winner ran on this context, which
        /// suppresses the drain above.
        executed: bool,
    },
    /// Benchmark-only fork. Mutations and outputs are discarded after measurement.
    Bench(Box<Context<CubeFusionHandle<R>>>),
    /// Wasm try-all fallback fork. `Drop` captures this fork's full handle set into the
    /// shared staging area so the paired `Original` can promote genuinely-new outputs
    /// at drain time.
    Fork {
        context: Box<Context<CubeFusionHandle<R>>>,
        new_handles: Arc<HandleCollector<R>>,
    },
}

impl<'a, R: Runtime, O> TuneInput<'a, R, O> {
    pub(crate) fn new(context: &'a mut Context<CubeFusionHandle<R>>, optimization: O) -> Self {
        Self {
            optimization: Arc::new(optimization),
            state: TuneState::Original {
                context,
                new_handles: Arc::new(HandleCollector::new()),
                executed: false,
            },
        }
    }

    /// Fork the context into a `Bench` variant for a benchmark trial. See
    /// [`FusionInputGen`].
    fn for_benchmark(&self) -> Self {
        let context = match &self.state {
            TuneState::Original { context, .. } => context.fork(),
            TuneState::Bench(c) => c.fork(),
            TuneState::Fork { context, .. } => context.fork(),
        };
        Self {
            optimization: self.optimization.clone(),
            state: TuneState::Bench(Box::new(context)),
        }
    }

    pub(crate) fn is_original(&self) -> bool {
        matches!(self.state, TuneState::Original { .. })
    }

    fn context(&self) -> &Context<CubeFusionHandle<R>> {
        match &self.state {
            TuneState::Original { context, .. } => context,
            TuneState::Bench(c) => c,
            TuneState::Fork { context, .. } => context,
        }
    }

    pub(crate) fn tensors(&self) -> &HashMap<TensorId, TensorIr> {
        &self.context().tensors
    }

    pub(crate) fn handles(&self) -> &HandleContainer<CubeFusionHandle<R>> {
        &self.context().handles
    }

    pub(crate) fn optimization(&self) -> &O {
        &self.optimization
    }

    /// Consume the input and run `f` with mutable access to the context and
    /// optimization. Consuming `self` is what keeps the `&mut Context` sound.
    pub(crate) fn execute<F, T>(mut self, f: F) -> T
    where
        F: FnOnce(&mut Context<CubeFusionHandle<R>>, &O) -> T,
    {
        match &mut self.state {
            TuneState::Original {
                context, executed, ..
            } => {
                // Suppresses drop-time persistence; the real context was written directly.
                *executed = true;
                f(context, &self.optimization)
            }
            TuneState::Bench(context) => f(context, &self.optimization),
            TuneState::Fork { context, .. } => f(context, &self.optimization),
        }
    }
}

impl<'a, R: Runtime, O> Clone for TuneInput<'a, R, O> {
    fn clone(&self) -> Self {
        // `Bench` clones are per-benchmark-trial sandboxes — no tracking. `Original`
        // clones come from the wasm fallback path (`operations.fastest(i).execute(
        // inputs.clone())`) and must track outputs for eventual promotion.
        let state = match &self.state {
            TuneState::Original {
                context,
                new_handles,
                ..
            } => TuneState::Fork {
                context: Box::new(context.fork()),
                new_handles: new_handles.clone(),
            },
            TuneState::Bench(c) => TuneState::Bench(Box::new(c.fork())),
            TuneState::Fork {
                context,
                new_handles,
            } => TuneState::Fork {
                context: Box::new(context.fork()),
                new_handles: new_handles.clone(),
            },
        };
        Self {
            optimization: self.optimization.clone(),
            state,
        }
    }
}

impl<'a, R: Runtime, O> Drop for TuneInput<'a, R, O> {
    fn drop(&mut self) {
        match &mut self.state {
            TuneState::Original {
                context,
                new_handles,
                executed,
            } => {
                if *executed {
                    return;
                }
                // Cache-hit path never ran. Drain anything the most recent `Fork`
                // deposited and register entries the real context doesn't already have
                // (those are genuine new outputs, not inputs inherited via `fork`).
                for (id, handle) in new_handles.take() {
                    if context.handles.get_handle_ref(&id).is_none() {
                        context.handles.register_handle(id, handle);
                    }
                }
            }
            TuneState::Bench(_) => {}
            TuneState::Fork {
                context,
                new_handles,
            } => {
                new_handles.capture(&context.handles);
            }
        }
    }
}
