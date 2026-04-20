use crate::CubeFusionHandle;
use burn_fusion::stream::Context;
use burn_ir::{HandleContainer, TensorId, TensorIr};
use cubecl::Runtime;
use cubecl::tune::TuneInputs;
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

/// Shared staging area for handles produced by the most recent forked execution.
///
/// [`Fork`](TuneState::Fork)'s `Drop` dumps *all* of the fork's handles here (replacing any
/// previous contents), and [`Original`](TuneState::Original)'s `Drop` drains this and filters
/// each entry against the real context's current handles — anything the real context doesn't
/// already have is a genuinely new output and gets registered. The filter happens at drain
/// time rather than at fork time, so we don't need to snapshot `Vec<TensorId>` per fork, and
/// we don't need a back-reference from `Fork` to `Original`.
///
/// The pipeline is strictly serial (cubecl drives every benchmark synchronously, and the wasm
/// try-all fallback also runs on the caller's single thread), so the lock is always uncontended.
/// `spin::Mutex` is there purely to give `Arc<HandleCollector<R>>` `Send + Sync`; in practice the
/// lock is never contended.
pub(crate) struct HandleCollector<R: Runtime>(spin::Mutex<HashMap<TensorId, CubeFusionHandle<R>>>);

impl<R: Runtime> HandleCollector<R> {
    fn new() -> Self {
        Self(spin::Mutex::new(HashMap::new()))
    }

    /// Clear the staging area and replace it with every handle in `handles`. Called from
    /// `Fork::drop`.
    fn capture(&self, handles: &HandleContainer<CubeFusionHandle<R>>) {
        let mut bag = self.0.lock();
        bag.clear();
        for id in handles.handle_ids() {
            if let Some(h) = handles.get_handle_ref(id) {
                bag.insert(*id, h.clone());
            }
        }
    }

    /// Take all staged handles, leaving the collector empty. Called from `Original::drop`.
    fn take(&self) -> HashMap<TensorId, CubeFusionHandle<R>> {
        core::mem::take(&mut *self.0.lock())
    }
}

/// Fusion input for autotuning.
///
/// Holds the real [`Context`] via an honest `&'a mut Context<…>`. There's **no `unsafe`** in
/// this file — cubecl's `LocalTuner::execute` now accepts non-`'static` input types via its
/// `'a` lifetime parameter, so we can thread the caller's borrow all the way through the
/// tuning pipeline and let the borrow checker verify that every fork is dropped before the
/// caller's stack frame ends.
pub(crate) struct TuneInput<'a, R: Runtime, O> {
    /// Shared staging area for output handles produced by the most recent forked execution.
    /// See [`HandleCollector`] for the full story on how the filter-at-drain trick works.
    new_handles: Arc<HandleCollector<R>>,
    optimization: Arc<O>,
    state: TuneState<'a, R>,
}

enum TuneState<'a, R: Runtime> {
    /// Wraps the real [`Context`] via a plain `&'a mut` — the borrow checker proves the
    /// lifetime is honored.
    Original {
        context: &'a mut Context<CubeFusionHandle<R>>,
        /// Set by [`TuneInput::execute`] when the closure has run on the real context. If
        /// true at drop time, the staging drain is skipped — the real context has already
        /// been written to directly, so there's nothing to merge.
        executed: bool,
    },
    /// Owned fork used by benchmark runs and by the wasm try-all fallback. Mutations stay
    /// local; [`Drop`] captures the fork's full handle set into the shared staging area so
    /// the original can filter-and-register at drain time.
    Fork(Box<Context<CubeFusionHandle<R>>>),
}

impl<'a, R: Runtime, O> TuneInput<'a, R, O> {
    /// Create a new autotune input from a [`Context`] and an optimization. The returned
    /// `TuneInput<'a, R, O>` borrows `context` for `'a`.
    pub(crate) fn new(context: &'a mut Context<CubeFusionHandle<R>>, optimization: O) -> Self {
        Self {
            new_handles: Arc::new(HandleCollector::new()),
            optimization: Arc::new(optimization),
            state: TuneState::Original {
                context,
                executed: false,
            },
        }
    }

    /// Whether this input wraps the real [`Context`] rather than a fork.
    pub(crate) fn is_original(&self) -> bool {
        matches!(self.state, TuneState::Original { .. })
    }

    /// Read-only access to the tensor map for autotune key generation.
    pub(crate) fn tensors(&self) -> &HashMap<TensorId, TensorIr> {
        match &self.state {
            TuneState::Original { context, .. } => &context.tensors,
            TuneState::Fork(context) => &context.tensors,
        }
    }

    /// Read-only access to the handle container for autotune key generation.
    pub(crate) fn handles(&self) -> &HandleContainer<CubeFusionHandle<R>> {
        match &self.state {
            TuneState::Original { context, .. } => &context.handles,
            TuneState::Fork(context) => &context.handles,
        }
    }

    /// Retrieve the optimization for the current input.
    pub(crate) fn optimization(&self) -> &O {
        &self.optimization
    }

    /// Consume the input and run a closure with mutable access to the [`Context`] and the
    /// optimization. Consuming `self` is what makes the `&mut Context` sound: no other
    /// borrow can exist once it's gone.
    pub(crate) fn execute<F, T>(mut self, f: F) -> T
    where
        F: FnOnce(&mut Context<CubeFusionHandle<R>>, &O) -> T,
    {
        match &mut self.state {
            TuneState::Original { context, executed } => {
                // Suppresses drop-time persistence — the closure runs on the real context
                // directly, so the staging area is irrelevant.
                *executed = true;
                f(context, &self.optimization)
            }
            TuneState::Fork(context) => f(context, &self.optimization),
        }
    }
}

impl<'a, R: Runtime, O> Clone for TuneInput<'a, R, O> {
    fn clone(&self) -> Self {
        let forked = match &self.state {
            TuneState::Original { context, .. } => context.fork(),
            TuneState::Fork(context) => context.fork(),
        };

        Self {
            new_handles: self.new_handles.clone(),
            optimization: self.optimization.clone(),
            state: TuneState::Fork(Box::new(forked)),
        }
    }
}

impl<'a, R: Runtime, O> Drop for TuneInput<'a, R, O> {
    fn drop(&mut self) {
        match &mut self.state {
            TuneState::Original { context, executed } => {
                if *executed {
                    return;
                }
                // The original was never executed via the cache-hit path. Drain the staging
                // area, and for each staged handle register it in the real context *unless*
                // the real context already has it (in which case it's an input inherited via
                // `handles.fork()`, not a new output).
                for (id, handle) in self.new_handles.take() {
                    if context.handles.get_handle_ref(&id).is_none() {
                        context.handles.register_handle(id, handle);
                    }
                }
            }
            TuneState::Fork(context) => {
                // Capture every handle this fork owns into the shared bag, replacing any
                // previous fork's contribution.
                self.new_handles.capture(&context.handles);
            }
        }
    }
}
