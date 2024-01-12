use super::{ExecutionMode, Exploration, Explorer};
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{OptimizationId, OptimizationItem, OptimizationStore};
use crate::stream::{Stream, TensorOpsDescription};
use crate::{FusionBackend, HandleContainer, OptimizationBuilder};

/// Process the [stream](Stream) following a [policy](Policy).
///
/// Explore and create new opitmizations using explorations
pub(crate) struct Processor<B: FusionBackend> {
    policy: Policy<B::Optimization>,
    explorer: Explorer<B>,
}

impl<B: FusionBackend> Processor<B> {
    /// Create a new stream processor.
    pub fn new(optimizations: Vec<Box<dyn OptimizationBuilder<B>>>) -> Self {
        Self {
            policy: Policy::new(),
            explorer: Explorer::new(optimizations),
        }
    }

    /// Process the [stream](Stream) with the provided mode.
    pub fn process(
        &mut self,
        stream: &mut Stream<B>,
        optimizations: &mut OptimizationStore<B::Optimization>,
        handles: &mut HandleContainer<B>,
        mode: ExecutionMode,
    ) {
        loop {
            if stream.is_empty() {
                break;
            }

            match self.action(optimizations, stream, mode) {
                Action::Explore => {
                    self.explore(stream, optimizations, handles, mode);

                    if self.explorer.up_to_date() {
                        break;
                    }
                }
                Action::Defer => {
                    self.explorer.defer();

                    match mode {
                        ExecutionMode::Lazy => break,
                        ExecutionMode::Sync => panic!("Can't defer while sync"),
                    };
                }
                Action::Execute(id) => {
                    stream.execute(Some(id), handles, optimizations);
                    self.reset(optimizations, stream);
                }
            };

            if let ExecutionMode::Lazy = mode {
                break;
            }
        }
    }

    fn explore(
        &mut self,
        stream: &mut Stream<B>,
        optimizations: &mut OptimizationStore<B::Optimization>,
        handles: &mut HandleContainer<B>,
        mode: ExecutionMode,
    ) {
        match self.explorer.explore(stream, mode) {
            Exploration::NewOptimization(optim) => {
                let id = optim.map(|optim| {
                    Self::on_new_optimization(&self.policy, stream, optimizations, optim, mode)
                });

                stream.execute(id, handles, optimizations);
                self.reset(optimizations, stream);
            }
            Exploration::Continue => {
                if let ExecutionMode::Sync = mode {
                    panic!("Can't continue exploring when sync.")
                }
            }
        }
    }

    fn reset(&mut self, store: &mut OptimizationStore<B::Optimization>, stream: &Stream<B>) {
        self.explorer.reset(stream);
        self.policy.reset();

        // Reset the policy state.
        for i in 0..stream.relative.len() {
            self.policy.update(store, &stream.relative[i]);
        }
    }

    fn action<'a>(
        &'a mut self,
        cache: &'a OptimizationStore<B::Optimization>,
        stream: &Stream<B>,
        mode: ExecutionMode,
    ) -> Action {
        let (stream, next_ops) = Self::split_stream_ref(stream, mode);

        if let Some(next_ops) = next_ops {
            self.policy.update(cache, next_ops)
        }

        self.policy.action(&cache, stream, mode)
    }

    fn split_stream_owned(
        stream: &Stream<B>,
        mode: ExecutionMode,
    ) -> (Vec<TensorOpsDescription>, Option<TensorOpsDescription>) {
        match mode {
            ExecutionMode::Lazy => {
                let stream = stream.split_relative_stream();
                (stream.0.to_vec(), stream.1.cloned())
            }
            ExecutionMode::Sync => (stream.relative.clone(), None),
        }
    }

    fn split_stream_ref(
        stream: &Stream<B>,
        mode: ExecutionMode,
    ) -> (&[TensorOpsDescription], Option<&TensorOpsDescription>) {
        match mode {
            ExecutionMode::Lazy => stream.split_relative_stream(),
            ExecutionMode::Sync => (stream.relative.as_slice(), None),
        }
    }

    fn on_new_optimization(
        policy: &Policy<B::Optimization>,
        stream: &Stream<B>,
        store: &mut OptimizationStore<B::Optimization>,
        builder: &dyn OptimizationBuilder<B>,
        mode: ExecutionMode,
    ) -> OptimizationId {
        let (stream_relative, next_ops) = Self::split_stream_owned(stream, mode);

        // Check if an optimization is available for this stream before creating a new opitmization.
        //
        // Specify a sync execution mode signaling that we want to know if an optimization is
        // available right now even if it isn't the best one.
        match policy.action(store, &stream_relative, ExecutionMode::Sync) {
            Action::Execute(id) => {
                // When we are in lazy mode, a next operation will be available.
                //
                // Since we are adding new opitmization only when the policy action is explore, we
                // know the existing opitmization wasn't flagged as optimal, since the `next_ops'
                // wasn't included in the `end_conditions`.
                //
                // But in this case, we aren't able to actually find a better opitmization, so we
                // flag the next ops as a stopping criteria, so we won't enter exploration mode the
                // next time we see a similar stream following the same pattern.
                if let Some(next_ops) = next_ops {
                    store.add_end_condition(id, next_ops);
                }
                id
            }
            _ => store.add(OptimizationItem {
                stream: stream_relative,
                end_conditions: next_ops.map(|op| vec![op]).unwrap_or_default(),
                value: builder.build(),
            }),
        }
    }
}
