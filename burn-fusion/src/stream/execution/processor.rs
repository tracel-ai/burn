use super::{ExecutionMode, Explorer, ExplorerResult};
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{
    ExecutionStrategy, Exploration, ExplorationId, ExplorationStore, StopCriterion,
};
use crate::stream::{OperationDescription, Stream};
use crate::{FusionBackend, HandleContainer, OptimizationBuilder};

/// Process the [stream](Stream) following a [policy](Policy).
///
/// Explore and create new optimizations using explorations
pub(crate) struct Processor<B: FusionBackend> {
    policy: Policy<B::Optimization>,
    explorer: Explorer<B::Optimization>,
}

pub trait StreamProcessing<O> {
    fn operations<'a>(&'a self) -> &[OperationDescription];
    fn execute(&mut self, id: ExplorationId, store: &mut ExplorationStore<O>);
}

struct StreamItem<'a, B: FusionBackend> {
    stream: &'a mut Stream<B>,
    handles: &'a mut HandleContainer<B>,
}

impl<'i, B: FusionBackend> StreamProcessing<B::Optimization> for StreamItem<'i, B> {
    fn operations<'a>(&'a self) -> &[OperationDescription] {
        &self.stream.relative
    }

    fn execute(&mut self, id: ExplorationId, store: &mut ExplorationStore<B::Optimization>) {
        self.stream.execute(id, self.handles, store)
    }
}

impl<B: FusionBackend> Processor<B> {
    /// Create a new stream processor.
    pub fn new(optimizations: Vec<Box<dyn OptimizationBuilder<B::Optimization>>>) -> Self {
        Self {
            policy: Policy::new(),
            explorer: Explorer::new(optimizations),
        }
    }

    /// Process the [stream](Stream) with the provided mode.
    pub fn process(
        &mut self,
        stream: &mut Stream<B>,
        store: &mut ExplorationStore<B::Optimization>,
        handles: &mut HandleContainer<B>,
        mode: ExecutionMode,
    ) {
        loop {
            if stream.is_empty() {
                break;
            }

            match self.action(store, stream, mode) {
                Action::Explore => {
                    self.explore(stream, store, handles, mode);

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
                    stream.execute(id, handles, store);
                    self.reset(store, stream);
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
        store: &mut ExplorationStore<B::Optimization>,
        handles: &mut HandleContainer<B>,
        mode: ExecutionMode,
    ) {
        match self.explorer.explore(&stream.relative, mode) {
            ExplorerResult::Found(optim) => {
                stream.execute(
                    Self::on_optimization_found(&self.policy, stream, store, optim, mode),
                    handles,
                    store,
                );
                self.reset(store, stream);
            }
            ExplorerResult::NotFound { num_explored } => {
                stream.execute(
                    Self::on_optimization_not_found(
                        &self.policy,
                        stream,
                        store,
                        mode,
                        num_explored,
                    ),
                    handles,
                    store,
                );
                self.reset(store, stream);
            }
            ExplorerResult::Continue => {
                if let ExecutionMode::Sync = mode {
                    panic!("Can't continue exploring when sync.")
                }
            }
        }
    }

    fn reset(&mut self, store: &mut ExplorationStore<B::Optimization>, stream: &Stream<B>) {
        self.explorer.reset(&stream.relative);
        self.policy.reset();

        // Reset the policy state.
        for i in 0..stream.relative.len() {
            self.policy.update(store, &stream.relative[i]);
        }
    }

    fn action(
        &mut self,
        store: &ExplorationStore<B::Optimization>,
        stream: &Stream<B>,
        mode: ExecutionMode,
    ) -> Action {
        if let ExecutionMode::Lazy = mode {
            // We update the policy in lazy mode, since
            self.policy.update(
                store,
                &stream
                    .relative
                    .last()
                    .expect("At least on operation in the stream."),
            );
        };

        self.policy.action(store, stream.relative.as_slice(), mode)
    }

    fn on_optimization_found(
        policy: &Policy<B::Optimization>,
        stream: &Stream<B>,
        store: &mut ExplorationStore<B::Optimization>,
        builder: &dyn OptimizationBuilder<B::Optimization>,
        mode: ExecutionMode,
    ) -> ExplorationId {
        let num_fused = builder.len();
        let relative = &stream.relative[0..num_fused];

        match mode {
            ExecutionMode::Lazy => {
                let next_ops = &stream.relative[num_fused];
                let criterion = StopCriterion::OnOperation(next_ops.clone());

                match policy.action(store, relative, ExecutionMode::Sync) {
                    Action::Execute(id) => {
                        store.add_stop_criterion(id, criterion);
                        id
                    }
                    _ => store.add(Exploration {
                        stream: relative.to_vec(),
                        criteria: vec![criterion],
                        execution: ExecutionStrategy::Optimization(builder.build()),
                    }),
                }
            }
            ExecutionMode::Sync => match policy.action(store, &relative, ExecutionMode::Sync) {
                Action::Execute(id) => {
                    store.add_stop_criterion(id, StopCriterion::OnSync);
                    id
                }
                _ => store.add(Exploration {
                    stream: stream.relative.clone(),
                    criteria: vec![StopCriterion::OnSync],
                    execution: ExecutionStrategy::Optimization(builder.build()),
                }),
            },
        }
    }

    fn on_optimization_not_found(
        policy: &Policy<B::Optimization>,
        stream: &Stream<B>,
        store: &mut ExplorationStore<B::Optimization>,
        mode: ExecutionMode,
        num_explored: usize,
    ) -> ExplorationId {
        let relative = &stream.relative[0..num_explored];
        let criterion = match mode {
            ExecutionMode::Lazy => StopCriterion::Always,
            ExecutionMode::Sync => StopCriterion::OnSync,
        };

        match policy.action(store, &relative, ExecutionMode::Sync) {
            Action::Execute(id) => {
                store.add_stop_criterion(id, criterion);
                id
            }
            _ => store.add(Exploration {
                stream: relative.to_vec(),
                criteria: vec![criterion],
                execution: ExecutionStrategy::Operations,
            }),
        }
    }
}

#[cfg(test)]
mod tests {}
