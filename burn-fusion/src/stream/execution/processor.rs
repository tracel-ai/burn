use super::{ExecutionMode, Explorer, ExplorerResult};
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{
    ExecutionStrategy, Exploration, ExplorationId, ExplorationStore, StopCriterion,
};
use crate::stream::OperationDescription;
use crate::OptimizationBuilder;

/// Process a [stream segment](StreamSegment) following a [policy](Policy).
///
/// Explore and create new [optimizations](crate::Optimization) using [explorations](Exploration).
pub(crate) struct Processor<O> {
    policy: Policy<O>,
    explorer: Explorer<O>,
}

/// A part of a stream that can be executed partially using [explorations](Exploration).
pub trait StreamSegment<O> {
    /// The operations in the segment.
    fn operations<'a>(&'a self) -> &[OperationDescription];
    /// Execute part of the segment using the given exploration id.
    fn execute(&mut self, id: ExplorationId, store: &mut ExplorationStore<O>);
}

impl<O> Processor<O> {
    /// Create a new stream processor.
    pub fn new(optimizations: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            policy: Policy::new(),
            explorer: Explorer::new(optimizations),
        }
    }

    /// Process the [stream segment](StreamSegment) with the provided [mode](ExecutionMode).
    pub fn process<Segment>(
        &mut self,
        mut segment: Segment,
        store: &mut ExplorationStore<O>,
        mode: ExecutionMode,
    ) where
        Segment: StreamSegment<O>,
    {
        loop {
            if segment.operations().is_empty() {
                break;
            }

            match self.action(store, segment.operations(), mode) {
                Action::Explore => {
                    self.explore(&mut segment, store, mode);

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
                    if let ExecutionMode::Sync = mode {
                        store.add_stop_criterion(id, StopCriterion::OnSync);
                    }

                    segment.execute(id, store);
                    self.reset(store, segment.operations());
                }
            };

            if let ExecutionMode::Lazy = mode {
                break;
            }
        }
    }

    fn explore<Item: StreamSegment<O>>(
        &mut self,
        item: &mut Item,
        store: &mut ExplorationStore<O>,
        mode: ExecutionMode,
    ) {
        match self.explorer.explore(&item.operations(), mode) {
            ExplorerResult::Found(optim) => {
                let id = Self::on_optimization_found(
                    &self.policy,
                    item.operations(),
                    store,
                    optim,
                    mode,
                );
                item.execute(id, store);
                self.reset(store, item.operations());
            }
            ExplorerResult::NotFound { num_explored } => {
                let id = Self::on_optimization_not_found(
                    &self.policy,
                    item.operations(),
                    store,
                    mode,
                    num_explored,
                );
                item.execute(id, store);
                self.reset(store, item.operations());
            }
            ExplorerResult::Continue => {
                if let ExecutionMode::Sync = mode {
                    panic!("Can't continue exploring when sync.")
                }
            }
        }
    }

    fn reset(&mut self, store: &mut ExplorationStore<O>, stream: &[OperationDescription]) {
        self.explorer.reset(stream);
        self.policy.reset();

        // Reset the policy state.
        for i in 0..stream.len() {
            self.policy.update(store, &stream[i]);
        }
    }

    fn action(
        &mut self,
        store: &ExplorationStore<O>,
        stream: &[OperationDescription],
        mode: ExecutionMode,
    ) -> Action {
        if let ExecutionMode::Lazy = mode {
            // We update the policy in lazy mode, since
            self.policy.update(
                store,
                &stream.last().expect("At least on operation in the stream."),
            );
        };

        self.policy.action(store, stream, mode)
    }

    fn on_optimization_found(
        policy: &Policy<O>,
        stream: &[OperationDescription],
        store: &mut ExplorationStore<O>,
        builder: &dyn OptimizationBuilder<O>,
        mode: ExecutionMode,
    ) -> ExplorationId {
        let num_fused = builder.len();
        let relative = &stream[0..num_fused];

        match mode {
            ExecutionMode::Lazy => {
                let next_ops = stream.get(num_fused);

                let criterion = if let Some(next_ops) = next_ops {
                    StopCriterion::OnOperation(next_ops.clone())
                } else {
                    // Happens if the next ops is included in the fused operation, and there is no
                    // way the builder can still continue fusing.
                    StopCriterion::Always
                };

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
                    stream: stream.to_vec(),
                    criteria: vec![StopCriterion::OnSync],
                    execution: ExecutionStrategy::Optimization(builder.build()),
                }),
            },
        }
    }

    fn on_optimization_not_found(
        policy: &Policy<O>,
        stream: &[OperationDescription],
        store: &mut ExplorationStore<O>,
        mode: ExecutionMode,
        num_explored: usize,
    ) -> ExplorationId {
        let relative = &stream[0..num_explored];
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
