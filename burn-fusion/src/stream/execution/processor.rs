use super::{ExecutionMode, Exploration, Explorer};
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{
    ExecutionPlan, ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy, ExecutionTrigger,
};
use crate::stream::OperationDescription;
use crate::OptimizationBuilder;

/// Process a [stream segment](StreamSegment) following a [policy](Policy).
pub(crate) struct Processor<O> {
    policy: Policy<O>,
    explorer: Explorer<O>,
}

/// A part of a stream that can be executed partially using [execution plan](ExecutionPlan).
pub(crate) trait StreamSegment<O> {
    /// The operations in the segment.
    fn operations(&self) -> &[OperationDescription];
    /// Execute part of the segment using the given plan id.
    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<O>);
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
        store: &mut ExecutionPlanStore<O>,
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

                    if self.explorer.is_up_to_date() {
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
                        store.add_trigger(id, ExecutionTrigger::OnSync);
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
        store: &mut ExecutionPlanStore<O>,
        mode: ExecutionMode,
    ) {
        match self.explorer.explore(item.operations(), mode) {
            Exploration::Found(optim) => {
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
            Exploration::NotFound { num_explored } => {
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
            Exploration::Continue => {
                if let ExecutionMode::Sync = mode {
                    panic!("Can't continue exploring when sync.")
                }
            }
        }
    }

    fn reset(&mut self, store: &mut ExecutionPlanStore<O>, operations: &[OperationDescription]) {
        self.explorer.reset(operations);
        self.policy.reset();

        // Reset the policy state.
        for operation in operations.iter() {
            self.policy.update(store, operation);
        }
    }

    fn action(
        &mut self,
        store: &ExecutionPlanStore<O>,
        operations: &[OperationDescription],
        mode: ExecutionMode,
    ) -> Action {
        if let ExecutionMode::Lazy = mode {
            // We update the policy in lazy mode, since
            self.policy.update(
                store,
                operations
                    .last()
                    .expect("At least one operation in the operation list."),
            );
        };

        self.policy.action(store, operations, mode)
    }

    fn on_optimization_found(
        policy: &Policy<O>,
        operations: &[OperationDescription],
        store: &mut ExecutionPlanStore<O>,
        builder: &dyn OptimizationBuilder<O>,
        mode: ExecutionMode,
    ) -> ExecutionPlanId {
        let num_fused = builder.len();
        let relative = &operations[0..num_fused];

        match mode {
            ExecutionMode::Lazy => {
                let next_ops = operations.get(num_fused);

                let trigger = if let Some(next_ops) = next_ops {
                    ExecutionTrigger::OnOperation(next_ops.clone())
                } else {
                    // Happens if the next ops is included in the fused operation, and there is no
                    // way the builder can still continue fusing.
                    ExecutionTrigger::Always
                };

                match policy.action(store, relative, ExecutionMode::Sync) {
                    Action::Execute(id) => {
                        store.add_trigger(id, trigger);
                        id
                    }
                    _ => store.add(ExecutionPlan {
                        operations: relative.to_vec(),
                        triggers: vec![trigger],
                        strategy: ExecutionStrategy::Optimization(builder.build()),
                    }),
                }
            }
            ExecutionMode::Sync => match policy.action(store, relative, ExecutionMode::Sync) {
                Action::Execute(id) => {
                    store.add_trigger(id, ExecutionTrigger::OnSync);
                    id
                }
                _ => store.add(ExecutionPlan {
                    operations: operations.to_vec(),
                    triggers: vec![ExecutionTrigger::OnSync],
                    strategy: ExecutionStrategy::Optimization(builder.build()),
                }),
            },
        }
    }

    fn on_optimization_not_found(
        policy: &Policy<O>,
        operations: &[OperationDescription],
        store: &mut ExecutionPlanStore<O>,
        mode: ExecutionMode,
        num_explored: usize,
    ) -> ExecutionPlanId {
        let relative = &operations[0..num_explored];
        let trigger = match mode {
            ExecutionMode::Lazy => ExecutionTrigger::Always,
            ExecutionMode::Sync => ExecutionTrigger::OnSync,
        };

        match policy.action(store, relative, ExecutionMode::Sync) {
            Action::Execute(id) => {
                store.add_trigger(id, trigger);
                id
            }
            _ => store.add(ExecutionPlan {
                operations: relative.to_vec(),
                triggers: vec![trigger],
                strategy: ExecutionStrategy::Operations,
            }),
        }
    }
}
