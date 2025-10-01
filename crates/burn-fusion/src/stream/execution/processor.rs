use burn_ir::OperationIr;

use super::{ExecutionMode, ExplorationAction, Explorer};
use crate::search::BlockOptimization;
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{ExecutionPlan, ExecutionPlanId, ExecutionPlanStore, ExecutionTrigger};
use crate::{NumOperations, OptimizationBuilder};

/// Process a [stream segment](StreamSegment) following a [policy](Policy).
pub(crate) struct Processor<O> {
    policy: Policy<O>,
    explorer: Explorer<O>,
}

/// A part of a stream that can be executed partially using [execution plan](ExecutionPlan).
pub(crate) trait StreamSegment<O> {
    /// The operations in the segment.
    fn operations(&self) -> &[OperationIr];
    /// Execute part of the segment using the given plan id.
    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<O>);
}

impl<O: NumOperations> Processor<O> {
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
        // We assume that we always register a new operation in lazy mode.
        if let ExecutionMode::Lazy = mode {
            self.on_new_operation(&segment, store);
        }

        loop {
            if segment.operations().is_empty() {
                break;
            }

            let action = self.policy.action(store, segment.operations(), mode);

            match action {
                Action::Explore => {
                    self.explore(&mut segment, store, mode);

                    if self.explorer.is_up_to_date() {
                        break;
                    }
                }
                Action::Defer => {
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
        }
    }

    fn on_new_operation<Segment>(&mut self, segment: &Segment, store: &mut ExecutionPlanStore<O>)
    where
        Segment: StreamSegment<O>,
    {
        self.policy.update(
            store,
            segment
                .operations()
                .last()
                .expect("At least one operation in the operation list."),
        );
        self.explorer.on_new_operation();
    }

    fn explore<Item: StreamSegment<O>>(
        &mut self,
        item: &mut Item,
        store: &mut ExecutionPlanStore<O>,
        mode: ExecutionMode,
    ) {
        match self.explorer.explore(item.operations(), mode) {
            ExplorationAction::Completed(optim) => {
                let id = Self::on_exploration_completed(
                    &self.policy,
                    item.operations(),
                    store,
                    optim,
                    mode,
                );
                item.execute(id, store);
                self.reset(store, item.operations());
            }
            ExplorationAction::Continue => {
                if let ExecutionMode::Sync = mode {
                    panic!("Can't continue exploring when sync.")
                }
            }
        }
    }

    fn reset(&mut self, store: &mut ExecutionPlanStore<O>, operations: &[OperationIr]) {
        self.explorer.reset(operations);
        self.policy.reset();

        // Reset the policy state with the remaining operations
        for operation in operations.iter() {
            self.policy.update(store, operation);
        }
    }

    /// We found an optimization (i.e. a new execution plan).
    /// Cache it in the store.
    fn on_exploration_completed(
        policy: &Policy<O>,
        operations: &[OperationIr],
        store: &mut ExecutionPlanStore<O>,
        optimization: BlockOptimization<O>,
        mode: ExecutionMode,
    ) -> ExecutionPlanId {
        let num_optimized = optimization.ordering.len();
        let relative = &operations[0..num_optimized];

        match mode {
            ExecutionMode::Lazy => {
                let next_ops = &operations[num_optimized..operations.len()];

                let trigger = if next_ops.is_empty() {
                    // Happens if the next ops is included in the fused operation, and there is no
                    // way the builder can still continue fusing.
                    ExecutionTrigger::Always
                } else {
                    ExecutionTrigger::OnOperations(next_ops.to_vec())
                };

                match policy.action(store, relative, ExecutionMode::Sync) {
                    Action::Execute(id) => {
                        store.add_trigger(id, trigger);
                        id
                    }
                    _ => {
                        let plan = ExecutionPlan {
                            operations: relative.to_vec(),
                            triggers: vec![trigger],
                            optimization,
                        };
                        store.add(plan)
                    }
                }
            }
            ExecutionMode::Sync => match policy.action(store, relative, ExecutionMode::Sync) {
                Action::Execute(id) => {
                    store.add_trigger(id, ExecutionTrigger::OnSync);
                    id
                }
                _ => {
                    let plan = ExecutionPlan {
                        operations: relative.to_vec(),
                        triggers: vec![ExecutionTrigger::OnSync],
                        optimization,
                    };
                    store.add(plan)
                }
            },
        }
    }
}
