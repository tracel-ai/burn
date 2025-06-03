use burn_ir::HandleContainer;

use crate::{
    FusionRuntime, NumOperations, Optimization,
    stream::{
        Context, OperationConverter, OperationQueue, RelativeOps,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStep, ExecutionStrategy},
    },
};

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

enum Execution<'a, R: FusionRuntime> {
    Single {
        handles: &'a mut HandleContainer<R::FusionHandle>,
        converter: &'a mut OperationConverter,
        operations: Vec<Option<Box<dyn Operation<R>>>>,
        ordering: Vec<usize>,
        num_drained: usize,
    },
    Multiple {
        context: &'a mut Context<'a, R::FusionHandle>,
        operations: Vec<Option<Box<dyn Operation<R>>>>,
        ordering: Vec<usize>,
        num_drained: usize,
    },
}

impl<'a, R: FusionRuntime> Execution<'a, R> {
    fn execute(
        step: &mut ExecutionStep<R::Optimization>,
        converter: &'a mut OperationConverter,
        handles: &'a mut HandleContainer<R::FusionHandle>,
        operations: Vec<Box<dyn Operation<R>>>,
    ) -> (Vec<Box<dyn Operation<R>>>, usize) {
        let ordering = step.ordering.clone();
        let operations = operations.into_iter().map(|a| Some(a)).collect();

        if matches!(&step.strategy, ExecutionStrategy::Composed(..)) {
            let mut context = converter.context(handles);
            let mut this = Execution::Multiple {
                context: &mut context,
                operations,
                ordering,
                num_drained: 0,
            };

            this = this.execute_strategy(&mut step.strategy);

            match this {
                Execution::Multiple {
                    num_drained,
                    mut operations,
                    ..
                } => {
                    let operations = operations
                        .drain(num_drained..)
                        .map(|a| a.expect("Should not be consumed"))
                        .collect();
                    (operations, num_drained)
                }
                _ => unreachable!(),
            }
        } else {
            let mut this = Execution::Single {
                handles,
                converter,
                ordering,
                operations,
                num_drained: 0,
            };
            this = this.execute_strategy(&mut step.strategy);

            match this {
                Execution::Single {
                    mut operations,
                    num_drained,
                    ..
                } => {
                    let operations = operations
                        .drain(num_drained..)
                        .into_iter()
                        .map(|a| a.expect("Should not be consumed"))
                        .collect();
                    (operations, num_drained)
                }
                _ => unreachable!(),
            }
        }
    }

    fn execute_strategy(mut self, strategy: &mut ExecutionStrategy<R::Optimization>) -> Self {
        match &mut self {
            Execution::Single {
                handles,
                operations,
                converter,
                num_drained,
                ordering,
            } => match strategy {
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    let mut context = converter.context(handles);
                    *num_drained += Self::execute_optimization_with_fallbacks(
                        opt,
                        &mut context,
                        operations,
                        fallbacks,
                        ordering,
                    );
                }
                ExecutionStrategy::Optimization(opt) => {
                    let mut context = converter.context(handles);
                    *num_drained +=
                        Self::execute_optimization(opt, &mut context, operations, ordering);
                }
                ExecutionStrategy::Operations(size) => {
                    Self::execute_operations(handles, operations, *size, ordering);
                    *num_drained += *size;
                }
                ExecutionStrategy::Composed(_) => unreachable!(),
            },
            Execution::Multiple {
                context,
                operations,
                num_drained,
                ordering,
            } => match strategy {
                ExecutionStrategy::Optimization(opt) => {
                    *num_drained += Self::execute_optimization(opt, context, operations, ordering);
                }
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    *num_drained += Self::execute_optimization_with_fallbacks(
                        opt, context, operations, fallbacks, ordering,
                    );
                }
                ExecutionStrategy::Operations(size) => {
                    Self::execute_operations(&mut context.handles, operations, *size, ordering);
                    *num_drained += *size;
                }
                ExecutionStrategy::Composed(items) => {
                    for item in items.iter_mut() {
                        self = self.execute_strategy(item);
                    }
                }
            },
        };
        self
    }
    fn execute_optimization(
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        operations: &mut Vec<Option<Box<dyn Operation<R>>>>,
        ordering: &mut Vec<usize>,
    ) -> usize {
        let num_drained = optimization.len();
        optimization.execute(context, operations, ordering);

        for i in 0..num_drained {
            let index = ordering[i];
            let _ = operations[index].take().unwrap();
        }

        ordering.drain(0..num_drained);
        num_drained
    }

    fn execute_optimization_with_fallbacks(
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        operations: &mut Vec<Option<Box<dyn Operation<R>>>>,
        fallbacks: &mut Vec<usize>, // TODO: Check maybe if can be out of order.
        ordering: &mut Vec<usize>,
    ) -> usize {
        let num_drained = optimization.len() + fallbacks.len();

        optimization.execute(context, operations, &ordering);

        for i in 0..num_drained {
            let index = ordering[i];
            let op = operations[index].take().unwrap();
            if fallbacks.contains(&i) {
                op.execute(context.handles);
            }
        }

        ordering.drain(0..num_drained);

        num_drained
    }
    fn execute_operations(
        handles: &mut HandleContainer<R::FusionHandle>,
        operations: &mut Vec<Option<Box<dyn Operation<R>>>>,
        size: usize,
        ordering: &mut Vec<usize>,
    ) {
        for i in 0..size {
            let index = ordering[i];
            let op = operations[index].take().unwrap();
            op.execute(handles);
        }
        ordering.drain(0..size);
    }
}

/// General trait to abstract how a single operation is executed.
pub trait Operation<R: FusionRuntime>: Send + Sync {
    /// Execute the operation.
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>);
}

impl<R: FusionRuntime> OperationQueue<R> {
    /// Execute the queue partially following the execution strategy from the plan.
    pub(crate) fn execute(
        &mut self,
        id: ExecutionPlanId,
        handles: &mut HandleContainer<R::FusionHandle>,
        store: &mut ExecutionPlanStore<R::Optimization>,
    ) {
        let plan = store.get_mut_unchecked(id);
        self.execute_step(&mut plan.execution, handles);
    }

    fn execute_step(
        &mut self,
        step: &mut ExecutionStep<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let mut operations = Vec::new();
        core::mem::swap(&mut operations, &mut self.operations);
        let (operations, num_drained) =
            Execution::execute(step, &mut self.converter, handles, operations);

        self.operations = operations;
        self.drain_queue(num_drained, handles);
    }

    /// Bookkeeping after executing `num_drained` operations from the queue.
    fn drain_queue(&mut self, num_drained: usize, handles: &mut HandleContainer<R::FusionHandle>) {
        self.global[0..num_drained]
            .iter()
            .flat_map(|desc| desc.nodes())
            .for_each(|tensor| handles.free(tensor));

        self.global.drain(0..num_drained);
        self.reset_relative();
        assert_eq!(self.global.len(), self.operations.len());
    }

    fn reset_relative(&mut self) {
        self.relative.clear();
        self.converter.clear();

        for node in self.global.iter() {
            let relative = node.to_relative(&mut self.converter);
            self.relative.push(relative);
        }
    }
}
