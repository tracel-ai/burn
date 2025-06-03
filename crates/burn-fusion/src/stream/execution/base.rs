use burn_ir::HandleContainer;

use crate::{
    FusionRuntime, NumOperations, Optimization,
    stream::{
        Context, OperationConverter, OperationQueue, RelativeOps,
        store::{ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy},
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
        operations: Vec<Box<dyn Operation<R>>>,
        num_drained: usize,
    },
    Multiple {
        context: &'a mut Context<'a, R::FusionHandle>,
        operations: Vec<Box<dyn Operation<R>>>,
        num_drained: usize,
    },
}

impl<'a, R: FusionRuntime> Execution<'a, R> {
    fn execute(
        strategy: &mut ExecutionStrategy<R::Optimization>,
        converter: &'a mut OperationConverter,
        handles: &'a mut HandleContainer<R::FusionHandle>,
        operations: Vec<Box<dyn Operation<R>>>,
    ) -> (Vec<Box<dyn Operation<R>>>, usize) {
        if matches!(&strategy, ExecutionStrategy::Composed(..)) {
            let mut context = converter.context(handles);
            let mut this = Execution::Multiple {
                context: &mut context,
                operations,
                num_drained: 0,
            };

            this = this.execute_step(strategy);

            match this {
                Execution::Multiple {
                    num_drained,
                    operations,
                    ..
                } => (operations, num_drained),
                _ => unreachable!(),
            }
        } else {
            let mut this = Execution::Single {
                handles,
                converter,
                operations,
                num_drained: 0,
            };
            this = this.execute_step(strategy);

            match this {
                Execution::Single {
                    operations,
                    num_drained,
                    ..
                } => (operations, num_drained),
                _ => unreachable!(),
            }
        }
    }

    fn execute_step(mut self, strategy: &mut ExecutionStrategy<R::Optimization>) -> Self {
        match &mut self {
            Execution::Single {
                handles,
                operations,
                converter,
                num_drained,
            } => match strategy {
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    let mut context = converter.context(handles);
                    *num_drained += Self::execute_optimization_with_fallbacks(
                        opt,
                        &mut context,
                        operations,
                        fallbacks,
                    );
                }
                ExecutionStrategy::Optimization(opt) => {
                    let mut context = converter.context(handles);
                    *num_drained += Self::execute_optimization(opt, &mut context, operations);
                }
                ExecutionStrategy::Operations(size) => {
                    Self::execute_operations(handles, operations, *size);
                    *num_drained += *size;
                }
                ExecutionStrategy::Composed(_) => unreachable!(),
            },
            Execution::Multiple {
                context,
                operations,
                num_drained,
            } => match strategy {
                ExecutionStrategy::Optimization(opt) => {
                    *num_drained += Self::execute_optimization(opt, context, operations);
                }
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    *num_drained += Self::execute_optimization_with_fallbacks(
                        opt, context, operations, fallbacks,
                    );
                }
                ExecutionStrategy::Operations(size) => {
                    Self::execute_operations(&mut context.handles, operations, *size);
                    *num_drained += *size;
                }
                ExecutionStrategy::Composed(items) => {
                    for item in items.iter_mut() {
                        self = self.execute_step(item);
                    }
                }
            },
        };
        self
    }
    fn execute_optimization(
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        operations: &mut Vec<Box<dyn Operation<R>>>,
    ) -> usize {
        let num_drained = optimization.len();
        optimization.execute(context, operations);
        operations.drain(0..num_drained);
        num_drained
    }

    fn execute_optimization_with_fallbacks(
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        operations: &mut Vec<Box<dyn Operation<R>>>,
        fallbacks: &mut Vec<usize>,
    ) -> usize {
        let num_drained = optimization.len() + fallbacks.len();

        optimization.execute(context, operations);

        for (i, op) in operations.drain(0..num_drained).enumerate() {
            if fallbacks.contains(&i) {
                op.execute(context.handles);
            }
        }
        num_drained
    }
    fn execute_operations(
        handles: &mut HandleContainer<R::FusionHandle>,
        operations: &mut Vec<Box<dyn Operation<R>>>,
        size: usize,
    ) {
        for operation in operations.drain(0..size) {
            operation.execute(handles);
        }
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
        self.execute_strategy(&mut plan.strategy, handles);
    }

    fn execute_strategy(
        &mut self,
        strategy: &mut ExecutionStrategy<R::Optimization>,
        handles: &mut HandleContainer<R::FusionHandle>,
    ) {
        let mut operations = Vec::new();
        core::mem::swap(&mut operations, &mut self.operations);
        let (operations, num_drained) =
            Execution::execute(strategy, &mut self.converter, handles, operations);

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
