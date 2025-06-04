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

/// Manage the execution of potentially multiple optimizations and operations out of order.
pub struct OrderedExecution<R: FusionRuntime> {
    operations: Vec<Box<dyn Operation<R>>>,
    ordering: Vec<usize>,
    cursor: usize,
}

impl<R: FusionRuntime> OrderedExecution<R> {
    fn new(operations: Vec<Box<dyn Operation<R>>>, ordering: Vec<usize>) -> Self {
        Self {
            operations,
            ordering,
            cursor: 0,
        }
    }

    fn finish(mut self) -> (Vec<Box<dyn Operation<R>>>, usize) {
        println!("Executed {:?}", &self.ordering[0..self.cursor]);
        self.operations.drain(0..self.cursor);
        (self.operations, self.cursor)
    }

    fn execute_optimization(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
    ) {
        let num_drained = optimization.len();
        optimization.execute(context, self);
        self.cursor += num_drained;
    }

    /// Returns the operation that can be executed without impacting the state of the execution.
    ///
    /// This is usefull to implement fallback for optimizations.
    pub fn operation_within_optimization(&self, index: usize) -> &Box<dyn Operation<R>> {
        let position = self.cursor + index;
        let index = self.ordering[position];
        &self.operations[index]
    }

    fn execute_optimization_with_fallbacks(
        &mut self,
        optimization: &mut R::Optimization,
        context: &mut Context<'_, R::FusionHandle>,
        fallbacks: &mut Vec<usize>,
    ) {
        let num_drained = optimization.len() + fallbacks.len();

        optimization.execute(context, self);

        for _ in 0..num_drained {
            let index = self.ordering[self.cursor];

            if fallbacks.contains(&self.cursor) {
                let op = &self.operations[index];
                op.execute(context.handles);
            }
            self.cursor += 1;
        }
    }
    fn execute_operations(&mut self, handles: &mut HandleContainer<R::FusionHandle>, size: usize) {
        for _ in 0..size {
            let index = self.ordering[self.cursor];
            let op = &self.operations[index];
            op.execute(handles);
            self.cursor += 1;
        }
    }
}

enum Execution<'a, R: FusionRuntime> {
    Single {
        handles: &'a mut HandleContainer<R::FusionHandle>,
        converter: &'a mut OperationConverter,
        execution: OrderedExecution<R>,
    },
    Multiple {
        context: &'a mut Context<'a, R::FusionHandle>,
        execution: OrderedExecution<R>,
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
        let execution = OrderedExecution::new(operations, ordering);

        if matches!(&step.strategy, ExecutionStrategy::Composed(..)) {
            let mut context = converter.context(handles);
            let mut this = Execution::Multiple {
                context: &mut context,
                execution,
            };

            this = this.execute_strategy(&mut step.strategy);

            match this {
                Execution::Multiple { execution, .. } => execution.finish(),
                _ => unreachable!(),
            }
        } else {
            let mut this = Execution::Single {
                handles,
                converter,
                execution,
            };
            this = this.execute_strategy(&mut step.strategy);

            match this {
                Execution::Single { execution, .. } => execution.finish(),
                _ => unreachable!(),
            }
        }
    }

    fn execute_strategy(mut self, strategy: &mut ExecutionStrategy<R::Optimization>) -> Self {
        match &mut self {
            Execution::Single {
                handles,
                converter,
                execution,
            } => match strategy {
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    let mut context = converter.context(handles);
                    execution.execute_optimization_with_fallbacks(opt, &mut context, fallbacks)
                }
                ExecutionStrategy::Optimization(opt) => {
                    let mut context = converter.context(handles);
                    execution.execute_optimization(opt, &mut context)
                }
                ExecutionStrategy::Operations(size) => execution.execute_operations(handles, *size),
                ExecutionStrategy::Composed(_) => unreachable!(),
            },
            Execution::Multiple { context, execution } => match strategy {
                ExecutionStrategy::Optimization(opt) => {
                    execution.execute_optimization(opt, context);
                }
                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks) => {
                    execution.execute_optimization_with_fallbacks(opt, context, fallbacks);
                }
                ExecutionStrategy::Operations(size) => {
                    execution.execute_operations(&mut context.handles, *size);
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
