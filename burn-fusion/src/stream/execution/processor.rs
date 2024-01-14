use super::{ExecutionMode, Explorer, ExplorerResult};
use crate::stream::execution::{Action, Policy};
use crate::stream::store::{
    ExecutionStrategy, Exploration, ExplorationId, ExplorationStore, StopCriterion,
};
use crate::stream::OperationDescription;
use crate::OptimizationBuilder;

/// Process the [stream](Stream) following a [policy](Policy).
///
/// Explore and create new optimizations using explorations
pub(crate) struct Processor<O> {
    policy: Policy<O>,
    explorer: Explorer<O>,
}

pub trait Process<O> {
    fn operations<'a>(&'a self) -> &[OperationDescription];
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

    /// Process the [stream](Stream) with the provided mode.
    pub fn process<P: Process<O>>(
        &mut self,
        mut process: P,
        store: &mut ExplorationStore<O>,
        mode: ExecutionMode,
    ) {
        loop {
            if process.operations().is_empty() {
                break;
            }

            match self.action(store, process.operations(), mode) {
                Action::Explore => {
                    println!("Action::Explore");
                    self.explore(&mut process, store, mode);

                    if self.explorer.up_to_date() {
                        break;
                    }
                }
                Action::Defer => {
                    println!("Action::Defer");
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

                    process.execute(id, store);
                    self.reset(store, process.operations());
                }
            };

            if let ExecutionMode::Lazy = mode {
                break;
            }
        }
    }

    fn explore<Item: Process<O>>(
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
        println!("Num fused {num_fused}");
        println!("Stream size {}", stream.len());
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

#[cfg(test)]
mod tests {
    use crate::{
        stream::{
            BinaryOperationDescription, NumericOperationDescription, ScalarOperationDescription,
        },
        OptimizationProperties, OptimizationStatus, TensorDescription, TensorId, TensorStatus,
    };

    use super::*;

    struct TestStream {
        processor: Processor<TestOptimization>,
        store: ExplorationStore<TestOptimization>,
        executed: Vec<ExplorationId>,
        operations: Vec<OperationDescription>,
    }

    impl TestStream {
        fn new(optimizations: Vec<Box<dyn OptimizationBuilder<TestOptimization>>>) -> Self {
            Self {
                processor: Processor::<TestOptimization>::new(optimizations),
                store: ExplorationStore::<TestOptimization>::new(),
                executed: Vec::new(),
                operations: Vec::new(),
            }
        }

        fn add(&mut self, operation: OperationDescription) {
            self.operations.push(operation);
            self.processor.process(
                TestProcess::new(&mut self.operations, &mut self.executed),
                &mut self.store,
                ExecutionMode::Lazy,
            );
        }

        fn sync(&mut self) {
            self.processor.process(
                TestProcess::new(&mut self.operations, &mut self.executed),
                &mut self.store,
                ExecutionMode::Sync,
            );
        }

        fn assert_executed_exploration(
            &self,
            id: ExplorationId,
            expected: Exploration<TestOptimization>,
        ) {
            let actual = self.store.get_unchecked(id);
            assert_eq!(actual.criteria, expected.criteria,);
            assert_eq!(actual.stream, expected.stream,);
        }
    }

    #[test]
    fn scenario1() {
        let builder_id_1 = 0;
        let builder_id_2 = 1;
        let exploration_id_1 = 0;
        let exploration_id_2 = 1;
        let exploration_id_3 = 2;

        let builder_1 = TestBuilder::new(
            builder_id_1,
            vec![operation_1(), operation_2()],
            StopCriterion::Always,
        );
        let builder_2 = TestBuilder::new(
            builder_id_2,
            vec![operation_2(), operation_2()],
            StopCriterion::OnOperation(operation_1()),
        );
        let mut stream = TestStream::new(vec![Box::new(builder_1), Box::new(builder_2)]);

        // No optimization found.
        stream.add(operation_1());
        stream.add(operation_1());
        stream.assert_executed_exploration(
            exploration_id_1,
            Exploration {
                stream: vec![operation_1(), operation_1()],
                criteria: vec![StopCriterion::Always],
                execution: ExecutionStrategy::Operations,
            },
        );
        assert!(stream.operations.is_empty());
        println!("0000000000000000");

        // Optimization found.
        stream.add(operation_1());
        stream.add(operation_2());
        stream.assert_executed_exploration(
            exploration_id_2,
            Exploration {
                stream: vec![operation_1(), operation_2()],
                criteria: vec![StopCriterion::Always],
                execution: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
            },
        );
        assert!(stream.operations.is_empty());
        println!("1111111111111111");

        // Optimization found.
        stream.add(operation_2());
        stream.add(operation_2());
        stream.add(operation_1());
        stream.assert_executed_exploration(
            exploration_id_3,
            Exploration {
                stream: vec![operation_2(), operation_2()],
                criteria: vec![StopCriterion::OnOperation(operation_1())],
                execution: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_2, 2)),
            },
        );
        assert_eq!(stream.operations.len(), 1);
        println!("2222222222222222");

        // Optimization found.
        stream.add(operation_2());
        stream.assert_executed_exploration(
            exploration_id_2,
            Exploration {
                stream: vec![operation_1(), operation_2()],
                criteria: vec![StopCriterion::Always],
                execution: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
            },
        );
        assert_eq!(stream.operations.len(), 0);
        println!("3333333333333333");

        stream.add(operation_2());
        stream.add(operation_2());
        stream.sync();
        stream.assert_executed_exploration(
            exploration_id_3,
            Exploration {
                stream: vec![operation_2(), operation_2()],
                criteria: vec![
                    StopCriterion::OnOperation(operation_1()),
                    StopCriterion::OnSync,
                ],
                execution: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_2, 2)),
            },
        );
    }

    struct TestBuilder {
        builder_id: usize,
        expected_operations: Vec<OperationDescription>,
        expected_criterion: StopCriterion,
        actual: Vec<OperationDescription>,
    }

    impl TestBuilder {
        pub fn new(
            builder_id: usize,
            operations: Vec<OperationDescription>,
            criteria: StopCriterion,
        ) -> Self {
            Self {
                builder_id,
                expected_operations: operations,
                actual: Vec::new(),
                expected_criterion: criteria,
            }
        }
    }

    impl OptimizationBuilder<TestOptimization> for TestBuilder {
        fn register(&mut self, operation: &OperationDescription) {
            self.actual.push(operation.clone());
        }

        fn build(&self) -> TestOptimization {
            TestOptimization::new(self.builder_id, self.len())
        }

        fn reset(&mut self) {
            self.actual.clear();
        }

        fn status(&self) -> OptimizationStatus {
            let actual_equal_expected = self.actual == self.expected_operations;

            if self.actual.len() < self.expected_operations.len() {
                let operations = &self.expected_operations[0..self.actual.len()];
                let is_contained = &self.actual == operations;

                if is_contained {
                    return OptimizationStatus::Open;
                } else {
                    return OptimizationStatus::Closed;
                }
            }

            if self.actual.len() == self.expected_operations.len() {
                if actual_equal_expected {
                    if let StopCriterion::Always = self.expected_criterion {
                        return OptimizationStatus::Closed;
                    } else {
                        return OptimizationStatus::Open;
                    }
                } else {
                    return OptimizationStatus::Closed;
                }
            }

            OptimizationStatus::Closed
        }

        fn properties(&self) -> OptimizationProperties {
            if self.actual.len() < self.expected_operations.len() {
                return OptimizationProperties {
                    score: 0,
                    ready: false,
                };
            }

            let stream_is_ok =
                &self.actual[0..self.expected_operations.len()] == &self.expected_operations;

            if !stream_is_ok {
                return OptimizationProperties {
                    score: 0,
                    ready: false,
                };
            }

            if let StopCriterion::OnOperation(next_ops) = &self.expected_criterion {
                if let Some(op) = self.actual.get(self.expected_operations.len()) {
                    if op != next_ops {
                        return OptimizationProperties {
                            score: 0,
                            ready: false,
                        };
                    }
                }
            };

            OptimizationProperties {
                score: 1,
                ready: true,
            }
        }

        fn len(&self) -> usize {
            self.expected_operations.len()
        }
    }

    #[derive(new, Debug, PartialEq)]
    struct TestOptimization {
        builder_id: usize,
        size: usize,
    }

    #[derive(new)]
    struct TestProcess<'i> {
        operations: &'i mut Vec<OperationDescription>,
        executed: &'i mut Vec<ExplorationId>,
    }

    impl<'i> Process<TestOptimization> for TestProcess<'i> {
        fn operations<'a>(&'a self) -> &[OperationDescription] {
            &self.operations
        }

        fn execute(&mut self, id: ExplorationId, store: &mut ExplorationStore<TestOptimization>) {
            println!("Execute exploration {id}");
            let exploration = store.get_unchecked(id);

            match &exploration.execution {
                ExecutionStrategy::Optimization(optimization) => {
                    self.operations.drain(0..optimization.size);
                }
                ExecutionStrategy::Operations => self.operations.clear(),
            };

            self.executed.push(id);
        }
    }

    fn operation_1() -> OperationDescription {
        OperationDescription::NumericFloat(NumericOperationDescription::Add(
            BinaryOperationDescription {
                lhs: TensorDescription {
                    id: TensorId::new(0),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                },
                rhs: TensorDescription {
                    id: TensorId::new(1),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                },
                out: TensorDescription {
                    id: TensorId::new(2),
                    shape: vec![32, 32],
                    status: TensorStatus::NotInit,
                },
            },
        ))
    }

    fn operation_2() -> OperationDescription {
        OperationDescription::NumericFloat(NumericOperationDescription::AddScalar(
            ScalarOperationDescription {
                lhs: TensorDescription {
                    id: TensorId::new(0),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                },
                rhs: 5.0,
                out: TensorDescription {
                    id: TensorId::new(2),
                    shape: vec![32, 32],
                    status: TensorStatus::NotInit,
                },
            },
        ))
    }
}
