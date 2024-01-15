///! A testing module that ensures the correctness of the explorer, policy, and processor.
///!
///! The primary focus is on validating the seamless interaction between these three components to
///! execute and optimize a stream of operations accurately.
///!
///! To test these components effectively, we create mock types for the stream, optimization,
///! optimization builder, and stream segment. These mock types aid in comprehensively
///! understanding the process of optimizing streams.
use crate::{
    stream::{
        store::{
            ExecutionPlan, ExecutionPlanId, ExecutionStrategy, ExecutionPlanStore, ExecutionTrigger,
        },
        BinaryOperationDescription, NumericOperationDescription, OperationDescription,
        ScalarOperationDescription,
    },
    OptimizationBuilder, OptimizationProperties, OptimizationStatus, TensorDescription, TensorId,
    TensorStatus,
};

use super::*;

/// A fake stream of operations for testing purpose.
struct TestStream {
    processor: Processor<TestOptimization>,
    store: ExecutionPlanStore<TestOptimization>,
    executed: Vec<ExecutionPlanId>,
    operations: Vec<OperationDescription>,
}

/// A fake [optimization builder](OptimizationBuilder) for testing purpose.
struct TestOptimizationBuilder {
    builder_id: usize,
    expected_operations: Vec<OperationDescription>,
    expected_criterion: ExecutionTrigger,
    actual: Vec<OperationDescription>,
}

/// A fake optimization for testing purpose.
#[derive(new, Debug, PartialEq)]
struct TestOptimization {
    builder_id: usize,
    size: usize,
}

/// A fake [stream segment](StreamSegment) for testing purpose.
#[derive(new)]
struct TestSegment<'i> {
    operations: &'i mut Vec<OperationDescription>,
    executed: &'i mut Vec<ExecutionPlanId>,
}

/// This is a substantial test case that examines a lengthy scenario with a diverse set of conditions.
///
/// While it's usually preferable to split tests into multiple independent scenarios, in this case, it is
/// crucial to verify that the stream's state is correctly updated when various cases occur consecutively.
///
/// Although it might complicate identifying the source of a bug in the code, having this comprehensive
/// test case covers nearly all aspects of the implementation, while remaining easy to read and
/// maintainable.
#[test]
fn should_support_complex_stream() {
    // We have 2 different optimization builders in this test case.
    let builder_id_1 = 0;
    let builder_id_2 = 1;

    // We will have a total of 3 explorations to execute.
    let exploration_id_1 = 0;
    let exploration_id_2 = 1;
    let exploration_id_3 = 2;

    // The first builder only contains 2 operations, and the optimization is always available when
    // the pattern is meet.
    let builder_1 = TestOptimizationBuilder::new(
        builder_id_1,
        vec![operation_1(), operation_2()],
        ExecutionTrigger::Always,
    );
    // The second builder also contains 2 operations, but only becomes available when an operation
    // is meet.
    let builder_2 = TestOptimizationBuilder::new(
        builder_id_2,
        vec![operation_2(), operation_2()],
        ExecutionTrigger::OnOperation(operation_1()),
    );

    // We finaly build the stream with those optimization builders.
    let mut stream = TestStream::new(vec![Box::new(builder_1), Box::new(builder_2)]);

    // Nothing to execute for the first operation.
    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(0);

    // No optimization found for the first two operations.
    stream.add(operation_1());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(1);
    stream.assert_last_executed(exploration_id_1);
    stream.assert_exploration(
        exploration_id_1,
        ExecutionPlan {
            operations: vec![operation_1(), operation_1()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Operations,
        },
    );

    // Nothing to execute.
    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(1);

    // Now we should trigger the first optimization builder.
    stream.add(operation_2());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(2);
    stream.assert_last_executed(exploration_id_2);
    stream.assert_exploration(
        exploration_id_2,
        ExecutionPlan {
            operations: vec![operation_1(), operation_2()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
        },
    );

    // Nothing to execute.
    stream.add(operation_2());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(2);

    // Nothing to execute.
    stream.add(operation_2());
    stream.assert_number_of_operations(2);
    stream.assert_number_of_executions(2);

    // Now we should trigger the second optimization builder.
    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(3);
    stream.assert_last_executed(exploration_id_3);
    stream.assert_exploration(
        exploration_id_3,
        ExecutionPlan {
            operations: vec![operation_2(), operation_2()],
            triggers: vec![ExecutionTrigger::OnOperation(operation_1())],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_2, 2)),
        },
    );

    // Now we should trigger the first optimization builder (second exploration).
    stream.add(operation_2());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(4);
    stream.assert_last_executed(exploration_id_2);
    stream.assert_exploration(
        exploration_id_2,
        ExecutionPlan {
            operations: vec![operation_1(), operation_2()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
        },
    );

    // Nothing to execute.
    stream.add(operation_2());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(4);

    // Nothing to execute.
    stream.add(operation_2());
    stream.assert_number_of_operations(2);
    stream.assert_number_of_executions(4);

    // On sync we should execute all exploration when if their stop criteria isn't meet.
    // In this case the optimization from builder 2 (exploration 3).
    stream.sync();
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(5);
    stream.assert_last_executed(exploration_id_3);
    stream.assert_exploration(
        exploration_id_3,
        ExecutionPlan {
            operations: vec![operation_2(), operation_2()],
            triggers: vec![
                ExecutionTrigger::OnOperation(operation_1()),
                ExecutionTrigger::OnSync, // We also add OnSync in the stop criteria.
            ],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_2, 2)),
        },
    );
}

impl TestStream {
    /// Create a new stream with the given optimization builders.
    fn new(optimizations: Vec<Box<dyn OptimizationBuilder<TestOptimization>>>) -> Self {
        Self {
            processor: Processor::<TestOptimization>::new(optimizations),
            store: ExecutionPlanStore::<TestOptimization>::new(),
            executed: Vec::new(),
            operations: Vec::new(),
        }
    }

    /// Add an operation to the stream.
    fn add(&mut self, operation: OperationDescription) {
        self.operations.push(operation);
        self.processor.process(
            TestSegment::new(&mut self.operations, &mut self.executed),
            &mut self.store,
            ExecutionMode::Lazy,
        );
    }

    /// Sync the stream.
    fn sync(&mut self) {
        self.processor.process(
            TestSegment::new(&mut self.operations, &mut self.executed),
            &mut self.store,
            ExecutionMode::Sync,
        );
    }

    /// Assert that the exploration has been executed as provided.
    fn assert_exploration(&self, id: ExecutionPlanId, expected: ExecutionPlan<TestOptimization>) {
        let actual = self.store.get_unchecked(id);
        assert_eq!(actual.triggers, expected.triggers);
        assert_eq!(actual.operations, expected.operations);
    }

    /// Assert that the given exploration id has been the last executed.
    fn assert_last_executed(&self, id: ExecutionPlanId) {
        match self.executed.last() {
            Some(last_id) => assert_eq!(*last_id, id),
            None => panic!("No exploration has been executed"),
        }
    }

    /// Assert the number of executions since the start of the stream.
    fn assert_number_of_executions(&self, number: usize) {
        assert_eq!(self.executed.len(), number);
    }

    /// Assert the number of operations queued.
    fn assert_number_of_operations(&self, number: usize) {
        assert_eq!(self.operations.len(), number);
    }
}

impl TestOptimizationBuilder {
    /// Create a new optimization builder that follows the pattern with the stop criterion.
    fn new(
        builder_id: usize,
        operations: Vec<OperationDescription>,
        criteria: ExecutionTrigger,
    ) -> Self {
        Self {
            builder_id,
            expected_operations: operations,
            actual: Vec::new(),
            expected_criterion: criteria,
        }
    }
}

impl OptimizationBuilder<TestOptimization> for TestOptimizationBuilder {
    /// Register a new operation.
    fn register(&mut self, operation: &OperationDescription) {
        self.actual.push(operation.clone());
    }

    /// Build the optimization.
    fn build(&self) -> TestOptimization {
        TestOptimization::new(self.builder_id, self.len())
    }

    /// Reset the state.
    fn reset(&mut self) {
        self.actual.clear();
    }

    /// Return the optimization status.
    fn status(&self) -> OptimizationStatus {
        let actual_equal_expected = self.actual == self.expected_operations;

        if self.actual.len() < self.expected_operations.len() {
            let operations = &self.expected_operations[0..self.actual.len()];

            return match &self.actual == operations {
                // Still optimizing.
                true => OptimizationStatus::Open,
                // Never gonna be possible on that stream.
                false => OptimizationStatus::Closed,
            };
        }

        if self.actual.len() == self.expected_operations.len() {
            if actual_equal_expected {
                return match self.expected_criterion {
                    // Stop right away.
                    ExecutionTrigger::Always => OptimizationStatus::Closed,
                    // Wait for the next operation to show up.
                    ExecutionTrigger::OnOperation(_) => OptimizationStatus::Open,
                    // Doesn't matter on sync, even open should trigger a build if possible.
                    ExecutionTrigger::OnSync => OptimizationStatus::Open,
                };
            }
        }

        OptimizationStatus::Closed
    }

    /// Return the properties of this optimization.
    fn properties(&self) -> OptimizationProperties {
        if self.actual.len() < self.expected_operations.len() {
            // Optimization not possible.
            return OptimizationProperties {
                score: 0,
                ready: false,
            };
        }

        let stream_is_ok =
            &self.actual[0..self.expected_operations.len()] == &self.expected_operations;

        if !stream_is_ok {
            // Optimization not possible.
            return OptimizationProperties {
                score: 0,
                ready: false,
            };
        }

        // Optimization possible.
        OptimizationProperties {
            score: 1,
            ready: true,
        }
    }

    // The number of operations that should be handle by the optimization.
    fn len(&self) -> usize {
        self.expected_operations.len()
    }
}

impl<'i> StreamSegment<TestOptimization> for TestSegment<'i> {
    // The operations in the process.
    fn operations<'a>(&'a self) -> &[OperationDescription] {
        &self.operations
    }

    // Execute the process.
    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<TestOptimization>) {
        let exploration = store.get_unchecked(id);

        match &exploration.strategy {
            ExecutionStrategy::Optimization(optimization) => {
                self.operations.drain(0..optimization.size);
            }
            ExecutionStrategy::Operations => self.operations.clear(),
        };

        self.executed.push(id);
    }
}

/// Just a simple operation.
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

/// Just a simple operation.
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
