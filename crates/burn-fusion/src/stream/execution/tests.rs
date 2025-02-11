//! A testing module that ensures the correctness of the explorer, policy, and processor.
//!
//! The primary focus is on validating the seamless interaction between these three components to
//! execute and optimize a stream of operations accurately.
//!
//! To test these components effectively, we create mock types for the stream, optimization,
//! optimization builder, and stream segment. These mock types aid in comprehensively
//! understanding the process of optimizing streams.
use burn_ir::{
    BinaryOpIr, FloatOperationIr, NumericOperationIr, OperationIr, ScalarOpIr, TensorId, TensorIr,
    TensorStatus, UnaryOpIr,
};
use burn_tensor::DType;

use crate::{
    stream::store::{
        ExecutionPlan, ExecutionPlanId, ExecutionPlanStore, ExecutionStrategy, ExecutionTrigger,
    },
    OptimizationBuilder, OptimizationProperties, OptimizationStatus,
};

use super::*;

/// A fake stream of operations for testing purpose.
struct TestStream {
    processor: Processor<TestOptimization>,
    store: ExecutionPlanStore<TestOptimization>,
    executed: Vec<ExecutionPlanId>,
    operations: Vec<OperationIr>,
}

/// A fake [optimization builder](OptimizationBuilder) for testing purpose.
///
/// The optimizer tries to fuse only the `expected_operations` if they appear
/// in the operations queue
struct TestOptimizationBuilder {
    builder_id: usize,
    expected_operations: Vec<OperationIr>,
    actual: Vec<OperationIr>,
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
    operations: &'i mut Vec<OperationIr>,
    executed: &'i mut Vec<ExecutionPlanId>,
}

/// This is a substantial test case that examines a lengthy scenario with a diverse set of conditions.
///
/// While it's usually preferable to split tests into multiple independent scenarios, in this case, it is
/// crucial to verify that the stream's state is correctly updated when various cases occur consecutively.
#[test]
fn should_support_complex_stream() {
    // We have 2 different optimization builders in this test case.
    let builder_id_1 = 0;
    let builder_id_2 = 1;

    // We will have a total of 3 execution plans to execute.
    let plan_id_1 = 0;
    let plan_id_2 = 1;
    let plan_id_3 = 2;

    let builder_1 = TestOptimizationBuilder::new(builder_id_1, vec![operation_1(), operation_2()]);
    let builder_2 = TestOptimizationBuilder::new(builder_id_2, vec![operation_2(), operation_2()]);
    let mut stream = TestStream::new(vec![Box::new(builder_1), Box::new(builder_2)]);

    // builder_1 is still waiting to see next op is operation_2
    // builder_2 is closed because it's not the right operation
    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(0);

    // No optimization found for the first two operations.
    stream.add(operation_1());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(1);
    stream.assert_last_executed(plan_id_1);
    stream.assert_plan(
        plan_id_1,
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
    stream.assert_last_executed(plan_id_2);
    stream.assert_plan(
        plan_id_2,
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

    // Now we should trigger the second optimization builder.
    stream.add(operation_2());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(3);
    stream.assert_last_executed(plan_id_3);
    stream.assert_plan(
        plan_id_3,
        ExecutionPlan {
            operations: vec![operation_2(), operation_2()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_2, 2)),
        },
    );

    // Nothing to execute.
    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(3);

    // Now we should trigger the first optimization builder (second plan).
    stream.add(operation_2());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(4);
    stream.assert_last_executed(plan_id_2);
    stream.assert_plan(
        plan_id_2,
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

    // Now we should trigger the first optimization builder (third plan).
    stream.add(operation_2());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(5);
    stream.assert_last_executed(plan_id_3);
}

/// In this scenario we will never use an optimization, but we check that we reuse the execution plan stored.
#[test]
fn should_reuse_basic_operations() {
    let builder_id_1 = 0;
    let plan_id_1 = 0;
    let plan_id_2 = 1;

    let builder_1 = TestOptimizationBuilder::new(builder_id_1, vec![operation_1(), operation_2()]);
    let mut stream = TestStream::new(vec![Box::new(builder_1)]);

    stream.add(operation_3());
    stream.assert_last_executed(plan_id_1);
    stream.assert_number_of_operations(0);
    stream.assert_plan(
        plan_id_1,
        ExecutionPlan {
            operations: vec![operation_3()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Operations,
        },
    );

    stream.add(operation_3());
    stream.assert_last_executed(plan_id_1);
    stream.assert_number_of_operations(0);
    stream.assert_plan(
        plan_id_1,
        ExecutionPlan {
            operations: vec![operation_3()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Operations,
        },
    );

    // Lazy try to build optimization 1.
    stream.add(operation_1());
    // But not possible.
    stream.add(operation_3());

    // Creates a new plan with both operations.
    stream.assert_plan(
        plan_id_2,
        ExecutionPlan {
            operations: vec![operation_1(), operation_3()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Operations,
        },
    );
    stream.assert_number_of_operations(0);
    stream.assert_last_executed(plan_id_2);
}

// In this scenario we validate that we support multiple optimization builders with overlapping
// operations.
//
// This is a very long scenario that validates a lot of things.
#[test]
fn should_support_overlapping_optimizations() {
    // We have 2 different optimization builders in this test case.
    let builder_id_1 = 0;
    let builder_id_2 = 0;

    // We will have a total of 5 execution plans to execute.
    let plan_id_1 = 0;
    let plan_id_2 = 1;
    let plan_id_3 = 2;
    let plan_id_4 = 3;
    let plan_id_5 = 4;

    let builder_1 = TestOptimizationBuilder::new(builder_id_1, vec![operation_1(), operation_2()]);
    let builder_2 = TestOptimizationBuilder::new(
        builder_id_2,
        vec![operation_1(), operation_2(), operation_1(), operation_1()],
    );
    let mut stream = TestStream::new(vec![Box::new(builder_1), Box::new(builder_2)]);

    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(0);

    stream.add(operation_2());
    stream.assert_number_of_operations(2);
    stream.assert_number_of_executions(0);

    stream.add(operation_1());
    stream.assert_number_of_operations(3);
    stream.assert_number_of_executions(0);

    stream.add(operation_2());
    stream.assert_number_of_operations(2);
    stream.assert_number_of_executions(1);
    stream.assert_last_executed(plan_id_1);
    stream.assert_plan(
        plan_id_1,
        ExecutionPlan {
            operations: vec![operation_1(), operation_2()],
            triggers: vec![ExecutionTrigger::OnOperations(vec![
                operation_1(),
                operation_2(),
            ])],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
        },
    );

    stream.add(operation_2());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(3);
    stream.assert_plan(
        plan_id_1,
        ExecutionPlan {
            operations: vec![operation_1(), operation_2()],
            triggers: vec![
                ExecutionTrigger::OnOperations(vec![operation_1(), operation_2()]),
                ExecutionTrigger::OnOperations(vec![operation_2()]),
            ],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
        },
    );
    stream.assert_plan(
        plan_id_2,
        ExecutionPlan {
            operations: vec![operation_2()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Operations,
        },
    );

    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(3);

    stream.add(operation_2());
    stream.assert_number_of_operations(2);
    stream.assert_number_of_executions(3);

    stream.add(operation_1());
    stream.assert_number_of_operations(3);
    stream.assert_number_of_executions(3);

    stream.add(operation_1());
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(4);

    stream.assert_plan(
        plan_id_3,
        ExecutionPlan {
            operations: vec![operation_1(), operation_2(), operation_1(), operation_1()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 4)),
        },
    );

    stream.add(operation_1());
    stream.assert_number_of_operations(1);
    stream.assert_number_of_executions(4);

    stream.add(operation_2());
    stream.assert_number_of_operations(2);
    stream.assert_number_of_executions(4);

    stream.add(operation_1());
    stream.assert_number_of_operations(3);
    stream.assert_number_of_executions(4);

    stream.sync();
    stream.assert_number_of_operations(0);
    stream.assert_number_of_executions(6);
    stream.assert_plan(
        plan_id_1,
        ExecutionPlan {
            operations: vec![operation_1(), operation_2()],
            triggers: vec![
                ExecutionTrigger::OnOperations(vec![operation_1(), operation_2()]),
                ExecutionTrigger::OnOperations(vec![operation_2()]),
                ExecutionTrigger::OnSync,
            ],
            strategy: ExecutionStrategy::Optimization(TestOptimization::new(builder_id_1, 2)),
        },
    );
    stream.assert_plan(
        plan_id_4,
        ExecutionPlan {
            operations: vec![operation_1()],
            triggers: vec![ExecutionTrigger::OnSync],
            strategy: ExecutionStrategy::Operations,
        },
    );

    stream.add(operation_3());
    stream.assert_last_executed(plan_id_5);
    stream.assert_plan(
        plan_id_5,
        ExecutionPlan {
            operations: vec![operation_3()],
            triggers: vec![ExecutionTrigger::Always],
            strategy: ExecutionStrategy::Operations,
        },
    );

    stream.add(operation_3());
    stream.assert_last_executed(plan_id_5);
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
    fn add(&mut self, operation: OperationIr) {
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

    /// Assert that the plan has been executed as provided.
    fn assert_plan(&self, id: ExecutionPlanId, expected: ExecutionPlan<TestOptimization>) {
        let actual = self.store.get_unchecked(id);
        assert_eq!(actual.operations, expected.operations, "Same operations");
        assert_eq!(actual.triggers, expected.triggers, "Same triggers");
    }

    /// Assert that the given plan id has been the last executed.
    fn assert_last_executed(&self, id: ExecutionPlanId) {
        match self.executed.last() {
            Some(last_id) => assert_eq!(*last_id, id),
            None => panic!("No plan has been executed"),
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
    /// Create a new optimization builder that follows a pattern with a trigger.
    fn new(builder_id: usize, operations: Vec<OperationIr>) -> Self {
        Self {
            builder_id,
            expected_operations: operations,
            actual: Vec::new(),
        }
    }
}

impl OptimizationBuilder<TestOptimization> for TestOptimizationBuilder {
    /// Register a new operation.
    fn register(&mut self, operation: &OperationIr) {
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
        if self.actual.len() < self.expected_operations.len() {
            let operations = &self.expected_operations[0..self.actual.len()];

            return match self.actual == operations {
                // Still optimizing.
                true => OptimizationStatus::Open,
                // Never gonna be possible on that stream.
                false => OptimizationStatus::Closed,
            };
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
            self.actual[0..self.expected_operations.len()] == self.expected_operations;

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

impl StreamSegment<TestOptimization> for TestSegment<'_> {
    // The operations in the process.
    fn operations(&self) -> &[OperationIr] {
        self.operations
    }

    // Execute the process.
    fn execute(&mut self, id: ExecutionPlanId, store: &mut ExecutionPlanStore<TestOptimization>) {
        let execution_plan = store.get_unchecked(id);

        match &execution_plan.strategy {
            ExecutionStrategy::Optimization(optimization) => {
                self.operations.drain(0..optimization.size);
            }
            ExecutionStrategy::Operations => self.operations.clear(),
        };

        self.executed.push(id);
    }
}

/// Just a simple operation.
fn operation_1() -> OperationIr {
    OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Add(BinaryOpIr {
            lhs: TensorIr {
                id: TensorId::new(0),
                shape: vec![32, 32],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32,
            },
            rhs: TensorIr {
                id: TensorId::new(1),
                shape: vec![32, 32],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32,
            },
            out: TensorIr {
                id: TensorId::new(2),
                shape: vec![32, 32],
                status: TensorStatus::NotInit,
                dtype: DType::F32,
            },
        }),
    )
}

/// Just a simple operation.
fn operation_2() -> OperationIr {
    OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::AddScalar(ScalarOpIr {
            lhs: TensorIr {
                id: TensorId::new(0),
                shape: vec![32, 32],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32,
            },
            rhs: 5.0,
            out: TensorIr {
                id: TensorId::new(2),
                shape: vec![32, 32],
                status: TensorStatus::NotInit,
                dtype: DType::F32,
            },
        }),
    )
}

/// Just a simple operation.
fn operation_3() -> OperationIr {
    OperationIr::Float(
        DType::F32,
        FloatOperationIr::Log(UnaryOpIr {
            input: TensorIr {
                id: TensorId::new(0),
                shape: vec![32, 32],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32,
            },
            out: TensorIr {
                id: TensorId::new(0),
                shape: vec![32, 32],
                status: TensorStatus::NotInit,
                dtype: DType::F32,
            },
        }),
    )
}
