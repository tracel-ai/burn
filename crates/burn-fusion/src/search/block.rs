use crate::{
    NumOperations, OptimizationBuilder, OptimizationStatus, stream::store::ExecutionStrategy,
};
use burn_ir::{OperationIr, TensorId, TensorIr};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A block represents a list of operations, not necessary in the same order as the execution
/// stream.
///
/// The start and end position of the relative execution stream is tracked in the block alonside
/// the ordering.
pub struct Block<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    operations: Vec<OperationIr>,
    ids: HashSet<TensorId>,
    ordering: Vec<usize>,
    /// The start position in the relative execution stream.
    pub start_pos: usize,
    /// The end position in the relative execution stream.
    pub end_pos: usize,
}

/// The result of [registering](Block::register) an [operation](OperationIr).
pub enum RegistrationResult {
    /// If the [operation](OperationIr) is correctly registered.
    Accepted,
    /// If the [operation](OperationIr) isn't part of the graph.
    ///
    /// In this case the operation isn't registered.
    NotPartOfTheGraph,
}

/// The optimization found for a [block](Block).
#[derive(Debug, new, Serialize, Deserialize)]
pub struct BlockOptimization<O> {
    /// The [execution strategy](ExecutionStrategy) to be used to execute the [block](Block).
    pub strategy: ExecutionStrategy<O>,
    /// The ordering of each operation in the relative execution stream.
    pub ordering: Vec<usize>,
}

impl<O: NumOperations> Block<O> {
    /// Create a new block that will be optimized with the provided [optimization builders](OptimizationBuilder).
    pub fn new(builders: &[Box<dyn OptimizationBuilder<O>>]) -> Self {
        Self {
            builders: builders.iter().map(|o| o.clone_dyn()).collect(),
            operations: Vec::new(),
            ids: HashSet::new(),
            ordering: Vec::new(),
            start_pos: usize::MAX,
            end_pos: usize::MIN,
        }
    }

    /// Sort the [blocks](Block) based on the start position.
    pub fn sort(blocks: &mut [Self]) {
        blocks.sort_by(|a, b| a.start_pos.cmp(&b.start_pos));
    }

    /// Optimize the block.
    pub fn optimize(mut self) -> BlockOptimization<O> {
        match find_best_optimization_index(&mut self.builders) {
            Some(index) => {
                let opt = self.builders[index].build();
                let opt_len = opt.len();
                let opt = ExecutionStrategy::Optimization(opt);

                if opt_len < self.operations.len() {
                    self.ordering.drain(opt_len..);
                }

                BlockOptimization::new(opt, self.ordering)
            }
            None => {
                let strategy = ExecutionStrategy::Operations(self.operations.len());
                BlockOptimization::new(strategy, self.ordering)
            }
        }
    }

    /// Returns if the block contains the provided [tensors](TensorIr).
    pub fn contains_tensors(&self, tensors: &[&TensorIr]) -> bool {
        for node in tensors {
            if self.ids.contains(&node.id) {
                return true;
            }
        }

        false
    }

    /// Merge the current block with the other one and returns if the operation is successful.
    ///
    /// # Warning
    ///
    /// This will modify the current block even if the other block isn't correctly merged.
    pub fn merge(&mut self, other: &Block<O>) -> bool {
        for (op, pos) in other.operations.iter().zip(&other.ordering) {
            self.register(op, *pos, true);
        }

        // The operation is successful if the current block can still be optimized.
        self.still_optimizing()
    }

    /// Register an [operation](OperationIr) in the current block.
    ///
    /// You need to provide the order of the operation as well as a force flag.
    /// When the force flag is true, the builder will always accept the operation, otherwise it
    /// might refuse it if the operation [isn't part of the graph](RegistrationResult::NotPartOfTheGraph).
    pub fn register(
        &mut self,
        operation: &OperationIr,
        order: usize,
        force: bool,
    ) -> RegistrationResult {
        if self.ids.is_empty() {
            self.register_op(operation, order);
            return RegistrationResult::Accepted;
        }
        let mut contains = false;
        for node in operation.nodes() {
            contains = self.ids.contains(&node.id);

            if contains {
                break;
            }
        }

        if !contains && !force {
            return RegistrationResult::NotPartOfTheGraph;
        }

        self.register_op(operation, order);
        RegistrationResult::Accepted
    }

    /// If the block can still be optimized further.
    pub fn still_optimizing(&self) -> bool {
        let mut num_stopped = 0;

        for optimization in self.builders.iter() {
            if let OptimizationStatus::Closed = optimization.status() {
                num_stopped += 1
            }
        }

        num_stopped < self.builders.len()
    }

    fn register_op(&mut self, operation: &OperationIr, pos: usize) {
        self.operations.push(operation.clone());
        self.ordering.push(pos);

        if pos < self.start_pos {
            self.start_pos = pos;
        }
        if pos + 1 > self.end_pos {
            self.end_pos = pos + 1;
        }

        for builder in self.builders.iter_mut() {
            builder.register(operation);
        }

        for node in operation.nodes() {
            self.ids.insert(node.id);
        }
    }
}

fn find_best_optimization_index<O>(
    optimizations: &mut [Box<dyn OptimizationBuilder<O>>],
) -> Option<usize> {
    let mut best_index = None;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = optimization.properties();

        if properties.ready && properties.score >= best_score {
            best_index = Some(i);
            best_score = properties.score;
        }
    }

    best_index
}

impl<O> PartialEq for Block<O> {
    fn eq(&self, other: &Self) -> bool {
        // Since the ordering can be seen as operation ids, we can use it to compare
        // blocks.
        let mut sorted_a = self.ordering.clone();
        let mut sorted_b = other.ordering.clone();
        sorted_a.sort();
        sorted_b.sort();

        sorted_a == sorted_b
    }
}

impl<O> core::fmt::Debug for Block<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Block {{ pos: [{:?}, {:?}; {:?}] }}",
            self.start_pos,
            self.end_pos,
            self.ordering.len(),
        ))
    }
}

impl<O> Clone for Block<O> {
    fn clone(&self) -> Self {
        Self {
            builders: self.builders.iter().map(|b| b.clone_dyn()).collect(),
            operations: self.operations.clone(),
            ids: self.ids.clone(),
            ordering: self.ordering.clone(),
            start_pos: self.start_pos,
            end_pos: self.end_pos,
        }
    }
}
