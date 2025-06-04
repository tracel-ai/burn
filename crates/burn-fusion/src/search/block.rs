use std::collections::HashSet;

use burn_ir::{OperationIr, TensorId, TensorIr};

use crate::{
    NumOperations, OptimizationBuilder, OptimizationStatus, stream::store::ExecutionStrategy,
};

/// A block represents a list of operations, not necessary in the same order as the execution
/// stream. The start and end position of the relative execution stream is tracked in the block.
pub struct Block<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    operations: Vec<OperationIr>,
    ids: HashSet<TensorId>,
    positions: Vec<usize>,
    /// The start position in the relative execution stream.
    pub start_pos: usize,
    /// The end position in the relative execution stream.
    pub end_pos: usize,
}

impl<O> PartialEq for Block<O> {
    fn eq(&self, other: &Self) -> bool {
        let mut sorted_a = self.positions.clone();
        let mut sorted_b = other.positions.clone();
        sorted_a.sort();
        sorted_b.sort();

        sorted_a == sorted_b
    }
}

impl<O> core::fmt::Debug for Block<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Block {{ pos: [{:?}, {:?}; {:?}], ids: {:?}, ops: {:?}}}",
            self.start_pos,
            self.end_pos,
            self.positions.len(),
            self.ids,
            self.operations
        ))
    }
}

impl<O> Clone for Block<O> {
    fn clone(&self) -> Self {
        Self {
            builders: self.builders.iter().map(|b| b.clone_dyn()).collect(),
            operations: self.operations.clone(),
            ids: self.ids.clone(),
            positions: self.positions.clone(),
            start_pos: self.start_pos.clone(),
            end_pos: self.end_pos.clone(),
        }
    }
}

pub enum Registration {
    Accepted,
    NotPartOfTheGraph,
}

pub enum GraphMergingResult {
    Fail,
    Succeed,
}

impl<O: NumOperations> Block<O> {
    pub fn sort(blocks: &mut Vec<Self>) {
        blocks.sort_by(|a, b| a.start_pos.cmp(&b.start_pos));
    }
    pub fn new(builders: &[Box<dyn OptimizationBuilder<O>>]) -> Self {
        Self {
            builders: builders.iter().map(|o| o.clone_dyn()).collect(),
            operations: Vec::new(),
            ids: HashSet::new(),
            positions: Vec::new(),
            start_pos: usize::MAX,
            end_pos: usize::MIN,
        }
    }

    pub fn compile(mut self) -> (ExecutionStrategy<O>, Vec<usize>) {
        match find_best_optimization_index(&mut self.builders) {
            Some(index) => {
                let opt = self.builders[index].build();
                let opt_len = opt.len();
                let opt = ExecutionStrategy::Optimization(opt);

                if opt_len < self.operations.len() {
                    self.positions.drain(opt_len..);
                    (opt, self.positions)
                } else {
                    (opt, self.positions)
                }
            }
            None => {
                let strategy = ExecutionStrategy::Operations(self.operations.len());
                (strategy, self.positions)
            }
        }
    }

    pub fn should_include_nodes(&self, nodes: &[&TensorIr]) -> bool {
        for node in nodes {
            if self.ids.contains(&node.id) {
                return true;
            }
        }

        false
    }

    pub fn merge(&mut self, other: &Block<O>) -> GraphMergingResult {
        for (op, pos) in other.operations.iter().zip(&other.positions) {
            self.register(op, true, *pos);
        }

        match self.still_optimizing() {
            false => GraphMergingResult::Fail,
            true => GraphMergingResult::Succeed,
        }
    }

    pub fn register(&mut self, operation: &OperationIr, force: bool, pos: usize) -> Registration {
        if self.ids.is_empty() {
            self.register_op(operation, pos);
            return Registration::Accepted;
        }
        let mut contains = false;
        for node in operation.nodes() {
            contains = self.ids.contains(&node.id);

            if contains {
                break;
            }
        }

        if !contains && !force {
            return Registration::NotPartOfTheGraph;
        }

        self.register_op(operation, pos);
        Registration::Accepted
    }

    fn register_op(&mut self, operation: &OperationIr, pos: usize) {
        self.operations.push(operation.clone());
        self.positions.push(pos);

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

    pub fn still_optimizing(&self) -> bool {
        let mut num_stopped = 0;

        for optimization in self.builders.iter() {
            if let OptimizationStatus::Closed = optimization.status() {
                num_stopped += 1
            }
        }

        num_stopped < self.builders.len()
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
