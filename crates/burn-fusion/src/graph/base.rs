use std::collections::HashSet;

use burn_ir::{OperationIr, TensorId, TensorIr};

use crate::{OptimizationBuilder, OptimizationStatus};

pub struct MultiGraphs<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    pub graphs: Vec<Graph<O>>,
    length: usize,
}

pub struct Graph<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    operations: Vec<OperationIr>,
    ids: HashSet<TensorId>,
}

impl<O> Clone for Graph<O> {
    fn clone(&self) -> Self {
        Self {
            builders: self.builders.iter().map(|b| b.clone_dyn()).collect(),
            operations: self.operations.clone(),
            ids: self.ids.clone(),
        }
    }
}

pub enum Registration {
    Accepted,
    NotPartOfTheGraph,
}

pub enum MergeGraphResult {
    Succeed,
    NoNeed,
    Fail,
}

impl<O> MultiGraphs<O> {
    pub fn register(&mut self, operation: &OperationIr) -> OptimizationStatus {
        match self.merge_graphs(operation) {
            MergeGraphResult::Succeed | MergeGraphResult::NoNeed => {}
            MergeGraphResult::Fail => return OptimizationStatus::Closed,
        }

        let mut added_count = 0;

        for graph in self.graphs.iter_mut() {
            match graph.register(operation, false) {
                Registration::Accepted => {
                    added_count += 1;
                }
                Registration::NotPartOfTheGraph => {}
            }
        }
        if added_count == 0 {
            self.on_operation_skipped(operation);
        } else {
            assert_eq!(added_count, 1, "Can only add the operation to one graph.");
        }

        self.status()
    }

    pub fn merge_graphs(&mut self, operation: &OperationIr) -> MergeGraphResult {
        let nodes = operation.nodes();
        let mut graph_merges = Vec::new();

        for (i, graph) in self.graphs.iter().enumerate() {
            if graph.should_include_nodes(&nodes) {
                graph_merges.push(i);
            }
        }

        if graph_merges.is_empty() {
            return MergeGraphResult::NoNeed;
        }
        let combinations = generate_combinations(graph_merges.len(), graph_merges.len());

        for mut combination in combinations.into_iter() {
            let first = combination.remove(0);
            let first_index = graph_merges[first];
            let mut result = self.graphs[first_index].clone();

            for i in combination {
                let index = graph_merges[i];
                match result.try_merge(&self.graphs[index]) {
                    GraphMergingResult::Fail => {
                        continue;
                    }
                    GraphMergingResult::Succeed => {
                        let mut num_removed = 0;
                        // If ordered it's fine
                        for g in graph_merges {
                            self.graphs.remove(g - num_removed);
                            num_removed += 1;
                        }
                        self.graphs.push(result);
                        return MergeGraphResult::Succeed;
                    }
                }
            }
        }

        MergeGraphResult::Fail
    }

    pub fn on_operation_skipped(&mut self, operation: &OperationIr) {
        let mut graph = Graph {
            builders: self.builders.iter().map(|o| o.clone_dyn()).collect(),
            operations: Vec::new(),
            ids: HashSet::new(),
        };

        graph.register(operation, true);
        self.graphs.push(graph);
    }

    pub fn status(&self) -> OptimizationStatus {
        for g in self.graphs.iter() {
            match g.status() {
                OptimizationStatus::Closed => {}
                OptimizationStatus::Open => return OptimizationStatus::Open,
            }
        }

        OptimizationStatus::Closed
    }
}

pub enum GraphMergingResult {
    Fail,
    Succeed,
}

impl<O> Graph<O> {
    pub fn should_include_nodes(&self, nodes: &[&TensorIr]) -> bool {
        for node in nodes {
            if self.ids.contains(&node.id) {
                return true;
            }
        }

        false
    }

    pub fn test_merge(&self, other: &Graph<O>) -> GraphMergingResult {
        let mut base_graph = self.clone();
        base_graph.try_merge(other)
    }

    pub fn try_merge(&mut self, other: &Graph<O>) -> GraphMergingResult {
        for op in &other.operations {
            self.register(op, true);
        }

        match self.status() {
            OptimizationStatus::Closed => GraphMergingResult::Fail,
            OptimizationStatus::Open => GraphMergingResult::Succeed,
        }
    }

    pub fn status(&self) -> OptimizationStatus {
        for b in self.builders.iter() {
            match b.status() {
                OptimizationStatus::Closed => {}
                OptimizationStatus::Open => return OptimizationStatus::Open,
            }
        }

        OptimizationStatus::Closed
    }
    pub fn register(&mut self, operation: &OperationIr, force: bool) -> Registration {
        if self.ids.is_empty() {
            self.operations.push(operation.clone());
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

        self.operations.push(operation.clone());

        for builder in self.builders.iter_mut() {
            builder.register(operation);
        }

        Registration::Accepted
    }
}

fn generate_combinations(digits: usize, n: usize) -> Vec<Vec<usize>> {
    let total = n.pow(digits as u32); // Total combinations: n^digits
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let mut combination = Vec::with_capacity(digits);
        let mut num = i;
        for _ in 0..digits {
            combination.push(num % n); // Get the next digit (0 to n-1)
            num /= n;
        }
        // Reverse to match the expected order (most significant digit first)
        combination.reverse();
        result.push(combination);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_combinations_test() {
        let values = generate_combinations(3, 3);
        panic!("{values:?}");
    }
}
