use std::collections::HashSet;

use burn_ir::{OperationIr, TensorId, TensorIr};

use crate::{
    NumOperations, OptimizationBuilder, OptimizationStatus, stream::store::ExecutionStrategy,
};

pub struct MultiGraphs<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    pub graphs: Vec<Graph<O>>,
    length: usize,
    merged_failed: bool,
}

pub struct Graph<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    operations: Vec<OperationIr>,
    ids: HashSet<TensorId>,
    operations_positions: Vec<usize>,
}

impl<O> core::fmt::Debug for Graph<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Graph {{ pos: {:?}, }}",
            self.operations_positions,
        ))
    }
}
impl<O> Clone for Graph<O> {
    fn clone(&self) -> Self {
        Self {
            builders: self.builders.iter().map(|b| b.clone_dyn()).collect(),
            operations: self.operations.clone(),
            ids: self.ids.clone(),
            operations_positions: self.operations_positions.clone(),
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

impl<O: NumOperations> MultiGraphs<O> {
    pub fn new(builders: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            builders,
            graphs: Vec::new(),
            length: 0,
            merged_failed: false,
        }
    }
    pub fn register(&mut self, operation: &OperationIr) {
        match self.merge_graphs(operation) {
            MergeGraphResult::Succeed | MergeGraphResult::NoNeed => {}
            MergeGraphResult::Fail => {
                self.merged_failed = true;
                return;
            }
        }

        let mut added_count = 0;

        for graph in self.graphs.iter_mut() {
            match graph.register(operation, false, self.length) {
                Registration::Accepted => {
                    added_count += 1;
                }
                Registration::NotPartOfTheGraph => {}
            }
        }
        if added_count == 0 {
            self.on_new_graph(operation);
        } else {
            assert_eq!(added_count, 1, "Can only add the operation to one graph.");
        }
        self.length += 1;
    }

    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.graphs.clear();
        self.merged_failed = false;
    }

    pub fn strategy(&mut self) -> (ExecutionStrategy<O>, usize) {
        self.graphs
            .sort_by(|a, b| a.operations_positions[0].cmp(&b.operations_positions[0]));

        let mut strategies = Vec::with_capacity(self.graphs.len());

        let mut all_positions = Vec::new();

        for graph in self.graphs.iter() {
            let strategy = graph.clone().into_strategy();
            match strategy {
                IntoStrategy::Full(execution_strategy, positions) => {
                    all_positions.append(&mut positions.clone());
                    strategies.push(Box::new(execution_strategy));
                }
                IntoStrategy::Partial(execution_strategy, positions) => {
                    all_positions.append(&mut positions.clone());
                    strategies.push(Box::new(execution_strategy));
                    // Check if positions are OK before breaking.
                    break;
                }
            }
        }

        if strategies.is_empty() {
            panic!("Whoo");
        }

        let end = all_positions.len();
        all_positions.sort();
        assert_eq!(all_positions, (0..end).collect::<Vec<_>>());

        (ExecutionStrategy::Composed(strategies), all_positions.len())
    }

    pub fn still_optimizing(&self) -> bool {
        if self.graphs.is_empty() {
            return true;
        }
        if self.merged_failed {
            return false;
        }

        let mut num_stopped = 0;

        for graph in self.graphs.iter() {
            if !graph.still_optimizing() {
                num_stopped += 1
            }
        }

        num_stopped < self.graphs.len()
    }

    fn merge_graphs(&mut self, operation: &OperationIr) -> MergeGraphResult {
        let nodes = operation.nodes();
        let mut graph_merges = Vec::new();

        for (i, graph) in self.graphs.iter().enumerate() {
            if graph.should_include_nodes(&nodes) {
                graph_merges.push(i);
            }
        }

        if graph_merges.len() <= 1 {
            return MergeGraphResult::NoNeed;
        }

        let combinations = generate_combinations(graph_merges.len(), graph_merges.len());

        for mut combination in combinations.into_iter() {
            let first = combination.remove(0);
            let first_index = graph_merges[first];
            let mut result = self.graphs[first_index].clone();
            let mut working = true;

            for i in combination {
                let index = graph_merges[i];
                match result.merge(&self.graphs[index]) {
                    GraphMergingResult::Fail => {
                        working = false;
                        break;
                    }
                    GraphMergingResult::Succeed => {}
                }
            }

            if working {
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

        MergeGraphResult::Fail
    }

    fn on_new_graph(&mut self, operation: &OperationIr) {
        let mut graph = Graph {
            builders: self.builders.iter().map(|o| o.clone_dyn()).collect(),
            operations: Vec::new(),
            ids: HashSet::new(),
            operations_positions: Vec::new(),
        };

        graph.register(operation, true, self.length);
        self.graphs.push(graph);
    }
}

pub enum GraphMergingResult {
    Fail,
    Succeed,
}

pub enum IntoStrategy<O> {
    Full(ExecutionStrategy<O>, Vec<usize>),
    Partial(ExecutionStrategy<O>, Vec<usize>),
}

impl<O: NumOperations> Graph<O> {
    pub(crate) fn into_strategy(mut self) -> IntoStrategy<O> {
        match find_best_optimization_index(&mut self.builders) {
            Some(index) => {
                let opt = self.builders[index].build();
                let opt_len = opt.len();
                let opt = ExecutionStrategy::Optimization(opt);

                if opt_len < self.operations.len() {
                    self.operations_positions.drain(opt_len..);
                    IntoStrategy::Partial(opt, self.operations_positions)
                } else {
                    IntoStrategy::Full(opt, self.operations_positions)
                }
            }
            None => {
                let strategy = ExecutionStrategy::Operations(self.operations.len());
                IntoStrategy::Full(strategy, self.operations_positions)
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

    pub fn merge(&mut self, other: &Graph<O>) -> GraphMergingResult {
        for (op, pos) in other.operations.iter().zip(&other.operations_positions) {
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
        self.operations_positions.push(pos);

        for builder in self.builders.iter_mut() {
            builder.register(operation);
        }

        for node in operation.nodes() {
            self.ids.insert(node.id);
        }
    }

    fn still_optimizing(&self) -> bool {
        let mut num_stopped = 0;

        for optimization in self.builders.iter() {
            if let OptimizationStatus::Closed = optimization.status() {
                num_stopped += 1
            }
        }

        num_stopped < self.builders.len()
    }
}

fn generate_combinations(digits: usize, n: usize) -> Vec<Vec<usize>> {
    // TODO: Optimize that.
    let total = n.pow(digits as u32);
    let mut result = Vec::with_capacity(total);

    for i in 0..total {
        let mut combination = Vec::with_capacity(digits);
        let mut num = i;
        for _ in 0..digits {
            combination.push(num % n);
            num /= n;
        }
        combination.reverse();

        let mut seen = vec![false; n];
        let mut unique = true;
        for &x in &combination {
            if seen[x] {
                unique = false;
                break;
            }
            seen[x] = true;
        }
        if unique {
            result.push(combination);
        }
    }

    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_combinations_test() {
        let values = generate_combinations(3, 3);
        panic!("{values:?}");
    }
}
