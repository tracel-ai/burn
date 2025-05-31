use crate::{NumOperations, stream::store::ExecutionStrategy};

use super::{Graph, merging::merge_graphs};

pub struct Compilation<O> {
    graphs: Vec<Graph<O>>,
    resolved: Vec<bool>,
    last_checked: usize,
}

impl<O: NumOperations> Compilation<O> {
    pub fn new(mut graphs: Vec<Graph<O>>) -> Self {
        graphs.sort_by(|a, b| a.start_pos.cmp(&b.start_pos));
        let num_ops: usize = graphs.iter().map(|g| g.end_pos).max().unwrap();

        Self {
            graphs,
            resolved: vec![false; num_ops],
            last_checked: 0,
        }
    }

    pub fn compile(mut self) -> (ExecutionStrategy<O>, usize) {
        self = self.merging_pass();

        let mut strategies = Vec::with_capacity(self.graphs.len());

        let mut graphs = Vec::new();
        core::mem::swap(&mut graphs, &mut self.graphs);

        let mut num_optimized = 0;

        for graph in graphs {
            let last_index = graph.end_pos;
            let (strategy, positions) = graph.compile();
            let opt_size = positions.len();

            for pos in positions {
                self.update_check(pos);
            }

            if self.last_checked != num_optimized + opt_size {
                if num_optimized > 0 {
                    // Don't include that graph and need furthur exploring.
                    break;
                } else {
                    num_optimized += opt_size;
                    match strategy {
                        ExecutionStrategy::Optimization(opt) => {
                            let fallbacks = self.add_missing_ops(last_index);
                            let strategy =
                                ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks);
                            strategies.push(Box::new(strategy));
                            break;
                        }
                        _ => unreachable!(),
                    };
                }
            }

            num_optimized += opt_size;
            strategies.push(Box::new(strategy));
        }

        (ExecutionStrategy::Composed(strategies), num_optimized)
    }

    fn update_check(&mut self, pos: usize) {
        self.resolved[pos] = true;

        if pos == self.last_checked {
            self.last_checked += 1;

            for i in self.last_checked..self.resolved.len() {
                if self.resolved[i] {
                    self.last_checked += 1;
                } else {
                    return;
                }
            }
        }
    }

    fn add_missing_ops(&self, last: usize) -> Vec<usize> {
        let mut fallbacks = Vec::new();

        for i in self.last_checked..last {
            if !self.resolved[i] {
                fallbacks.push(i);
            }
        }

        fallbacks
    }

    fn merging_pass(mut self) -> Self {
        if self.graphs.len() == 1 {
            return self;
        }
        // TODO: Can be slow.
        if let Some(graph) = merge_graphs(&self.graphs.iter().collect::<Vec<_>>()) {
            self.graphs = vec![graph];
        }

        self
    }
}
