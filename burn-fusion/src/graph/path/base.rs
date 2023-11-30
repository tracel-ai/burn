use super::starter::Starters;
use crate::graph::TensorOpsDescription;

/// The cache works by keeping track of all possible optimizations for the current graph path.
///
/// # Details
///
/// This is pretty different from a normal key-value cache.
/// There is no key to access the cached values, since computing a key for a graph is very expensive.
/// Instead, we keep track of each new edge added to the graph and invalidate potential optimizations
/// when we see a different edge is added while keeping track of the current graph path.
///
/// Therefore, the overhead is really minimal, since the time-complexity of checking the cache
/// scales with the number of concurrent potential optimizations for the current path, which isn't
/// supposed to be big at any time.
pub(crate) struct OptimizationCache<O> {
    candidates: Vec<OptimizationId>,
    availables: Vec<(OptimizationId, usize)>,
    optimizations: Vec<OptimizationItem<O>>,
    starters: Starters,
    found: Option<OptimizationId>,
}

impl<O> OptimizationCache<O> {
    pub(crate) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            availables: Vec::new(),
            optimizations: Vec::new(),
            starters: Starters::default(),
            found: None,
        }
    }

    /// Follow the current path on the provided graph with the end_condition.
    ///
    /// # Notes
    ///
    /// It is assumed that this function will be called for each new edge added to the graph (for
    /// each new operation). Only one graph can be cache at a time.
    pub(crate) fn follow<'a>(
        &'a mut self,
        graph: &[TensorOpsDescription],
        end_condition: EndCondition,
    ) -> CacheResult<'a, O> {
        if graph.is_empty() {
            // Starter
            let ops = match end_condition {
                EndCondition::NextOps(ops) => ops,
                EndCondition::Forced => return CacheResult::Miss, // Force en empty graph...
            };
            let candidates = self.starters.get(ops);
            if candidates.is_empty() {
                return CacheResult::Miss;
            }
            self.candidates = candidates;
            return CacheResult::OnPath;
        }

        if let Some(candidate) = self.found {
            return CacheResult::Found(&self.optimizations.get(candidate).unwrap().ops);
        };

        // Invalidate candidates.
        let mut invalidated_candidate = Vec::new();
        for candidate in self.candidates.iter() {
            let graph_candidate = match self.optimizations.get(*candidate) {
                Some(val) => val,
                None => panic!("Should have candidate"),
            };
            let next_ops = graph.last().expect("Validated earlier");
            let next_ops_index = graph.len() - 1;
            let next_ops_candidate = match graph_candidate.graph.get(next_ops_index) {
                Some(val) => val,
                None => {
                    invalidated_candidate.push(*candidate);
                    continue;
                }
            };

            if next_ops_candidate != next_ops {
                invalidated_candidate.push(*candidate);
                continue;
            }

            if graph_candidate.graph.len() == graph.len() {
                let ops = match end_condition {
                    EndCondition::NextOps(ops) => ops,
                    EndCondition::Forced => {
                        self.found = Some(*candidate);
                        return CacheResult::Found(&graph_candidate.ops);
                    }
                };

                if graph_candidate.end_condition.contains(ops) {
                    self.found = Some(*candidate);
                    return CacheResult::Found(&graph_candidate.ops);
                } else {
                    self.availables.push((*candidate, graph.len()));
                    invalidated_candidate.push(*candidate);
                }
            }
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut self.candidates);

        self.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| !invalidated_candidate.contains(candidate))
            .collect();

        if self.candidates.is_empty() {
            CacheResult::Miss
        } else {
            CacheResult::OnPath
        }
    }

    /// Signal the completion of a graph path that reached a new optimization.
    ///
    /// # Notes
    ///
    /// The optimization factory will only be call if the optimization is on a new graph.
    /// When the optimization already exist, but with a different end condition, a new end
    /// condition will be registered, but the old optimization will be used in following call. This
    /// is indented since we want to factory to be call only once per graph, but reused as much as
    /// possible.
    pub fn complete<'a, Factory: OptimizationFactory<O>>(
        &'a mut self,
        factory: &Factory,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) -> &'a O {
        let existing_optim = self
            .availables
            .iter()
            .find(|(_candidate, len)| *len == graph.len());

        if let Some((id, _)) = existing_optim {
            let optimization = self.optimizations.get_mut(*id).unwrap();

            if let Some(ops) = next_ops {
                optimization.end_condition.push(ops)
            };

            return &optimization.ops;
        };

        self.starters
            .insert(graph.first().unwrap(), self.optimizations.len());
        let ops = factory.create();
        let optimization = OptimizationItem {
            graph,
            end_condition: match next_ops {
                Some(val) => vec![val],
                None => Vec::new(),
            },
            ops,
        };

        self.optimizations.push(optimization);
        &self.optimizations.last().unwrap().ops
    }

    // Signal that a new path will begin.
    pub(crate) fn reset(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.found = None;
    }
}

/// Action to be made depending on the graph.
#[derive(PartialEq, Eq)]
pub enum CacheResult<'a, T> {
    /// Continue exploring optimizations but using the [fusion ops builder](crate::FusionOpsBuilder).
    Miss,
    /// The current graph indicates that some optimization maybe possible in the future, so the
    /// best action is to wait for the optimization to become available.
    ///
    /// Sometime, if can be a false positive and a new opitmization should be built from scratch,
    /// therefore it is important to keep the previous operations to rebuilt the state if it
    /// happens.
    OnPath,
    /// An optimization has been found, and the best action is to execute it!
    Found(&'a T),
}

/// When checking if an optimization is possible, a end condition assure that this optimization is
/// always optimal.
///
/// # Example
///
/// For the same beginning of a graph, an opitmization might be optimal only when followed by
/// another operation.
///
/// Graph: [Add - Accepted] - [Div - Accepted]
///
/// 1. Optimal
///     [Add - Accepted] - [Div - Accepted] - [Matmul - Refused]
///     In this case we should execute a fused kernel for [Add] and [Div]
///
/// 2. Non-Optimal
///     [Add - Accepted] - [Div - Accepted] - [Exp - Accepted] - [Matmul - Refused]
///     In this case we should not execute the fused kernel [Add] and [div], but wait to execute
///     the fused kernel [Add] - [Div] - [Exp].
#[derive(Clone)]
pub enum EndCondition<'a> {
    /// The next operation that signal the end of the operation.
    NextOps(&'a TensorOpsDescription),
    /// When forced, we should execute the optimization if found no matter what comes next.
    Forced,
}

impl<'a, T> core::fmt::Debug for CacheResult<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheResult::Miss => f.write_str("CacheResult::Miss"),
            CacheResult::OnPath => f.write_str("CacheResult::OnPath"),
            CacheResult::Found(_) => f.write_str("CacheResult::Found"),
        }
    }
}

/// Create an optimization.
pub trait OptimizationFactory<T> {
    /// Call only when a new optimization is found.
    fn create(&self) -> T;
}

pub(super) type OptimizationId = usize;

struct OptimizationItem<O> {
    graph: Vec<TensorOpsDescription>,
    end_condition: Vec<TensorOpsDescription>,
    ops: O,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{FloatOpsDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };

    #[test]
    fn should_cache_optimization_end_condition_forced() {
        // A graph with 3 ops.
        let graph = TestGraph::new(2);
        let mut path = OptimizationCache::new();

        // First following
        graph.follow_misses(&mut path);

        // Register the action.
        let optimization = path.complete(&Optimization1, graph.edges[0..2].to_vec(), None);

        assert_eq!(optimization, &Optimization1.create());

        // Second following on the same ops.
        path.reset();
        let result1 = path.follow(&[], EndCondition::NextOps(&graph.edges[0]));
        assert_eq!(result1, CacheResult::OnPath);

        let result2 = path.follow(&graph.edges[0..1], EndCondition::NextOps(&graph.edges[1]));
        assert_eq!(result2, CacheResult::OnPath);

        let result3 = path.follow(&graph.edges[0..2], EndCondition::Forced);
        match result3 {
            CacheResult::Found(ops) => assert_eq!(ops, &Optimization1.create()),
            _ => panic!("Should have found the cached operation"),
        };
    }

    #[test]
    fn once_found_perfect_should_always_return_found() {
        let mut graph = TestGraph::new(2);
        let mut path = OptimizationCache::new();
        graph.follow_misses(&mut path);

        // Register the action.
        let _optimization = path.complete(
            &Optimization1,
            graph.edges[0..1].to_vec(),
            Some(graph.edges[1].clone()),
        );

        path.reset();
        graph.new_ops();
        graph.new_ops();

        let result = path.follow(&[], EndCondition::NextOps(&graph.edges[0]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph.edges[0..1], EndCondition::NextOps(&graph.edges[1]));
        match result {
            CacheResult::Found(ops) => assert_eq!(ops, &Optimization1.create()),
            _ => panic!("Should have found the cached operation"),
        }

        let result = path.follow(&graph.edges[0..2], EndCondition::NextOps(&graph.edges[2]));
        match result {
            CacheResult::Found(ops) => assert_eq!(ops, &Optimization1.create()),
            _ => panic!("Should have found the cached operation"),
        }
    }

    #[test]
    fn should_cache_optimization_end_condition_next_ops() {
        // A graph with 4 ops.
        let graph = TestGraph::new(3);
        let mut path = OptimizationCache::new();

        // First following
        graph.follow_misses(&mut path);

        // Register the action.
        let optimization = path.complete(
            &Optimization1,
            graph.edges[0..2].to_vec(),
            Some(graph.edges[2].clone()),
        );

        assert_eq!(optimization, &Optimization1.create());

        // Second following on the same ops.
        path.reset();
        let result1 = path.follow(&[], EndCondition::NextOps(&graph.edges[0]));
        assert_eq!(result1, CacheResult::OnPath);

        let result2 = path.follow(&graph.edges[0..1], EndCondition::NextOps(&graph.edges[1]));
        assert_eq!(result2, CacheResult::OnPath);

        let result3 = path.follow(&graph.edges[0..2], EndCondition::NextOps(&graph.edges[2]));
        match result3 {
            CacheResult::Found(ops) => assert_eq!(ops, &Optimization1.create()),
            _ => panic!("Should have found the cached operation"),
        };
    }

    #[test]
    fn should_support_many_different_end_conditions() {
        let mut graph1 = TestGraph::new(2);
        graph1.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc)));

        let mut graph2 = TestGraph::new(2);
        graph2.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Log(desc)));

        let mut path = OptimizationCache::<String>::new();
        let last_edge_index = graph1.edges.len() - 1;

        // Follow graph 1 with only misses.
        graph1.follow_misses(&mut path);
        let _ = path.complete(
            &Optimization1,
            graph1.edges[0..last_edge_index].to_vec(),
            Some(graph1.edges[last_edge_index].clone()),
        );

        // Follow graph 2.
        let result = path.follow(&[], EndCondition::NextOps(&graph2.edges[0]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph2.edges[0..1], EndCondition::NextOps(&graph2.edges[1]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph2.edges[0..2], EndCondition::NextOps(&graph2.edges[2]));
        assert_eq!(result, CacheResult::Miss);

        let optimization = path.complete(
            &Optimization2,
            graph2.edges[0..last_edge_index].to_vec(),
            Some(graph2.edges[last_edge_index].clone()),
        );
        assert_eq!(
            optimization,
            &Optimization1.create(),
            "Optimization 1 should still be returned, since same graph but not same end condition."
        );
    }

    #[test]
    fn should_support_multiple_concurrent_paths() {
        // Two different graphs with a different second ops, but the same last ops.
        let mut graph1 = TestGraph::new(1);
        graph1.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc)));
        graph1.new_ops();

        let mut graph2 = TestGraph::new(1);
        graph2.register_ops(|desc| TensorOpsDescription::FloatOps(FloatOpsDescription::Cos(desc)));
        graph2.new_ops();

        let mut path = OptimizationCache::<String>::new();

        // Follow graph 1 with only misses.
        graph1.follow_misses(&mut path);

        // Register the opitmization 1 for graph 1.
        let last_edge_index = graph1.edges.len() - 1;
        let _ = path.complete(
            &Optimization1,
            graph1.edges[0..last_edge_index].to_vec(),
            Some(graph1.edges[last_edge_index].clone()),
        );

        // Follow graph 2 and register a new optimization.
        path.reset();

        let result = path.follow(&[], EndCondition::NextOps(&graph2.edges[0]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph2.edges[0..1], EndCondition::NextOps(&graph2.edges[1]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph2.edges[0..2], EndCondition::NextOps(&graph2.edges[2]));
        assert_eq!(
            result,
            CacheResult::Miss,
            "Should invalidate the second operation"
        );

        // Register new optimization for path 2.
        let _ = path.complete(
            &Optimization2,
            graph2.edges[0..last_edge_index].to_vec(),
            Some(graph2.edges[last_edge_index].clone()),
        );

        // Now let's validate that the cache works.

        // New path instance on graph 1.
        path.reset();

        let result = path.follow(&[], EndCondition::NextOps(&graph1.edges[0]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph1.edges[0..1], EndCondition::NextOps(&graph1.edges[1]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph1.edges[0..2], EndCondition::NextOps(&graph1.edges[2]));
        match result {
            CacheResult::Found(ops) => assert_eq!(ops, &Optimization1.create()),
            _ => panic!("Should have found the cached operation"),
        };

        // New path instance on graph 2.
        path.reset();

        let result = path.follow(&[], EndCondition::NextOps(&graph2.edges[0]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph2.edges[0..1], EndCondition::NextOps(&graph2.edges[1]));
        assert_eq!(result, CacheResult::OnPath);

        let result = path.follow(&graph2.edges[0..2], EndCondition::NextOps(&graph2.edges[2]));
        match result {
            CacheResult::Found(ops) => assert_eq!(ops, &Optimization2.create()),
            _ => panic!("Should have found the cached operation"),
        };
    }

    #[derive(Default, Debug)]
    struct TestGraph {
        nodes: Vec<TensorDescription>,
        edges: Vec<TensorOpsDescription>,
    }

    impl TestGraph {
        pub fn new(num_ops: usize) -> Self {
            let mut graph = Self::default();
            for _ in 0..num_ops {
                graph.new_ops();
            }

            graph
        }

        /// The first follow should only be cache miss.
        pub fn follow_misses(&self, path: &mut OptimizationCache<String>) {
            for i in 0..self.edges.len() {
                let result = path.follow(&self.edges[0..i], EndCondition::NextOps(&self.edges[i]));
                assert_eq!(result, CacheResult::Miss);
            }
        }

        pub fn register_ops<F>(&mut self, func: F)
        where
            F: Fn(UnaryOpsDescription) -> TensorOpsDescription,
        {
            self.new_empty_node();
            let desc = self.unary_description();
            self.edges.push(func(desc));
        }

        pub fn new_ops(&mut self) {
            if self.nodes.is_empty() {
                // Root node.
                self.new_empty_node();
            }

            // Out node.
            self.new_empty_node();

            self.edges
                .push(TensorOpsDescription::FloatOps(FloatOpsDescription::Log(
                    self.unary_description(),
                )));
        }

        fn new_empty_node(&mut self) {
            self.nodes.push(TensorDescription {
                id: TensorId::new(self.nodes.len() as u64),
                shape: vec![32, 32, 1],
                status: TensorStatus::NotInit,
            });
        }

        fn unary_description(&self) -> UnaryOpsDescription {
            let size = self.nodes.len();

            UnaryOpsDescription {
                input: self.nodes[size - 2].clone(),
                out: self.nodes[size - 1].clone(),
            }
        }
    }

    struct Optimization1;
    struct Optimization2;

    impl OptimizationFactory<String> for Optimization1 {
        fn create(&self) -> String {
            "Optimization1".to_string()
        }
    }

    impl OptimizationFactory<String> for Optimization2 {
        fn create(&self) -> String {
            "Optimization2".to_string()
        }
    }
}
