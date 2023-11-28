use super::starter::Starters;
use crate::graph::TensorOpsDescription;

pub struct OptimizationPath<O> {
    candidates: Vec<OptimizationId>,
    availables: Vec<(OptimizationId, usize)>,
    perfect: Option<OptimizationId>,
    cache: OptimizationCache<O>,
}

impl<O> OptimizationPath<O> {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            availables: Vec::new(),
            perfect: None,
            cache: OptimizationCache::new(),
        }
    }
    pub fn follow<'a>(
        &'a mut self,
        graph: &[TensorOpsDescription],
        end_condition: EndCondition,
    ) -> CacheResult<'a, O> {
        if graph.is_empty() {
            self.clear();
            // Starter
            let ops = match end_condition {
                EndCondition::NextOps(ops) => ops,
                EndCondition::Forced => return CacheResult::Miss, // Force en empty graph...
            };
            let candidates = self.cache.starters.get(&ops);
            if candidates.is_empty() {
                return CacheResult::Miss;
            }
            self.candidates = candidates;
            return CacheResult::OnPath;
        }

        if let Some(candidate) = self.perfect {
            return CacheResult::Found(&self.cache.optimizations.get(candidate).unwrap().ops);
        };

        // Invalidate candidates.
        let mut invalidated_candidate = Vec::new();
        for candidate in self.candidates.iter() {
            let graph_candidate = match self.cache.optimizations.get(*candidate) {
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
                        self.perfect = Some(*candidate);
                        return CacheResult::Found(&graph_candidate.ops);
                    }
                };

                if graph_candidate.end_condition.contains(ops) {
                    self.perfect = Some(*candidate);
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
            return CacheResult::Miss;
        } else {
            return CacheResult::OnPath;
        }
    }

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

        match existing_optim {
            Some((id, _)) => {
                let optimization = self.cache.optimizations.get_mut(*id).unwrap();
                match next_ops {
                    Some(ops) => optimization.end_condition.push(ops),
                    None => {}
                };

                return &optimization.ops;
            }
            None => {}
        };

        self.cache
            .starters
            .insert(graph.first().unwrap(), self.cache.optimizations.len());
        let ops = factory.create();
        let optimization = OptimizationItem {
            graph,
            end_condition: match next_ops {
                Some(val) => vec![val],
                None => Vec::new(),
            },
            ops,
        };

        self.cache.optimizations.push(optimization);
        &self.cache.optimizations.last().unwrap().ops
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.perfect = None;
    }
}

/// Guides what [action](Action) to take based on previously seems version of a graph.
///
/// It works by computing a [graph key](GraphKey) based on a relative version of a captured graph.
pub struct OptimizationCache<O> {
    starters: Starters,
    optimizations: Vec<OptimizationItem<O>>,
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

impl<T> OptimizationCache<T> {
    /// Create a new empty policy.
    pub fn new() -> Self {
        Self {
            starters: Starters::default(),
            optimizations: Vec::new(),
        }
    }
}

impl<'a, T> core::fmt::Debug for CacheResult<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheResult::Miss => f.write_str("Action::Build"),
            CacheResult::OnPath => f.write_str("Action::Wait"),
            CacheResult::Found(_) => f.write_str("Action::Execute"),
        }
    }
}

impl<T> Default for OptimizationCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Create an optimization.
pub trait OptimizationFactory<T> {
    /// Call only when a new optimization is found.
    fn create(&self) -> T;
}

pub(crate) type OptimizationId = usize;

struct OptimizationItem<O> {
    graph: Vec<TensorOpsDescription>,
    end_condition: Vec<TensorOpsDescription>,
    ops: O,
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::{BaseOpsDescription, FloatOpsDescription, ReshapeDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };

    use super::*;

    struct Action1;
    impl OptimizationFactory<String> for Action1 {
        fn create(&self) -> String {
            "Action1".to_string()
        }
    }
}
