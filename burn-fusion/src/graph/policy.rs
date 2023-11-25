use super::TensorOpsDescription;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

/// Guides what [action](Action) to take based on previously seems version of a graph.
///
/// It works by computing a [graph key](GraphKey) based on a relative version of a captured graph.
pub struct Policy<O> {
    starters: Starters,
    optimizations: Vec<OptimizationItem<O>>,
}

#[derive(Default)]
pub struct Starters {
    starter_indices: HashMap<u64, Vec<(TensorOpsDescription, usize)>>,
    starters: Vec<Vec<OptimizationId>>,
}

impl Starters {
    pub fn get(&self, ops: &TensorOpsDescription) -> Vec<OptimizationId> {
        let key = self.graph_key(ops);
        let values = match self.starter_indices.get(&key) {
            Some(val) => val,
            None => return Vec::new(),
        };

        if values.is_empty() {
            return Vec::new();
        }

        let (_, index) = match values.iter().find(|value| &value.0 == ops) {
            Some(val) => val,
            None => return Vec::new(),
        };

        let val = match self.starters.get(*index) {
            Some(value) => value.clone(),
            None => Vec::new(),
        };

        val
    }

    pub fn insert(&mut self, ops: &TensorOpsDescription, new_id: OptimizationId) {
        let key = self.graph_key(ops);
        let values = match self.starter_indices.get_mut(&key) {
            Some(val) => val,
            None => {
                // New starter ops.
                let index = self.starters.len();
                self.starters.push(vec![new_id]);
                self.starter_indices.insert(key, vec![(ops.clone(), index)]);

                return;
            }
        };
        let (_, index) = match values.iter_mut().find(|value| &value.0 == ops) {
            Some(val) => val,
            None => {
                // New with hash collision.
                let index = self.starters.len();
                self.starters.push(vec![new_id]);
                values.push((ops.clone(), index));
                return;
            }
        };

        // New optimization for an existing starter.
        self.starters
            .get_mut(*index)
            .expect("Should exist")
            .push(new_id);
    }

    fn graph_key(&self, ops: &TensorOpsDescription) -> u64 {
        let mut hasher = DefaultHasher::new();
        ops.hash(&mut hasher);
        hasher.finish()
    }
}

/// Create an optimization.
pub trait OptimizationFactory<T> {
    /// Call only when a new optimization is found.
    fn create(&self) -> T;
}

type OptimizationId = usize;

pub struct OptimizationItem<O> {
    graph: Vec<TensorOpsDescription>,
    end_condition: Vec<TensorOpsDescription>,
    ops: O,
}

#[derive(Default, Clone)]
pub struct OptimizationPath {
    candidates: Vec<OptimizationId>,
    availables: Vec<(OptimizationId, usize)>,
    perfect: Option<OptimizationId>,
}

impl OptimizationPath {
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.availables.clear();
        self.perfect = None;
    }
}

/// Action to be made depending on the graph.
#[derive(PartialEq, Eq)]
pub enum Action<'a, T> {
    /// Continue exploring optimizations but using the [fusion ops builder](crate::FusionOpsBuilder).
    Build,
    /// The current graph indicates that some optimization maybe possible in the future, so the
    /// best action is to wait for the optimization to become available.
    ///
    /// Sometime, if can be a false positive and a new opitmization should be built from scratch,
    /// therefore it is important to keep the previous operations to rebuilt the state if it
    /// happens.
    Wait,
    /// An optimization has been found, and the best action is to execute it!
    Execute(&'a T),
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
pub enum EndCondition<'a> {
    /// The next operation that signal the end of the operation.
    NextOps(&'a TensorOpsDescription),
    /// When forced, we should execute the optimization if found no matter what comes next.
    Forced,
}

impl<T> Policy<T> {
    /// Create a new empty policy.
    pub fn new() -> Self {
        Self {
            starters: Starters::default(),
            optimizations: Vec::new(),
        }
    }

    /// Compute the next [action](Action) to be taken.
    pub fn action<'a>(
        &'a self,
        path: &mut OptimizationPath,
        graph: &[TensorOpsDescription],
        end_condition: EndCondition,
    ) -> Action<'a, T> {
        if let Some(candidate) = path.perfect {
            return Action::Execute(&self.optimizations.get(candidate).unwrap().ops);
        };

        if graph.is_empty() && path.candidates.is_empty() {
            // Starter
            let ops = match end_condition {
                EndCondition::NextOps(ops) => ops,
                EndCondition::Forced => return Action::Build, // Force en empty graph...
            };
            let candidates = self.starters.get(&ops);
            if candidates.is_empty() {
                return Action::Build;
            }
            path.candidates = candidates;
            return Action::Wait;
        }

        // Invalidate candidates.
        let mut invalidated_candidate = Vec::new();
        for candidate in path.candidates.iter() {
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
                        path.perfect = Some(*candidate);
                        return Action::Execute(&graph_candidate.ops);
                    }
                };

                if graph_candidate.end_condition.contains(ops) {
                    path.perfect = Some(*candidate);
                    return Action::Execute(&graph_candidate.ops);
                } else {
                    path.availables.push((*candidate, graph.len()));
                    invalidated_candidate.push(*candidate);
                }
            }
        }

        let mut updated_candidates = Vec::new();
        core::mem::swap(&mut updated_candidates, &mut path.candidates);

        path.candidates = updated_candidates
            .into_iter()
            .filter(|candidate| !invalidated_candidate.contains(candidate))
            .collect();

        if path.candidates.is_empty() {
            return Action::Build;
        } else {
            return Action::Wait;
        }
    }

    /// Register a new optimization for the given graph and next operation.
    pub fn register_new<'a, Factory: OptimizationFactory<T>>(
        &'a mut self,
        path: &mut OptimizationPath,
        factory: &Factory,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) -> &'a T {
        let existing_optim = path
            .availables
            .iter()
            .find(|(_candidate, len)| *len == graph.len());

        match existing_optim {
            Some((id, _)) => {
                let optimization = self.optimizations.get_mut(*id).unwrap();
                match next_ops {
                    Some(ops) => optimization.end_condition.push(ops),
                    None => {}
                };

                return &optimization.ops;
            }
            None => {}
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
}

impl<'a, T> core::fmt::Debug for Action<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Build => f.write_str("Action::Build"),
            Action::Wait => f.write_str("Action::Wait"),
            Action::Execute(_) => f.write_str("Action::Execute"),
        }
    }
}

impl<T> Default for Policy<T> {
    fn default() -> Self {
        Self::new()
    }
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

    #[test]
    fn can_register_ops_for_a_graph() {
        let mut cache = Policy::<String>::new();
        let mut key = OptimizationPath::default();

        let ops1 = TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(UnaryOpsDescription {
            input: TensorDescription {
                id: TensorId::new(1),
                shape: vec![32, 64],
                status: TensorStatus::ReadOnly,
            },
            out: TensorDescription {
                id: TensorId::new(2),
                shape: vec![32, 64],
                status: TensorStatus::ReadOnly,
            },
        }));
        let ops2 = TensorOpsDescription::FloatOps(FloatOpsDescription::Log(UnaryOpsDescription {
            input: TensorDescription {
                id: TensorId::new(2),
                shape: vec![32, 64],
                status: TensorStatus::ReadOnly,
            },
            out: TensorDescription {
                id: TensorId::new(3),
                shape: vec![32, 64],
                status: TensorStatus::ReadOnly,
            },
        }));
        let ops3 =
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::Reshape(ReshapeDescription {
                input: TensorDescription {
                    id: TensorId::new(3),
                    shape: vec![32, 64],
                    status: TensorStatus::ReadOnly,
                },
                out: TensorDescription {
                    id: TensorId::new(4),
                    shape: vec![32, 2, 32],
                    status: TensorStatus::ReadOnly,
                },
            }));

        cache.register_new(
            &mut key,
            &Action1,
            vec![ops1.clone(), ops2.clone()],
            Some(ops3.clone()),
        );
        // Second run.
        let mut key = OptimizationPath::default();
        let mut graph = Vec::new();

        graph.push(ops1);

        let actual = cache.action(&mut key, &graph, EndCondition::NextOps(&ops2));
        let expected = Action::<String>::Wait;

        graph.push(ops2);

        let actual = cache.action(&mut key, &graph, EndCondition::NextOps(&ops3));
        let expected_ops = "Action1".to_string();
        let expected = Action::<String>::Execute(&expected_ops);

        let actual = cache.action(&mut key, &graph, EndCondition::Forced);
    }
}
