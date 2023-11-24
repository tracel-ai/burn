use super::TensorOpsDescription;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

/// Guides what [action](Action) to take based on previously seems version of a graph.
///
/// It works by computing a [graph key](GraphKey) based on a relative version of a captured graph.
pub struct Policy<O> {
    cache: HashMap<u64, Vec<CachedItem<O>>>,
}

/// Create an optimization.
pub trait OptimizationFactory<T> {
    /// Call only when a new optimization is found.
    fn create(&self) -> T;
}

/// The graph key keeps an upadted hash that represent the current graph.
#[derive(Default, Clone)]
pub struct GraphKey {
    hasher: DefaultHasher,
}

impl GraphKey {
    /// Register a new [ops](TensorOpsDescription) into the graph key.
    pub fn register(&mut self, desc: &TensorOpsDescription) {
        desc.hash(&mut self.hasher);
    }

    /// Clear the graph key state, should be called when starting a new relative graph.
    pub fn clear(&mut self) {
        self.hasher = DefaultHasher::default();
    }

    fn value(&self) -> u64 {
        self.hasher.finish()
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

// Cached Item
#[derive(new)]
struct CachedItem<O> {
    action: CachedAction<O>,
    graph: Vec<TensorOpsDescription>,
}

/// Cached action.
#[derive(Debug, PartialEq)]
enum CachedAction<O> {
    // In the path of finding an optimization that was already built.
    Wait,
    Execute {
        // Optimization to execute.
        optimization: O,
        // What are the possible next tensor operation that would indicate that we can't continue
        // fusing and that we should actually execute this optimization.
        next_possible_ops: Vec<TensorOpsDescription>,
    },
}

impl<T> CachedAction<T> {
    /// Multiple actions can be registered for the same graph, when multiple trajectories can lead
    /// to different optimizations.
    ///
    /// In this case, we have to keep the most relevant action or update the current one.
    fn merge<Factory: OptimizationFactory<T>>(self, other: CachedAction<&Factory>) -> Self {
        let (item_other, next_possible_ops_other) = match other {
            // The less informed action is to wait, so we discard it right away.
            CachedAction::Wait => return self,
            CachedAction::Execute {
                optimization: item,
                next_possible_ops,
            } => (item, next_possible_ops),
        };

        match self {
            // When the current action is to wait, we create a new Execute action with the data
            // provided by the other cached action.
            CachedAction::Wait => CachedAction::Execute {
                optimization: item_other.create(),
                next_possible_ops: next_possible_ops_other,
            },
            // When both actions have opitmizations, it means that the same optimization should be
            // taken when followed by different operations, so we simply merge the two
            // `next_possible_ops` together without duplicates.
            //
            // We keep the old optimization and avoid `creating` a new one, which is more
            // efficient (avoid potential many compilation steps).
            CachedAction::Execute {
                optimization: item,
                mut next_possible_ops,
            } => {
                let mut ops = next_possible_ops_other;
                for o in next_possible_ops.drain(..) {
                    if !ops.contains(&o) {
                        ops.push(o);
                    }
                }
                CachedAction::Execute {
                    optimization: item,
                    next_possible_ops: ops,
                }
            }
        }
    }
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
            cache: HashMap::new(),
        }
    }

    /// Compute the next [action](Action) to be taken.
    pub fn action<'a>(
        &'a self,
        key: &GraphKey,
        graph: &[TensorOpsDescription],
        end_condition: EndCondition,
    ) -> Action<'a, T> {
        let values = match self.cache.get(&key.value()) {
            Some(values) => values,
            None => return Action::Build,
        };

        let value = if values.len() > 1 {
            // Hash collision, find with graph.
            values
                .iter()
                .find(|item| item.graph == graph)
                .expect("When a collision happens, an entry with the same graph must be present.")
        } else {
            values
                .get(0)
                .expect("We never happen an empty list to the cache.")
        };

        match &value.action {
            CachedAction::Wait => Action::Wait,
            CachedAction::Execute {
                optimization: item,
                next_possible_ops,
            } => match end_condition {
                EndCondition::NextOps(next_ops) => {
                    if next_possible_ops.contains(next_ops) {
                        // When the end condition is validated, we execute the action.
                        Action::Execute(&item)
                    } else {
                        // Otherwise we check if the next operation is registered in the graph, so
                        // that we can wait for a following better optimization.
                        let mut next_key = key.clone();
                        next_key.register(next_ops);
                        if self.cache.contains_key(&next_key.value()) {
                            Action::Wait
                        } else {
                            // If not found, we have to create a new set of actions for this graph.
                            Action::Build
                        }
                    }
                }
                EndCondition::Forced => Action::Execute(&item),
            },
        }
    }

    /// Register a new optimization for the given graph and next operation.
    pub fn register<'a, Factory: OptimizationFactory<T>>(
        &'a mut self,
        key: &GraphKey,
        factory: &Factory,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) -> &'a T {
        // First we have to determine each action to be taken when a fraction of the graph is seen.
        // So we simulate a graph traversal with the correct hash key.
        let mut current_key = GraphKey::default();
        let mut current_graph = Vec::new();

        for (i, node) in graph.iter().enumerate() {
            // The last graph is a particular case where the action should be taken.
            if i == graph.len() - 1 {
                break;
            }

            current_key.register(&node);
            current_graph.push(node.clone());

            // We insert a cache action to the corresponding key where we wait for the potential
            // optimization to be executed.
            self.insert_action(
                current_key.value(),
                CachedAction::<&Factory>::Wait,
                current_graph.clone(),
                false, // We don't need to return the optimization.
            );
        }

        // We finally insert the given optimization to the graph.
        let (key, index) = self
            .insert_action(
                key.value(),
                CachedAction::Execute {
                    optimization: factory,
                    next_possible_ops: next_ops.map(|ops| vec![ops]).unwrap_or_default(),
                },
                graph,
                true,
            )
            .unwrap();

        // We retrieve the optimization from the cache so that it can be executed right away.
        match &self
            .cache
            .get(&key)
            .expect("Just saved the action")
            .get(index)
            .expect("The index given should be valid")
            .action
        {
            CachedAction::Wait => panic!("Should have saved an operation"),
            CachedAction::Execute {
                optimization,
                next_possible_ops: _,
            } => optimization,
        }
    }

    fn insert_action<Factory: OptimizationFactory<T>>(
        &mut self,
        key: u64,
        action: CachedAction<&Factory>,
        graph: Vec<TensorOpsDescription>,
        index: bool,
    ) -> Option<(u64, usize)> {
        let mut values = self.cache.remove(&key).unwrap_or_default();

        // Remove old entry.
        if !values.is_empty() {
            if let Some(existing) = values.iter_mut().find(|item| item.graph == graph) {
                // When a graph already exist, it means that we should merge the two action.
                let mut action_tmp = CachedAction::Wait;
                core::mem::swap(&mut action_tmp, &mut existing.action);
                existing.action = action_tmp.merge(action);
            } else {
                // Hash collision, new action with same Hash.
                values.push(CachedItem::new(action.build(), graph));
            }
        } else {
            // New action.
            values.push(CachedItem::new(action.build(), graph));
        }

        if !index {
            self.cache.insert(key, values);
            None
        } else {
            let returned = (key, values.len() - 1);
            self.cache.insert(key, values);
            Some(returned)
        }
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

impl<Factory> CachedAction<&Factory> {
    fn build<T>(self) -> CachedAction<T>
    where
        Factory: OptimizationFactory<T>,
    {
        match self {
            CachedAction::Wait => CachedAction::Wait,
            CachedAction::Execute {
                optimization: item,
                next_possible_ops,
            } => CachedAction::Execute {
                optimization: item.create(),
                next_possible_ops,
            },
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
        let mut key = GraphKey::default();

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

        key.register(&ops1);
        key.register(&ops2);
        cache.register(
            &key,
            &Action1,
            vec![ops1.clone(), ops2.clone()],
            Some(ops3.clone()),
        );
        // Second run.
        let mut key = GraphKey::default();
        let mut graph = Vec::new();

        key.register(&ops1);
        graph.push(ops1);

        let actual = cache.action(&key, &graph, EndCondition::NextOps(&ops2));
        let expected = Action::<String>::Wait;
        assert_eq!(expected, actual);

        key.register(&ops2);
        graph.push(ops2);

        let actual = cache.action(&key, &graph, EndCondition::NextOps(&ops3));
        let expected_ops = "Action1".to_string();
        let expected = Action::<String>::Execute(&expected_ops);
        assert_eq!(expected, actual);

        let actual = cache.action(&key, &graph, EndCondition::Forced);
        assert_eq!(expected, actual);
    }
}
