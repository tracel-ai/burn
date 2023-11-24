use super::TensorOpsDescription;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

pub struct Policy<O> {
    cache: HashMap<u64, Vec<CachedItem<O>>>,
}

pub(crate) trait ToBeCached<T> {
    /// Call only when not seen.
    fn build(&self) -> T;
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
    /// Continue exploring optimizations but using the [fusion ops builder](crate::FusionOps).
    Build,
    Wait,
    Execute(&'a T),
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

impl<Builder> CachedAction<&Builder> {
    pub fn build<T>(self) -> CachedAction<T>
    where
        Builder: ToBeCached<T>,
    {
        match self {
            CachedAction::Wait => CachedAction::Wait,
            CachedAction::Execute {
                optimization: item,
                next_possible_ops,
            } => CachedAction::Execute {
                optimization: item.build(),
                next_possible_ops,
            },
        }
    }
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
    pub fn merge<Builder: ToBeCached<T>>(mut self, other: CachedAction<&Builder>) -> Self {
        let (item_new, next_possible_ops_new) = match other {
            CachedAction::Wait => return self,
            CachedAction::Execute {
                optimization: item,
                next_possible_ops,
            } => (item, next_possible_ops),
        };

        match self {
            CachedAction::Wait => CachedAction::Execute {
                optimization: item_new.build(),
                next_possible_ops: next_possible_ops_new,
            },
            CachedAction::Execute {
                optimization: item,
                mut next_possible_ops,
            } => {
                let mut ops = next_possible_ops_new;
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

pub enum EndCondision<'a> {
    NextOps(&'a TensorOpsDescription),
    Forced,
}

impl<T> Policy<T> {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
    pub fn get<'a>(
        &'a self,
        key_: &GraphKey,
        graph: &[TensorOpsDescription],
        end_consition: EndCondision,
    ) -> Action<'a, T> {
        let key = key_.value();
        let values = match self.cache.get(&key) {
            Some(values) => values,
            None => return Action::Build,
        };

        let value = if values.len() > 1 {
            // Hash collision, find with graph.
            values.iter().find(|item| item.graph == graph).unwrap()
        } else {
            values.get(0).unwrap()
        };

        match &value.action {
            CachedAction::Wait => Action::Wait,
            CachedAction::Execute {
                optimization: item,
                next_possible_ops,
            } => match end_consition {
                EndCondision::NextOps(next_ops) => {
                    if next_possible_ops.contains(next_ops) {
                        Action::Execute(&item)
                    } else {
                        let mut next_key = key_.clone();
                        next_key.register(next_ops);
                        if self.cache.contains_key(&next_key.value()) {
                            Action::Wait
                        } else {
                            Action::Build
                        }
                    }
                }
                EndCondision::Forced => Action::Execute(&item),
            },
        }
    }

    pub fn insert<'a, Builder: ToBeCached<T>>(
        &'a mut self,
        key: &GraphKey,
        builder: &Builder,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) -> &'a T {
        let key = key.value();
        let mut hasher = DefaultHasher::new();
        let mut graph_current = Vec::new();

        for (i, node) in graph.iter().enumerate() {
            if i >= graph.len() + 1 {
                continue;
            }
            node.hash(&mut hasher);
            graph_current.push(node.clone());

            // Key and graph at this stage.
            let key = hasher.clone().finish();
            let graph = graph_current.clone();
            self.insert_action(key, CachedAction::<&Builder>::Wait, graph, false);
        }

        let (key, index) = self
            .insert_action(
                key,
                CachedAction::Execute {
                    optimization: builder,
                    next_possible_ops: next_ops.map(|ops| vec![ops]).unwrap_or_default(),
                },
                graph,
                true,
            )
            .unwrap();

        match &self.cache.get(&key).unwrap().get(index).unwrap().action {
            CachedAction::Wait => panic!("Should have saved an operation"),
            CachedAction::Execute {
                optimization,
                next_possible_ops: _,
            } => optimization,
        }
    }

    fn insert_action<Builder: ToBeCached<T>>(
        &mut self,
        key: u64,
        action: CachedAction<&Builder>,
        graph: Vec<TensorOpsDescription>,
        index: bool,
    ) -> Option<(u64, usize)> {
        let mut values = self.cache.remove(&key).unwrap_or_default();

        // Remove old entry.
        if !values.is_empty() {
            if let Some(existing) = values.iter_mut().find(|item| item.graph == graph) {
                // Update the action if the same graph, which probably mean a new end condition.
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

#[cfg(test)]
mod tests {
    use crate::{
        graph::{BaseOpsDescription, FloatOpsDescription, ReshapeDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };

    use super::*;

    struct Action1;
    impl ToBeCached<String> for Action1 {
        fn build(&self) -> String {
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
        cache.insert(
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

        let actual = cache.get(&key, &graph, EndCondision::NextOps(&ops2));
        let expected = Action::<String>::Wait;
        assert_eq!(expected, actual);

        key.register(&ops2);
        graph.push(ops2);

        let actual = cache.get(&key, &graph, EndCondision::NextOps(&ops3));
        let expected_ops = "Action1".to_string();
        let expected = Action::<String>::Execute(&expected_ops);
        assert_eq!(expected, actual);

        let actual = cache.get(&key, &graph, EndCondision::Forced);
        assert_eq!(expected, actual);
    }
}
