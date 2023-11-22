use super::TensorOpsDescription;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

#[derive(Default)]
pub struct Cache<T> {
    state: HashMap<u64, Vec<Item<T>>>,
}

pub trait ToBeCached<T> {
    /// Call only when not seen.
    fn build(self) -> T;
}

#[derive(Default, Clone)]
pub struct CacheKey {
    hasher: DefaultHasher,
}

impl CacheKey {
    pub fn register(&mut self, desc: &TensorOpsDescription) {
        desc.hash(&mut self.hasher);
    }

    fn value(&self) -> u64 {
        self.hasher.finish()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action<'a, T> {
    BuildFusionOps,
    WaitForFusionOps,
    ExecuteFusionOps(&'a T),
}

#[derive(Debug, PartialEq)]
enum ActionItem<T> {
    WaitForFusionOps,
    ExecuteFusionOps {
        item: T,
        next_possible_ops: Vec<TensorOpsDescription>,
    },
}

#[derive(new)]
struct Item<T> {
    action: ActionItem<T>,
    graph: Vec<TensorOpsDescription>,
}

// (Log Exp) [Reshape].
// (Log Exp) Add [Matmul, Reshape].
//
// (Log Exp) [Add]
impl<T> ActionItem<T> {
    pub fn merge<Builder: ToBeCached<T>>(&mut self, other: ActionItem<Builder>) {
        let (item_new, next_possible_ops_new) = match other {
            ActionItem::WaitForFusionOps => return,
            ActionItem::ExecuteFusionOps {
                item,
                next_possible_ops,
            } => (item, next_possible_ops),
        };

        let updated_action = match self {
            ActionItem::WaitForFusionOps => ActionItem::ExecuteFusionOps {
                item: item_new.build(),
                next_possible_ops: next_possible_ops_new,
            },
            ActionItem::ExecuteFusionOps {
                item,
                next_possible_ops,
            } => {
                let mut ops = next_possible_ops_new;
                for o in next_possible_ops.drain(..) {
                    if !ops.contains(&o) {
                        ops.push(o);
                    }
                }
                ActionItem::ExecuteFusionOps {
                    item: *item,
                    next_possible_ops: ops,
                }
            }
        };

        *self = updated_action;
    }
}

pub enum EndCondision<'a> {
    NextOps(&'a TensorOpsDescription),
    Forced,
}

impl<T> Cache<T> {
    pub fn get<'a>(
        &'a self,
        key_: &CacheKey,
        graph: &[TensorOpsDescription],
        end_consition: EndCondision,
    ) -> Action<'a, T> {
        let key = key_.value();
        let values = match self.state.get(&key) {
            Some(values) => values,
            None => return Action::BuildFusionOps,
        };

        let value = if values.len() > 1 {
            // Hash collision, find with graph.
            values.iter().find(|item| item.graph == graph).unwrap()
        } else {
            values.get(0).unwrap()
        };

        match &value.action {
            ActionItem::WaitForFusionOps => Action::WaitForFusionOps,
            ActionItem::ExecuteFusionOps {
                item,
                next_possible_ops,
            } => match end_consition {
                EndCondision::NextOps(next_ops) => {
                    if next_possible_ops.contains(next_ops) {
                        Action::ExecuteFusionOps(&item)
                    } else {
                        let mut next_key = key_.clone();
                        next_key.register(next_ops);
                        if self.state.contains_key(&next_key.value()) {
                            Action::WaitForFusionOps
                        } else {
                            Action::BuildFusionOps
                        }
                    }
                }
                EndCondision::Forced => Action::ExecuteFusionOps(&item),
            },
        }
    }

    pub fn insert<Builder: ToBeCached<T>>(
        &mut self,
        key: &CacheKey,
        builder: Builder,
        graph: Vec<TensorOpsDescription>,
        next_ops: Option<TensorOpsDescription>,
    ) {
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
            self.insert_action(key, ActionItem::<Builder>::WaitForFusionOps, graph);
        }

        self.insert_action(
            key,
            ActionItem::ExecuteFusionOps {
                item: builder,
                next_possible_ops: next_ops.map(|ops| vec![ops]).unwrap_or_default(),
            },
            graph,
        );
    }

    fn insert_action<Builder: ToBeCached<T>>(
        &mut self,
        key: u64,
        action: ActionItem<Builder>,
        graph: Vec<TensorOpsDescription>,
    ) {
        let mut values = self.state.remove(&key).unwrap_or_default();

        // Remove old entry.
        if !values.is_empty() {
            if let Some(existing) = values.iter_mut().find(|item| item.graph == graph) {
                // Update the action if the same graph, which probably mean a new end condition.
                existing.action.merge(action);
            } else {
                // Hash collision, new action with same Hash.
                values.push(Item::new(action, graph));
            }
        } else {
            // New action.
            values.push(Item::new(action, graph));
        }

        self.state.insert(key, values);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::{BaseOpsDescription, FloatOpsDescription, ReshapeDescription, UnaryOpsDescription},
        TensorDescription, TensorId, TensorStatus,
    };

    use super::*;

    #[test]
    fn can_register_ops_for_a_graph() {
        let mut cache = Cache::<String>::default();
        let mut key = CacheKey::default();

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
                shape: vec![32, 2, 32],
            }));

        key.register(&ops1);
        key.register(&ops2);
        cache.insert(
            &key,
            "Ops1 + Ops2 with end condition Ops3".to_string(),
            vec![ops1.clone(), ops2.clone()],
            Some(ops3.clone()),
        );
        // Second run.
        let mut key = CacheKey::default();
        let mut graph = Vec::new();

        key.register(&ops1);
        graph.push(ops1);

        let actual = cache.get(&key, &graph, EndCondision::NextOps(&ops2));
        let expected = Action::<String>::WaitForFusionOps;
        assert_eq!(expected, actual);

        key.register(&ops2);
        graph.push(ops2);

        let actual = cache.get(&key, &graph, EndCondision::NextOps(&ops3));
        let expected_ops = "Ops1 + Ops2 with end condition Ops3".to_string();
        let expected = Action::<String>::ExecuteFusionOps(&expected_ops);
        assert_eq!(expected, actual);

        let actual = cache.get(&key, &graph, EndCondision::Forced);
        assert_eq!(expected, actual);
    }
}
