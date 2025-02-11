use crate::stream::store::ExecutionPlanId;
use burn_ir::OperationIr;
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

/// Index used to search optimizations.
#[derive(Default, Serialize, Deserialize, Clone)]
pub struct ExecutionPlanIndex {
    /// We can't use `HashMap<OperationIr, Vec<ExecutionPlanId>>` since `OperationIr`
    /// doesn't implement [`Eq`](core::cmp::Eq).
    ///
    /// `OperationIr` can't implement `Eq` since float types don't implement it.
    ///
    /// We rely instead on [`PartialEq`](core::cmp::PartialEq) to manually handle hash collisions.
    /// This is OK because we use `relative` operations where any scalar values are set to zeros,
    /// see [`RelativeStreamConverter`](crate::stream::RelativeStreamConverter).
    ///
    /// Map from the hash of the `OperationIr` to a list of `(OperationIr, index)` pairs,
    /// where `index` is the index of all the execution plans that start with the `OperationIr`
    /// in the `starters` list.
    mapping: HashMap<u64, Vec<(OperationIr, usize)>>,
    starters: Vec<Vec<ExecutionPlanId>>,
}

pub enum SearchQuery<'a> {
    PlansStartingWith(&'a OperationIr),
}

pub enum InsertQuery<'a> {
    NewPlan {
        operations: &'a [OperationIr],
        id: ExecutionPlanId,
    },
}

impl ExecutionPlanIndex {
    /// Search optimizations with the given [query](SearchQuery).
    pub fn find(&self, query: SearchQuery<'_>) -> Vec<ExecutionPlanId> {
        match query {
            SearchQuery::PlansStartingWith(ops) => self.find_starting_with(ops),
        }
    }

    /// Register a new optimization with the given [query](InsertQuery).
    pub fn insert(&mut self, query: InsertQuery<'_>) {
        match query {
            InsertQuery::NewPlan { operations, id } => {
                if let Some(operation) = operations.first() {
                    self.insert_new_operation(operation, id)
                }
            }
        }
    }

    /// Find execution plans starting with the `OperationIr`
    fn find_starting_with(&self, operation: &OperationIr) -> Vec<ExecutionPlanId> {
        let key = self.operation_key(operation);
        let values = match self.mapping.get(&key) {
            Some(val) => val,
            None => return Vec::new(),
        };

        if values.is_empty() {
            return Vec::new();
        }

        let (_, index) = match values.iter().find(|value| &value.0 == operation) {
            Some(val) => val,
            None => return Vec::new(),
        };

        let val = match self.starters.get(*index) {
            Some(value) => value.clone(),
            None => Vec::new(),
        };

        val
    }

    /// Update the index for an execution plan starting with operation `ops`
    fn insert_new_operation(&mut self, ops: &OperationIr, new_id: ExecutionPlanId) {
        let key = self.operation_key(ops);
        let values = match self.mapping.get_mut(&key) {
            Some(val) => val,
            None => {
                // New starter ops.
                let index = self.starters.len();
                self.starters.push(vec![new_id]);
                self.mapping.insert(key, vec![(ops.clone(), index)]);

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

    // Hash the value of the first operation in a list.
    fn operation_key(&self, ops: &OperationIr) -> u64 {
        let mut hasher = DefaultHasher::new();
        ops.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use burn_ir::{BinaryOpIr, NumericOperationIr, ScalarOpIr, TensorId, TensorIr, TensorStatus};
    use burn_tensor::DType;

    use super::*;

    #[test]
    fn should_find_optimization_id_based_on_tensor_ops() {
        let mut index = ExecutionPlanIndex::default();
        let stream_1 = [ops_1()];
        let optimization_id_1 = 0;

        index.insert(InsertQuery::NewPlan {
            operations: &stream_1,
            id: optimization_id_1,
        });

        let found = index.find(SearchQuery::PlansStartingWith(&stream_1[0]));

        assert_eq!(found, vec![optimization_id_1]);
    }

    #[test]
    fn should_support_multiple_optimization_ids_with_same_starting_ops() {
        let mut index = ExecutionPlanIndex::default();
        let stream_1 = [ops_1(), ops_2(), ops_1()];
        let stream_2 = [ops_1(), ops_1(), ops_2()];
        let optimization_id_1 = 0;
        let optimization_id_2 = 1;

        index.insert(InsertQuery::NewPlan {
            operations: &stream_1,
            id: optimization_id_1,
        });
        index.insert(InsertQuery::NewPlan {
            operations: &stream_2,
            id: optimization_id_2,
        });

        let found = index.find(SearchQuery::PlansStartingWith(&stream_1[0]));

        assert_eq!(found, vec![optimization_id_1, optimization_id_2]);
    }

    #[test]
    fn should_only_find_optimization_with_correct_starting_ops() {
        let mut index = ExecutionPlanIndex::default();
        let stream_1 = [ops_1(), ops_1()];
        let stream_2 = [ops_2(), ops_1()];
        let optimization_id_1 = 0;
        let optimization_id_2 = 1;

        index.insert(InsertQuery::NewPlan {
            operations: &stream_1,
            id: optimization_id_1,
        });
        index.insert(InsertQuery::NewPlan {
            operations: &stream_2,
            id: optimization_id_2,
        });

        let found = index.find(SearchQuery::PlansStartingWith(&stream_1[0]));

        assert_eq!(found, vec![optimization_id_1]);
    }

    #[test]
    fn should_handle_hash_collisions() {
        let mut index = ExecutionPlanIndex::default();
        let stream_1 = [ops_1(), ops_1()];
        let stream_2 = [ops_3(), ops_1()];
        let optimization_id_1 = 0;
        let optimization_id_2 = 1;

        let stream_1_key = index.operation_key(&stream_1[0]);
        let stream_2_key = index.operation_key(&stream_2[0]);

        assert_eq!(
            stream_1_key, stream_2_key,
            "Ops 1 and Ops 3 have the same hash"
        );
        assert_ne!(stream_1[0], stream_2[0], "Ops 1 and Ops 3 are different.");

        index.insert(InsertQuery::NewPlan {
            operations: &stream_1,
            id: optimization_id_1,
        });
        index.insert(InsertQuery::NewPlan {
            operations: &stream_2,
            id: optimization_id_2,
        });

        let found = index.find(SearchQuery::PlansStartingWith(&stream_1[0]));

        assert_eq!(found, vec![optimization_id_1]);
    }

    fn ops_1() -> OperationIr {
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: TensorIr {
                    id: TensorId::new(0),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                    dtype: DType::F32,
                },
                rhs: TensorIr {
                    id: TensorId::new(1),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                    dtype: DType::F32,
                },
                out: TensorIr {
                    id: TensorId::new(2),
                    shape: vec![32, 32],
                    status: TensorStatus::NotInit,
                    dtype: DType::F32,
                },
            }),
        )
    }

    fn ops_2() -> OperationIr {
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::AddScalar(ScalarOpIr {
                lhs: TensorIr {
                    id: TensorId::new(0),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                    dtype: DType::F32,
                },
                rhs: 5.0,
                out: TensorIr {
                    id: TensorId::new(2),
                    shape: vec![32, 32],
                    status: TensorStatus::NotInit,
                    dtype: DType::F32,
                },
            }),
        )
    }

    fn ops_3() -> OperationIr {
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Sub(BinaryOpIr {
                lhs: TensorIr {
                    id: TensorId::new(0),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                    dtype: DType::F32,
                },
                rhs: TensorIr {
                    id: TensorId::new(1),
                    shape: vec![32, 32],
                    status: TensorStatus::ReadOnly,
                    dtype: DType::F32,
                },
                out: TensorIr {
                    id: TensorId::new(2),
                    shape: vec![32, 32],
                    status: TensorStatus::NotInit,
                    dtype: DType::F32,
                },
            }),
        )
    }
}
