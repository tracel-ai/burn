use smallvec::smallvec;

use crate::{
    OpSharding,
    Sharding::{self, PartialSum, Replicated, Split},
};

/// An embedding's table, its indices, and its rows, which are its output or, for its table's
/// gradient, that output's gradient. A split of the table's columns splits the rows' last dim,
/// and one of the indices splits the rows alike. A split of the table's rows gathers: a member
/// would need every row its indices name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddingRule {
    table: Sharding,
    indices: Sharding,
    rows: Sharding,
}

impl EmbeddingRule {
    /// The rule of an embedding, its indices of `num_index_dims` dims.
    pub fn new(table: Sharding, indices: Sharding, num_index_dims: usize) -> Self {
        match (table.split_dim(), indices.split_dim()) {
            (Some(1), _) => Self {
                table,
                indices: Replicated,
                rows: table.on_dim(num_index_dims),
            },
            (_, Some(_)) => Self {
                table: Replicated,
                indices,
                rows: indices,
            },
            _ => Self::gathered(),
        }
    }

    /// The rule of a table's gradient, from the table, the rows' gradient and the indices. A split
    /// along the indices' dims sums each member's rows into a whole table, so it must cut each row
    /// to one member.
    pub fn backward(
        table: Sharding,
        rows: Sharding,
        indices: Sharding,
        num_index_dims: usize,
    ) -> Self {
        match (table.split_dim(), rows.split_dim(), rows, indices) {
            (Some(1), ..) => Self::new(table, Replicated, num_index_dims),
            (_, Some(dim), ..) if dim == num_index_dims => Self {
                table: rows.on_dim(1),
                indices: Replicated,
                rows,
            },
            (_, Some(_), Split { .. }, _) => Self {
                table: Replicated,
                indices: rows,
                rows,
            },
            (_, _, _, Split { .. }) => Self::new(Replicated, indices, num_index_dims),
            _ => Self::gathered(),
        }
    }

    /// For the table and indices, then the rows.
    pub fn sharding(&self) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.table, self.indices],
            output: self.rows,
        }
    }

    /// For the table, rows and indices, then the table's gradient.
    pub fn grad_sharding(&self) -> OpSharding {
        let grad = match (self.table, self.indices) {
            (Replicated, Replicated) => Replicated,
            (Replicated, _) => PartialSum,
            (table, _) => table,
        };
        OpSharding {
            inputs: smallvec![self.table, self.rows, self.indices],
            output: grad,
        }
    }

    fn gathered() -> Self {
        Self {
            table: Replicated,
            indices: Replicated,
            rows: Replicated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::ReplicatedSplit;

    #[test]
    fn a_split_of_the_columns_splits_the_rows_last_dim() {
        let sharding = EmbeddingRule::new(Split { dim: 1 }, Split { dim: 0 }, 2).sharding();

        assert_eq!(sharding.inputs[..], [Split { dim: 1 }, Replicated]);
        assert_eq!(sharding.output, Split { dim: 2 });
    }

    #[test]
    fn a_split_of_the_indices_splits_the_rows_alike() {
        let sharding = EmbeddingRule::new(Split { dim: 0 }, Split { dim: 1 }, 2).sharding();

        assert_eq!(sharding.inputs[..], [Replicated, Split { dim: 1 }]);
        assert_eq!(sharding.output, Split { dim: 1 });
    }

    #[test]
    fn a_gradient_split_by_columns_stays_split_and_one_split_by_index_is_a_partial_sum() {
        assert_eq!(
            EmbeddingRule::backward(Replicated, Split { dim: 2 }, Replicated, 2)
                .grad_sharding()
                .output,
            Split { dim: 1 }
        );
        assert_eq!(
            EmbeddingRule::backward(Replicated, Split { dim: 0 }, Replicated, 2)
                .grad_sharding()
                .inputs[..],
            [Replicated, Split { dim: 0 }, Split { dim: 0 }]
        );
        assert_eq!(
            EmbeddingRule::backward(Replicated, Split { dim: 0 }, Replicated, 2)
                .grad_sharding()
                .output,
            PartialSum
        );
    }

    #[test]
    fn a_whole_gradient_follows_the_split_of_the_table_or_of_the_indices() {
        let by_columns = EmbeddingRule::backward(Split { dim: 1 }, Replicated, Replicated, 2);
        let by_index = EmbeddingRule::backward(Replicated, Replicated, Split { dim: 1 }, 2);

        assert_eq!(by_columns.grad_sharding().inputs[1], Split { dim: 2 });
        assert_eq!(by_columns.grad_sharding().output, Split { dim: 1 });
        assert_eq!(by_index.grad_sharding().inputs[1], Split { dim: 1 });
        assert_eq!(by_index.grad_sharding().output, PartialSum);
    }

    #[test]
    fn a_gradient_split_by_index_in_groups_gathers() {
        let rows = ReplicatedSplit { dim: 0, groups: 2 };

        assert_eq!(
            EmbeddingRule::backward(Replicated, rows, Replicated, 2)
                .grad_sharding()
                .output,
            Replicated
        );
    }
}
