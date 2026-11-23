use smallvec::smallvec;

use crate::{
    OpSharding,
    Sharding::{self, Replicated},
};

/// A gather_nd or scatter_nd, whose index tuples address the leading dims of its data and each
/// pick one entry of the trailing dims: a split of a trailing dim stays, carried by the entries
/// with the indices whole on every member, and anything else gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexNdRule {
    data: Sharding,
    entries: Sharding,
}

impl IndexNdRule {
    /// The tuples address `indexed_dims` dims of the data, and the entries' dims follow the
    /// `batch_dims` dims of the indices before the one holding the tuples.
    pub fn new(data: Sharding, indexed_dims: usize, batch_dims: usize) -> Self {
        match data.split_dim() {
            Some(dim) if dim >= indexed_dims => Self {
                data,
                entries: data.on_dim(batch_dims + dim - indexed_dims),
            },
            _ => Self {
                data: Replicated,
                entries: Replicated,
            },
        }
    }

    /// The data and the indices, gathered into the entries.
    pub fn gather_sharding(&self) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.data, Replicated],
            output: self.entries,
        }
    }

    /// The data, the indices and the entries, scattered into the data.
    pub fn scatter_sharding(&self) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.data, Replicated, self.entries],
            output: self.data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::Split;

    #[test]
    fn a_split_of_a_trailing_dim_moves_to_the_entries() {
        let rule = IndexNdRule::new(Split { dim: 2 }, 1, 2);

        assert_eq!(rule.gather_sharding().output, Split { dim: 3 });
        assert_eq!(rule.scatter_sharding().inputs[2], Split { dim: 3 });
        assert_eq!(rule.scatter_sharding().output, Split { dim: 2 });
    }

    #[test]
    fn a_split_of_an_addressed_dim_gathers() {
        let rule = IndexNdRule::new(Split { dim: 1 }, 2, 1);

        assert_eq!(rule.gather_sharding().inputs[..], [Replicated; 2]);
    }
}
