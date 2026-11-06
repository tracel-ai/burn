use burn_backend::ops::attention::AttentionShapes;
use burn_std::Shape;

use crate::{
    ElementwiseRule, OpSharding,
    Sharding::{self, Replicated, Split},
};

/// Which dim of an attention is split. Every sequence of the batch and every query head attends
/// apart, so a split of either needs no communication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionRule {
    /// Nothing is split: every member attends the whole.
    Unsplit,
    /// Each member attends its own sequences.
    Batch,
    /// Megatron's layout: each member attends its own query heads, with the key and value heads
    /// they read.
    Heads,
}

impl AttentionRule {
    /// The shardings of the query, key and value: a split along the heads picks the heads when
    /// their groups split together, else a split along the batch picks the batch.
    pub fn new(inputs: [Sharding; 3], shapes: &AttentionShapes, num_members: usize) -> Self {
        Self::of_heads(inputs, shapes.q_heads, shapes.kv_heads, num_members)
    }

    /// As [`new`](Self::new), from the query's and the key's head counts alone: for an attention
    /// over a key/value cache, whose pools are not laid out as a key.
    pub fn of_heads(
        inputs: [Sharding; 3],
        q_heads: usize,
        kv_heads: usize,
        num_members: usize,
    ) -> Self {
        let split_on = |dim| inputs.contains(&Split { dim });
        if split_on(1) && Self::heads_split_together(q_heads, kv_heads, num_members) {
            Self::Heads
        } else if split_on(0) {
            Self::Batch
        } else {
            Self::Unsplit
        }
    }

    /// Where the query, key and value, then the mask and bias the op has, must be, and where the
    /// output lands.
    pub fn sharding(self, inputs: &[&Shape]) -> OpSharding {
        let output = match self {
            Self::Unsplit => Replicated,
            Self::Batch => Split { dim: 0 },
            Self::Heads => Split { dim: 1 },
        };
        OpSharding {
            inputs: inputs
                .iter()
                .map(|shape| ElementwiseRule::input_sharding(output, shape, shape.num_dims()))
                .collect(),
            output,
        }
    }

    /// Each member's query heads read only its own chunk of the key and value heads when no group
    /// of query heads sharing one is cut: one query head per group, one key and value head kept
    /// whole by every member, or key and value heads that split evenly.
    fn heads_split_together(q_heads: usize, kv_heads: usize, num_members: usize) -> bool {
        q_heads == kv_heads || kv_heads == 1 || kv_heads.is_multiple_of(num_members)
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;

    use super::*;

    fn shapes(q_heads: usize, kv_heads: usize) -> AttentionShapes {
        let query = Shape::new([2, q_heads, 3, 4]);
        let kv = Shape::new([2, kv_heads, 5, 4]);
        AttentionShapes::new(&query, &kv, &kv)
    }

    #[test]
    fn heads_split_without_a_collective() {
        let rule = AttentionRule::new([Split { dim: 1 }; 3], &shapes(4, 4), 2);
        let scores = Shape::new([2, 4, 3, 5]);

        assert_eq!(
            rule.sharding(&[&scores; 4]),
            OpSharding {
                inputs: smallvec![Split { dim: 1 }; 4],
                output: Split { dim: 1 },
            }
        );
    }

    #[test]
    fn a_single_key_and_value_head_is_read_whole_by_every_member() {
        let shapes = shapes(4, 1);
        let rule = AttentionRule::new([Split { dim: 1 }, Replicated, Replicated], &shapes, 2);
        let (query, kv) = (Shape::new([2, 4, 3, 4]), Shape::new([2, 1, 5, 4]));

        assert_eq!(
            rule.sharding(&[&query, &kv, &kv]).inputs[..],
            [Split { dim: 1 }, Replicated, Replicated]
        );
    }

    #[test]
    fn key_and_value_heads_cut_unevenly_fall_back_to_gathering() {
        let rule = AttentionRule::new([Split { dim: 1 }; 3], &shapes(6, 3), 2);

        assert_eq!(rule, AttentionRule::Unsplit);
    }

    #[test]
    fn batch_split_without_a_collective() {
        let rule = AttentionRule::new([Split { dim: 0 }; 3], &shapes(4, 4), 2);
        let (qkv, mask) = (Shape::new([2, 4, 3, 4]), Shape::new([1, 4, 3, 5]));

        assert_eq!(
            rule.sharding(&[&qkv, &qkv, &qkv, &mask]),
            OpSharding {
                inputs: smallvec![
                    Split { dim: 0 },
                    Split { dim: 0 },
                    Split { dim: 0 },
                    Replicated
                ],
                output: Split { dim: 0 },
            }
        );
    }

    #[test]
    fn heads_that_cannot_split_together_leave_the_batch_split() {
        let rule = AttentionRule::new(
            [Split { dim: 1 }, Split { dim: 0 }, Replicated],
            &shapes(6, 3),
            2,
        );

        assert_eq!(rule, AttentionRule::Batch);
    }

    #[test]
    fn a_mask_broadcast_over_the_heads_stays_whole() {
        let rule = AttentionRule::new([Split { dim: 1 }; 3], &shapes(4, 4), 2);
        let (qkv, mask) = (Shape::new([2, 4, 3, 4]), Shape::new([2, 1, 3, 5]));

        assert_eq!(
            rule.sharding(&[&qkv, &qkv, &qkv, &mask]).inputs[3],
            Replicated
        );
    }
}
