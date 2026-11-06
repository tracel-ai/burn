use burn_backend::ops::attention::AttentionShapes;
use burn_std::Shape;

use crate::{
    ElementwiseRule, OpSharding,
    Sharding::{self, Replicated, ReplicatedSplit, Split},
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
    /// Fewer key and value heads than members: each is held by one of `groups` groups of
    /// contiguous members, every one attending its own query heads against it.
    SharedHeads { groups: usize },
}

impl AttentionRule {
    /// The shardings of the query, key and value: a split along the heads picks the heads when
    /// their groups split together, else a split along the batch picks the batch.
    pub fn new(inputs: [Sharding; 3], shapes: &AttentionShapes, num_members: usize) -> Self {
        let split_on = |dim| inputs.contains(&Split { dim });
        if let [
            Split { dim: 1 },
            kv @ ReplicatedSplit { dim: 1, groups },
            value,
        ] = inputs
            && value == kv
            && Self::heads_shared_by_groups(shapes, groups, num_members)
        {
            Self::SharedHeads { groups }
        } else if split_on(1) && Self::heads_split_together(shapes, num_members) {
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
            Self::Heads | Self::SharedHeads { .. } => Split { dim: 1 },
        };
        OpSharding {
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(input, shape)| match (self, input) {
                    (Self::SharedHeads { groups }, 1 | 2) => ReplicatedSplit { dim: 1, groups },
                    _ => ElementwiseRule::input_sharding(output, shape, shape.num_dims()),
                })
                .collect(),
            output,
        }
    }

    /// Each member's query heads read only its own chunk of the key and value heads when no group
    /// of query heads sharing one is cut: one query head per group, one key and value head kept
    /// whole by every member, or key and value heads that split evenly.
    fn heads_split_together(shapes: &AttentionShapes, num_members: usize) -> bool {
        shapes.groups() == 1 || shapes.kv_heads == 1 || shapes.kv_heads.is_multiple_of(num_members)
    }

    /// Each group's one key and value head is read by exactly the query heads its members hold:
    /// the query heads split evenly, so every member's run of them sits inside one head's.
    fn heads_shared_by_groups(shapes: &AttentionShapes, groups: usize, num_members: usize) -> bool {
        shapes.kv_heads == groups && shapes.q_heads.is_multiple_of(num_members)
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

    #[test]
    fn query_heads_split_over_four_read_the_key_and_value_head_their_pair_shares() {
        let kv = ReplicatedSplit { dim: 1, groups: 2 };
        let shapes = shapes(16, 2);
        let rule = AttentionRule::new([Split { dim: 1 }, kv, kv], &shapes, 4);
        let (query, kv_shape) = (Shape::new([2, 16, 3, 4]), Shape::new([2, 2, 5, 4]));

        assert_eq!(rule, AttentionRule::SharedHeads { groups: 2 });
        assert_eq!(
            rule.sharding(&[&query, &kv_shape, &kv_shape]),
            OpSharding {
                inputs: smallvec![Split { dim: 1 }, kv, kv],
                output: Split { dim: 1 },
            }
        );
    }

    /// Six query heads over four members cut the heads of one pair's key head across the pair's
    /// boundary, so a member would need both groups' key heads.
    #[test]
    fn query_heads_that_do_not_divide_the_members_are_not_shared_by_pairs() {
        let kv = ReplicatedSplit { dim: 1, groups: 2 };

        let rule = AttentionRule::new([Split { dim: 1 }, kv, kv], &shapes(6, 2), 4);

        assert_eq!(rule, AttentionRule::Unsplit);
    }

    /// A member's query heads would read only one of its group's two key heads, so the key and
    /// value split over every member instead.
    #[test]
    fn two_key_heads_per_group_split_over_every_member_instead() {
        let kv = ReplicatedSplit { dim: 1, groups: 2 };

        let rule = AttentionRule::new([Split { dim: 1 }, kv, kv], &shapes(16, 4), 4);

        assert_eq!(rule, AttentionRule::Heads);
    }
}
