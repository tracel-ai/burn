use super::Block;
use crate::{NumOperations, search::graph::Reachability};

#[derive(Debug, PartialEq)]
/// The result of [merging](merge_blocks) [blocks](Block).
// `Full`/`Partial` carry owned blocks by design; the size gap between variants is expected.
#[allow(clippy::large_enum_variant)]
pub enum MergeBlocksResult<O> {
    /// All [blocks](Block) merged into one.
    Full(Block<O>),
    /// Some [blocks](Block) merged and some failed.
    Partial {
        merged: Vec<Block<O>>,
        failed: Vec<Block<O>>,
    },
    /// All [blocks](Block) failed to merge.
    Fail,
}

/// Merge multiple [block](Block) together.
///
/// The resulting [blocks](Block) might be sorted if the flag is true, otherwise the order isn't
/// guarantee. This is mostly useful for testing.
///
/// # Strategy
///
/// The merging strategy is in two steps:
///
/// 1. The first step is to recursively try to merge adjacent blocks. This has the advantage of
///    trying multiple blocks ordering, therefore trying multiple permutation of the blocks.
///    However, it has the downside of not trying to merge blocks that are further away in the list
///    of blocks. Since trying all combinations possible is exponential, therefore not possible, we
///    fallback on the second strategy.
/// 2. The second step is to reduce blocks by setting an accumulator block, then sequentially
///    trying to merge the remaining blocks. We try some permutations based on the result from
///    step1.
pub fn merge_blocks<O: NumOperations>(
    blocks: &[&Block<O>],
    sorted: bool,
    guard: &Reachability,
) -> MergeBlocksResult<O> {
    if blocks.is_empty() {
        return MergeBlocksResult::Fail;
    }

    if blocks.len() == 1 {
        return MergeBlocksResult::Full(blocks[0].clone());
    }

    if blocks.len() == 2 {
        let block0 = blocks[0];
        let block1 = blocks[1];

        return match merge_two(block0, block1, guard) {
            Some(result) => MergeBlocksResult::Full(result),
            None => MergeBlocksResult::Fail,
        };
    }

    let mut step1 = merge_blocks_step1(blocks, guard);

    if step1.full.len() == 1 && step1.failed.is_empty() && step1.partial.is_empty() {
        MergeBlocksResult::Full(step1.full.remove(0))
    } else if step1.partial.len() == 1 && step1.failed.is_empty() && step1.full.is_empty() {
        MergeBlocksResult::Full(step1.partial.remove(0))
    } else {
        let result = merge_blocks_step2(step1, guard);

        if !sorted {
            return result;
        }

        match result {
            MergeBlocksResult::Full(block) => MergeBlocksResult::Full(block),
            MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                Block::sort(&mut merged);
                Block::sort(&mut failed);

                MergeBlocksResult::Partial { merged, failed }
            }
            MergeBlocksResult::Fail => MergeBlocksResult::Fail,
        }
    }
}

struct MergeBlockStep1<O> {
    full: Vec<Block<O>>,
    partial: Vec<Block<O>>,
    failed: Vec<Block<O>>,
}

impl<O> Default for MergeBlockStep1<O> {
    fn default() -> Self {
        Self {
            full: Default::default(),
            partial: Default::default(),
            failed: Default::default(),
        }
    }
}

fn merge_blocks_step1<O: NumOperations>(
    blocks: &[&Block<O>],
    guard: &Reachability,
) -> MergeBlockStep1<O> {
    let step_size = blocks.len() / 2;
    let num_steps = f32::ceil(blocks.len() as f32 / step_size as f32) as usize;

    let mut result = MergeBlockStep1::default();

    for i in 0..num_steps {
        let start = i * step_size;
        let end = usize::min(start + step_size, blocks.len());

        match merge_blocks(&blocks[start..end], false, guard) {
            MergeBlocksResult::Full(block) => {
                result.full.push(block);
            }
            MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                result.partial.append(&mut merged);
                result.failed.append(&mut failed);
            }
            MergeBlocksResult::Fail => {
                for b in &blocks[start..end] {
                    result.failed.push((*b).clone());
                }
            }
        }
    }

    result
}

fn merge_blocks_step2<O: NumOperations>(
    mut step1: MergeBlockStep1<O>,
    guard: &Reachability,
) -> MergeBlocksResult<O> {
    // First let's try to merge partial graphs.
    if step1.partial.len() > 1 {
        match merge_accumulator(&step1.partial[0], &step1.partial[1..], guard) {
            MergeBlocksResult::Full(block) => {
                step1.partial = vec![block];
            }
            MergeBlocksResult::Partial { merged, mut failed } => {
                step1.partial = merged;
                step1.failed.append(&mut failed);
            }
            MergeBlocksResult::Fail => {}
        }
    }

    // Then let's try to merge partial graphs with failed merges.
    if !step1.failed.is_empty() {
        step1.partial.append(&mut step1.failed);
        match merge_accumulator(&step1.partial[0], &step1.partial[1..], guard) {
            MergeBlocksResult::Full(block) => {
                step1.partial = vec![block];
            }
            MergeBlocksResult::Partial { merged, mut failed } => {
                step1.partial = merged;
                step1.failed.append(&mut failed);
            }
            MergeBlocksResult::Fail => {}
        }
    }

    // Then let's try to merge full graphs.
    if step1.full.len() > 1 {
        match merge_accumulator(&step1.full[0], &step1.full[1..], guard) {
            MergeBlocksResult::Full(block) => {
                step1.full = vec![block];
            }
            MergeBlocksResult::Partial { merged, mut failed } => {
                step1.full = merged;
                step1.failed.append(&mut failed);
            }
            MergeBlocksResult::Fail => {}
        }
    }

    // Then let's try to merge full graphs with failed graphs.
    if !step1.full.is_empty() {
        step1.full.append(&mut step1.failed);
        match merge_accumulator(&step1.full[0], &step1.full[1..], guard) {
            MergeBlocksResult::Full(block) => {
                step1.full = vec![block];
            }
            MergeBlocksResult::Partial { merged, mut failed } => {
                step1.full = merged;
                step1.failed.append(&mut failed);
            }
            MergeBlocksResult::Fail => {}
        }
    }

    // Then let's try to merge full graphs with partial graphs.
    if !step1.full.is_empty() || !step1.partial.is_empty() {
        step1.full.append(&mut step1.partial);
        match merge_accumulator(&step1.full[0], &step1.full[1..], guard) {
            MergeBlocksResult::Full(block) => {
                step1.full = vec![block];
            }
            MergeBlocksResult::Partial { merged, mut failed } => {
                step1.full = merged;
                step1.failed.append(&mut failed);
            }
            MergeBlocksResult::Fail => {
                // We do nothing.
            }
        }
    }

    if step1.full.is_empty() {
        MergeBlocksResult::Fail
    } else if step1.failed.is_empty() {
        if step1.full.len() == 1 {
            MergeBlocksResult::Full(step1.full.remove(0))
        } else {
            MergeBlocksResult::Partial {
                merged: step1.full,
                failed: vec![],
            }
        }
    } else {
        MergeBlocksResult::Partial {
            merged: step1.full,
            failed: step1.failed,
        }
    }
}

fn merge_accumulator<O: NumOperations>(
    base: &Block<O>,
    blocks: &[Block<O>],
    guard: &Reachability,
) -> MergeBlocksResult<O> {
    let mut base = base.clone();
    let mut merged_failed = Vec::<Block<O>>::new();
    let mut merged_success = false;

    for block in blocks {
        // `merge_two` checks the cycle guard and tries both merge directions — the accumulator
        // may depend on the block, in which case only folding the accumulator *into* the block
        // yields a valid operation order.
        match merge_two(&base, block, guard) {
            Some(merged) => {
                merged_success = true;
                base = merged;
            }
            None => {
                merged_failed.push(block.clone());
            }
        }
    }

    if merged_success {
        if merged_failed.is_empty() {
            MergeBlocksResult::Full(base)
        } else {
            MergeBlocksResult::Partial {
                merged: vec![base],
                failed: merged_failed,
            }
        }
    } else {
        MergeBlocksResult::Fail
    }
}

fn merge_two<O: NumOperations>(
    a: &Block<O>,
    b: &Block<O>,
    guard: &Reachability,
) -> Option<Block<O>> {
    if !guard.can_contract(a.constituents(), b.constituents()) {
        return None;
    }

    // Test each direction's operation order before paying for a deep clone of the base block
    // (operations, builders, and data-flow sets all clone).
    if a.can_append(b) {
        let mut base = a.clone();
        if base.merge(b) {
            return Some(base);
        }
    }

    if b.can_append(a) {
        let mut base = b.clone();
        if base.merge(a) {
            return Some(base);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use crate::stream::execution::tests::{TestOptimization, TestOptimizationBuilder};
    use crate::{
        OperationFuser,
        search::graph::Dag,
        stream::tests::{operation_1, operation_2, operation_3},
    };

    /// Seed constituents by index, build the [Reachability] guard from the blocks, and merge
    /// (sorted).
    fn merge(mut blocks: Vec<Block<TestOptimization>>) -> MergeBlocksResult<TestOptimization> {
        for (i, block) in blocks.iter_mut().enumerate() {
            block.seed_constituent(i);
        }
        let refs = blocks.iter().collect::<Vec<_>>();
        let guard = Dag::new(&refs).reachability();
        merge_blocks(&refs, true, &guard)
    }

    #[test]
    fn test_merge_blocks_no_block() {
        let actual = merge(Vec::<Block<TestOptimization>>::new());

        assert_eq!(actual, MergeBlocksResult::Fail);
    }

    #[test]
    fn test_merge_blocks_single() {
        let builders = builders();
        let block = Block::new(&builders);
        let actual = merge(vec![block.clone()]);

        assert_eq!(actual, MergeBlocksResult::Full(block));
    }

    #[test]
    fn test_merge_blocks_two_blocks() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        block1.register(&operation_1(), 0, false);
        block1.register(&operation_1(), 1, false);
        block2.register(&operation_1(), 2, false);
        block2.register(&operation_1(), 3, false);

        let actual = merge(vec![block1, block2]);

        let mut expected = Block::new(&builders);
        expected.register(&operation_1(), 0, false);
        expected.register(&operation_1(), 1, false);
        expected.register(&operation_1(), 2, false);
        expected.register(&operation_1(), 3, false);

        assert_eq!(actual, MergeBlocksResult::Full(expected));
    }

    #[test]
    fn test_merge_blocks_three_blocks() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        let mut block3 = Block::new(&builders);
        block1.register(&operation_1(), 0, false);
        block2.register(&operation_1(), 1, false);
        block3.register(&operation_1(), 2, false);

        let actual = merge(vec![block1, block2, block3]);

        let mut expected = Block::new(&builders);
        expected.register(&operation_1(), 0, false);
        expected.register(&operation_1(), 1, false);
        expected.register(&operation_1(), 2, false);

        assert_eq!(actual, MergeBlocksResult::Full(expected));
    }

    #[test]
    fn test_merge_blocks_three_blocks_partial() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        let mut block3 = Block::new(&builders);
        block1.register(&operation_1(), 0, false);
        block2.register(&operation_2(), 1, false);
        block3.register(&operation_1(), 2, false);

        let actual = merge(vec![block1, block2, block3]);

        let mut expected1 = Block::new(&builders);
        let mut expected2 = Block::new(&builders);
        expected1.register(&operation_1(), 0, false);
        expected1.register(&operation_1(), 2, false);
        expected2.register(&operation_2(), 1, false);

        assert_eq!(
            actual,
            MergeBlocksResult::Partial {
                merged: vec![expected1, expected2],
                failed: vec![]
            }
        );
    }

    #[test]
    fn test_merge_blocks_four_blocks_partial_with_failure() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        let mut block3 = Block::new(&builders);
        let mut block4 = Block::new(&builders);
        block1.register(&operation_1(), 0, false);
        block2.register(&operation_2(), 1, false);
        block3.register(&operation_1(), 2, false);
        block4.register(&operation_3(), 3, false);

        let actual = merge(vec![block1, block2, block3, block4]);

        let mut expected1 = Block::new(&builders);
        let mut expected2 = Block::new(&builders);
        let mut failed = Block::new(&builders);
        expected1.register(&operation_1(), 0, false);
        expected1.register(&operation_1(), 2, false);
        expected2.register(&operation_2(), 1, false);
        failed.register(&operation_3(), 3, false);

        assert_eq!(
            actual,
            MergeBlocksResult::Partial {
                merged: vec![expected1],
                failed: vec![expected2, failed]
            }
        );
    }

    #[test]
    fn test_merge_blocks_five_blocks_partial_with_failure() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        let mut block3 = Block::new(&builders);
        let mut block4 = Block::new(&builders);
        let mut block5 = Block::new(&builders);
        block1.register(&operation_1(), 0, false);
        block2.register(&operation_2(), 1, false);
        block3.register(&operation_1(), 2, false);
        block4.register(&operation_3(), 3, false);
        block5.register(&operation_2(), 4, false);

        let actual = merge(vec![block1, block2, block3, block4, block5]);

        let mut expected1 = Block::new(&builders);
        let mut expected2 = Block::new(&builders);
        let mut failed = Block::new(&builders);
        expected1.register(&operation_1(), 0, false);
        expected1.register(&operation_1(), 2, false);
        expected2.register(&operation_2(), 1, false);
        expected2.register(&operation_2(), 4, false);
        failed.register(&operation_3(), 3, false);

        assert_eq!(
            actual,
            MergeBlocksResult::Partial {
                merged: vec![expected1, expected2],
                failed: vec![failed]
            }
        );
    }

    /// The accumulator may depend on a block it tries to absorb: folding that block's ops at the
    /// end would fuse a consumer before its producer. The merge must then happen in the other
    /// direction (the accumulator folded into the block), preserving a full merge instead of
    /// degrading to a partial one.
    #[test]
    fn test_merge_blocks_accumulator_reverses_direction() {
        use crate::search::testing::add;

        let x = add(1, 2, 10);
        let y = add(50, 3, 11); // Reads 50...
        let z = add(4, 5, 50); // ...which this op produces.
        let w = add(6, 7, 12); // Independent.

        // The builder accepts exactly the reversed merge order [z, x, y, w].
        let pattern = vec![z.clone(), x.clone(), y.clone(), w.clone(), x.clone()];
        let builders: Vec<Box<dyn OperationFuser<TestOptimization>>> =
            vec![Box::new(TestOptimizationBuilder::new(0, pattern))];

        let mut block1 = Block::new(&builders);
        block1.register(&x, 0, true);
        block1.register(&y, 5, true);
        let mut block2 = Block::new(&builders);
        block2.register(&z, 3, true);
        let mut block3 = Block::new(&builders);
        block3.register(&w, 6, true);

        let actual = merge(vec![block1, block2, block3]);

        let mut expected = Block::new(&builders);
        expected.register(&z, 3, true);
        expected.register(&x, 0, true);
        expected.register(&y, 5, true);
        expected.register(&w, 6, true);

        assert_eq!(actual, MergeBlocksResult::Full(expected));
    }

    fn builders() -> Vec<Box<dyn OperationFuser<TestOptimization>>> {
        let builder_1 = TestOptimizationBuilder::new(0, vec![operation_1(); 10]);
        let builder_2 = TestOptimizationBuilder::new(1, vec![operation_2(); 10]);

        vec![Box::new(builder_1), Box::new(builder_2)]
    }
}
