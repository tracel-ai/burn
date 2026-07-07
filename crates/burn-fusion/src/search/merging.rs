use super::{Block, MergeGuard};
use crate::NumOperations;

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
    guard: &MergeGuard,
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
    guard: &MergeGuard,
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
    guard: &MergeGuard,
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
    guard: &MergeGuard,
) -> MergeBlocksResult<O> {
    let mut base = base.clone();
    let mut merged_failed = Vec::<Block<O>>::new();
    let mut merged_success = false;

    for block in blocks {
        // Skip a contraction that would create a dependency cycle. `base.constituents` grows as
        // it absorbs blocks, so this correctly accounts for everything already merged in.
        if !guard.can_merge(base.constituents(), block.constituents()) {
            merged_failed.push((*block).clone());
            continue;
        }

        let mut base_current = base.clone();
        match base_current.merge(block) {
            false => {
                merged_failed.push((*block).clone());
            }
            true => {
                merged_success = true;
                base = base_current;
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
    guard: &MergeGuard,
) -> Option<Block<O>> {
    if !guard.can_merge(a.constituents(), b.constituents()) {
        return None;
    }

    let mut base = a.clone();

    if base.merge(b) {
        return Some(base);
    }

    let mut base = b.clone();

    match base.merge(a) {
        true => Some(base),
        false => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use crate::stream::execution::tests::{TestOptimization, TestOptimizationBuilder};
    use crate::{
        OperationFuser,
        stream::tests::{operation_1, operation_2, operation_3},
    };

    /// Seed constituents by index, build the [MergeGuard] from the blocks, and merge (sorted).
    fn merge(mut blocks: Vec<Block<TestOptimization>>) -> MergeBlocksResult<TestOptimization> {
        for (i, block) in blocks.iter_mut().enumerate() {
            block.seed_constituent(i);
        }
        let refs = blocks.iter().collect::<Vec<_>>();
        let guard = MergeGuard::new(&refs);
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

    fn builders() -> Vec<Box<dyn OperationFuser<TestOptimization>>> {
        let builder_1 = TestOptimizationBuilder::new(0, vec![operation_1(); 10]);
        let builder_2 = TestOptimizationBuilder::new(1, vec![operation_2(); 10]);

        vec![Box::new(builder_1), Box::new(builder_2)]
    }
}
