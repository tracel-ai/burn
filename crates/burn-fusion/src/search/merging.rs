use super::Block;
use crate::NumOperations;

#[derive(Debug, PartialEq)]
pub enum MergeBlockResult<O> {
    Full(Block<O>),
    Partial {
        merged: Vec<Block<O>>,
        failed: Vec<Block<O>>,
    },
    Fail,
}

pub fn merge_blocks<O: NumOperations>(blocks: &[&Block<O>], sorted: bool) -> MergeBlockResult<O> {
    if blocks.is_empty() {
        return MergeBlockResult::Fail;
    }
    if blocks.len() == 1 {
        return MergeBlockResult::Full(blocks[0].clone());
    }

    if blocks.len() == 2 {
        let block0 = blocks[0];
        let block1 = blocks[1];

        return match merge_two(block0, block1) {
            Some(result) => MergeBlockResult::Full(result),
            None => MergeBlockResult::Fail,
        };
    }

    let step_size = blocks.len() / 2;
    let num_steps = f32::ceil(blocks.len() as f32 / step_size as f32) as usize;

    let mut merged_full = Vec::new();
    let mut merged_partial = Vec::new();
    let mut failed_result = Vec::new();

    for i in 0..num_steps {
        let start = i * step_size;
        let end = usize::min(start + step_size, blocks.len());

        match merge_blocks(&blocks[start..end], false) {
            MergeBlockResult::Full(block) => {
                merged_full.push(block);
            }
            MergeBlockResult::Partial {
                mut merged,
                mut failed,
            } => {
                merged_partial.append(&mut merged);
                failed_result.append(&mut failed);
            }
            MergeBlockResult::Fail => {
                for b in &blocks[start..end] {
                    failed_result.push((*b).clone());
                }
            }
        }
    }

    if merged_full.len() == 1 && failed_result.len() == 0 {
        MergeBlockResult::Full(merged_full.remove(0))
    } else {
        let result = post_process_partial(merged_full, merged_partial, failed_result);

        if !sorted {
            return result;
        }

        match result {
            MergeBlockResult::Full(block) => MergeBlockResult::Full(block),
            MergeBlockResult::Partial {
                mut merged,
                mut failed,
            } => {
                Block::sort(&mut merged);
                Block::sort(&mut failed);

                MergeBlockResult::Partial { merged, failed }
            }
            MergeBlockResult::Fail => MergeBlockResult::Fail,
        }
    }
}

fn post_process_partial<O: NumOperations>(
    mut merged_full: Vec<Block<O>>,
    mut merged_partial: Vec<Block<O>>,
    mut merged_failed: Vec<Block<O>>,
) -> MergeBlockResult<O> {
    // First let's try to merge partial graphs.
    if merged_partial.len() > 1 {
        match merge_accumulator(&merged_partial[0], &merged_partial[1..]) {
            MergeBlockResult::Full(block) => {
                merged_partial = vec![block];
            }
            MergeBlockResult::Partial { merged, mut failed } => {
                merged_partial = merged;
                merged_failed.append(&mut failed);
            }
            MergeBlockResult::Fail => {}
        }
    }

    // Then let's try to merge partial graphs with failed merges.
    if !merged_failed.is_empty() {
        merged_partial.append(&mut merged_failed);
        match merge_accumulator(&merged_partial[0], &merged_partial[1..]) {
            MergeBlockResult::Full(block) => {
                merged_partial = vec![block];
            }
            MergeBlockResult::Partial { merged, mut failed } => {
                merged_partial = merged;
                merged_failed.append(&mut failed);
            }
            MergeBlockResult::Fail => {}
        }
    }

    // Then let's try to merge full graphs.
    if merged_full.len() > 1 {
        match merge_accumulator(&merged_full[0], &merged_full[1..]) {
            MergeBlockResult::Full(block) => {
                merged_full = vec![block];
            }
            MergeBlockResult::Partial { merged, mut failed } => {
                merged_full = merged;
                merged_failed.append(&mut failed);
            }
            MergeBlockResult::Fail => {}
        }
    }

    // Then let's try to merge full graphs with failed graphs.
    if !merged_full.is_empty() {
        merged_full.append(&mut merged_failed);
        match merge_accumulator(&merged_full[0], &merged_full[1..]) {
            MergeBlockResult::Full(block) => {
                merged_full = vec![block];
            }
            MergeBlockResult::Partial { merged, mut failed } => {
                merged_full = merged;
                merged_failed.append(&mut failed);
            }
            MergeBlockResult::Fail => {}
        }
    }

    // Then let's try to merge full graphs with partial graphs.
    if !merged_full.is_empty() || !merged_partial.is_empty() {
        merged_full.append(&mut merged_partial);
        match merge_accumulator(&merged_full[0], &merged_full[1..]) {
            MergeBlockResult::Full(block) => {
                merged_full = vec![block];
            }
            MergeBlockResult::Partial { merged, mut failed } => {
                merged_full = merged;
                merged_failed.append(&mut failed);
            }
            MergeBlockResult::Fail => {
                // We do nothing.
            }
        }
    }

    if merged_full.is_empty() {
        MergeBlockResult::Fail
    } else if merged_failed.is_empty() {
        if merged_full.len() == 1 {
            MergeBlockResult::Full(merged_full.remove(0))
        } else {
            MergeBlockResult::Partial {
                merged: merged_full,
                failed: vec![],
            }
        }
    } else {
        MergeBlockResult::Partial {
            merged: merged_full,
            failed: merged_failed,
        }
    }
}

fn merge_accumulator<O: NumOperations>(
    base: &Block<O>,
    blocks: &[Block<O>],
) -> MergeBlockResult<O> {
    let mut base = base.clone();
    let mut merged_failed = Vec::<Block<O>>::new();
    let mut merged_success = false;

    for block in blocks {
        println!("Merging {base:?} with {block:?}");
        let mut base_current = base.clone();
        match base_current.merge(block) {
            super::GraphMergingResult::Fail => {
                merged_failed.push((*block).clone());
            }
            super::GraphMergingResult::Succeed => {
                merged_success = true;
                base = base_current;
            }
        }
    }

    if merged_success {
        if merged_failed.is_empty() {
            MergeBlockResult::Full(base)
        } else {
            MergeBlockResult::Partial {
                merged: vec![base],
                failed: merged_failed,
            }
        }
    } else {
        MergeBlockResult::Fail
    }
}

fn merge_two<O: NumOperations>(a: &Block<O>, b: &Block<O>) -> Option<Block<O>> {
    let mut base = a.clone();

    if let super::GraphMergingResult::Succeed = base.merge(b) {
        return Some(base);
    }

    let mut base = b.clone();

    match base.merge(a) {
        super::GraphMergingResult::Succeed => Some(base),
        super::GraphMergingResult::Fail => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use crate::stream::execution::tests::{TestOptimization, TestOptimizationBuilder};
    use crate::{
        OptimizationBuilder,
        stream::tests::{operation_1, operation_2, operation_3},
    };

    #[test]
    fn test_merge_blocks_no_block() {
        let actual = merge_blocks::<TestOptimization>(&[], true);

        assert_eq!(actual, MergeBlockResult::Fail);
    }

    #[test]
    fn test_merge_blocks_single() {
        let builders = builders();
        let block = Block::new(&builders);
        let actual = merge_blocks::<TestOptimization>(&[&block], true);

        assert_eq!(actual, MergeBlockResult::Full(block));
    }

    #[test]
    fn test_merge_blocks_two_blocks() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        block1.register(&operation_1(), false, 0);
        block1.register(&operation_1(), false, 1);
        block2.register(&operation_1(), false, 2);
        block2.register(&operation_1(), false, 3);

        let actual = merge_blocks::<TestOptimization>(&[&block1, &block2], true);

        let mut expected = Block::new(&builders);
        expected.register(&operation_1(), false, 0);
        expected.register(&operation_1(), false, 1);
        expected.register(&operation_1(), false, 2);
        expected.register(&operation_1(), false, 3);

        assert_eq!(actual, MergeBlockResult::Full(expected));
    }

    #[test]
    fn test_merge_blocks_three_blocks() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        let mut block3 = Block::new(&builders);
        block1.register(&operation_1(), false, 0);
        block2.register(&operation_1(), false, 1);
        block3.register(&operation_1(), false, 2);

        let actual = merge_blocks::<TestOptimization>(&[&block1, &block2, &block3], true);

        let mut expected = Block::new(&builders);
        expected.register(&operation_1(), false, 0);
        expected.register(&operation_1(), false, 1);
        expected.register(&operation_1(), false, 2);

        assert_eq!(actual, MergeBlockResult::Full(expected));
    }

    #[test]
    fn test_merge_blocks_three_blocks_partial() {
        let builders = builders();
        let mut block1 = Block::new(&builders);
        let mut block2 = Block::new(&builders);
        let mut block3 = Block::new(&builders);
        block1.register(&operation_1(), false, 0);
        block2.register(&operation_2(), false, 1);
        block3.register(&operation_1(), false, 2);

        let actual = merge_blocks::<TestOptimization>(&[&block1, &block2, &block3], true);

        let mut expected1 = Block::new(&builders);
        let mut expected2 = Block::new(&builders);
        expected1.register(&operation_1(), false, 0);
        expected1.register(&operation_1(), false, 2);
        expected2.register(&operation_2(), false, 1);

        assert_eq!(
            actual,
            MergeBlockResult::Partial {
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
        block1.register(&operation_1(), false, 0);
        block2.register(&operation_2(), false, 1);
        block3.register(&operation_1(), false, 2);
        block4.register(&operation_3(), false, 3);

        let actual = merge_blocks::<TestOptimization>(&[&block1, &block2, &block3, &block4], true);

        let mut expected1 = Block::new(&builders);
        let mut expected2 = Block::new(&builders);
        let mut failed = Block::new(&builders);
        expected1.register(&operation_1(), false, 0);
        expected1.register(&operation_1(), false, 2);
        expected2.register(&operation_2(), false, 1);
        failed.register(&operation_3(), false, 3);

        assert_eq!(
            actual,
            MergeBlockResult::Partial {
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
        block1.register(&operation_1(), false, 0);
        block2.register(&operation_2(), false, 1);
        block3.register(&operation_1(), false, 2);
        block4.register(&operation_3(), false, 3);
        block5.register(&operation_2(), false, 4);

        let actual =
            merge_blocks::<TestOptimization>(&[&block1, &block2, &block3, &block4, &block5], true);

        let mut expected1 = Block::new(&builders);
        let mut expected2 = Block::new(&builders);
        let mut failed = Block::new(&builders);
        expected1.register(&operation_1(), false, 0);
        expected1.register(&operation_1(), false, 2);
        expected2.register(&operation_2(), false, 1);
        expected2.register(&operation_2(), false, 4);
        failed.register(&operation_3(), false, 3);

        assert_eq!(
            actual,
            MergeBlockResult::Partial {
                merged: vec![expected1, expected2],
                failed: vec![failed]
            }
        );
    }

    fn builders() -> Vec<Box<dyn OptimizationBuilder<TestOptimization>>> {
        let builder_1 = TestOptimizationBuilder::new(0, vec![operation_1(); 10]);
        let builder_2 = TestOptimizationBuilder::new(1, vec![operation_2(); 10]);

        vec![Box::new(builder_1), Box::new(builder_2)]
    }
}
