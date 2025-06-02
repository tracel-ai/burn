use super::Block;
use crate::NumOperations;

pub enum MergeBlockResult<O> {
    Full(Block<O>),
    Partial {
        merged: Vec<Block<O>>,
        failed: Vec<Block<O>>,
    },
    Fail,
}

pub fn merge_blocks<O: NumOperations>(blocks: &[&Block<O>]) -> MergeBlockResult<O> {
    if blocks.len() == 2 {
        let block0 = blocks[0];
        let block1 = blocks[1];

        return match merge_two(block0, block1) {
            Some(result) => MergeBlockResult::Full(result),
            None => MergeBlockResult::Fail,
        };
    }

    let step_size = blocks.len() / 2;
    let num_steps = blocks.len() / step_size;

    let mut merged_full = Vec::new();
    let mut merged_partial = Vec::new();
    let mut failed_result = Vec::new();

    for i in 0..num_steps {
        let start = i * step_size;
        let end = usize::min(start + step_size, blocks.len());

        match merge_blocks(&blocks[start..end]) {
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
    } else if merged_full.is_empty() {
        MergeBlockResult::Fail
    } else {
        post_process_partial(merged_full, merged_partial, failed_result)
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
            MergeBlockResult::Fail => todo!(),
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
    if !merged_full.is_empty() && !merged_partial.is_empty() {
        merged_full.append(&mut merged_partial);
        match merge_accumulator(&merged_full[0], &merged_full[1..]) {
            MergeBlockResult::Full(block) => {
                merged_full = vec![block];
            }
            MergeBlockResult::Partial { merged, mut failed } => {
                merged_full = merged;
                merged_failed.append(&mut failed);
            }
            MergeBlockResult::Fail => todo!(),
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
