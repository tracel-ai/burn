use crate::{NumOperations, stream::store::ExecutionStrategy};

use super::{Block, OptimizationSearchResult, merging::merge_blocks};

/// What we know here is that every block is independent at that time and can be executed
/// in any order.
///
/// The contract is that the length of operations executed must include all operations.
pub struct Searcher<O> {
    blocks: Vec<Block<O>>,
    resolved: Vec<bool>,
    last_checked: usize,
}

impl<O: NumOperations> Searcher<O> {
    pub fn new(blocks: Vec<Block<O>>) -> Self {
        let num_ops: usize = blocks.iter().map(|g| g.end_pos).max().unwrap();

        Self {
            blocks,
            resolved: vec![false; num_ops],
            last_checked: 0,
        }
    }

    pub fn search(mut self) -> OptimizationSearchResult<O> {
        self = self.merging_pass();

        let mut strategies = Vec::with_capacity(self.blocks.len());
        let mut blocks = Vec::new();
        let mut num_optimized = 0;

        core::mem::swap(&mut blocks, &mut self.blocks);

        for block in blocks {
            let last_index = block.end_pos;
            let (strategy, positions) = block.compile();
            let opt_size = positions.len();

            for pos in positions {
                self.update_check(pos);
            }

            if self.last_checked != num_optimized + opt_size {
                if num_optimized > 0 {
                    // Don't include that block and need furthur exploring.
                    break;
                } else {
                    num_optimized += opt_size;

                    match strategy {
                        ExecutionStrategy::Optimization { opt, mut positions } => {
                            let fallbacks = self.add_missing_ops(last_index);
                            num_optimized += fallbacks.len();
                            positions.append(&mut fallbacks.clone());
                            let strategy = ExecutionStrategy::OptimizationWithFallbacks {
                                opt,
                                fallbacks,
                                positions,
                            };

                            strategies.push(Box::new(strategy));
                            break;
                        }
                        ExecutionStrategy::Operations(size) => {
                            let fallbacks = self.add_missing_ops(last_index);
                            num_optimized += fallbacks.len();

                            let strategy = ExecutionStrategy::Operations(fallbacks.len() + size);
                            strategies.push(Box::new(strategy));
                            break;
                        }
                        _ => unreachable!(),
                    };
                }
            } else {
                num_optimized += opt_size;
            }

            strategies.push(Box::new(strategy));
        }

        if strategies.len() > 1 {
            OptimizationSearchResult {
                strategy: ExecutionStrategy::Composed(strategies),
                num_operations: num_optimized,
            }
        } else {
            OptimizationSearchResult {
                strategy: *strategies.remove(0),
                num_operations: num_optimized,
            }
        }
    }

    fn update_check(&mut self, pos: usize) {
        self.resolved[pos] = true;

        for i in self.last_checked..self.resolved.len() {
            if self.resolved[i] {
                self.last_checked += 1;
            } else {
                break;
            }
        }
    }

    fn add_missing_ops(&self, last: usize) -> Vec<usize> {
        let mut fallbacks = Vec::new();

        for i in self.last_checked..last {
            if !self.resolved[i] {
                fallbacks.push(i);
            }
        }

        fallbacks
    }

    fn merging_pass(mut self) -> Self {
        if self.blocks.len() == 1 {
            return self;
        }

        Block::sort(&mut self.blocks);
        let blocks = self.blocks.iter().collect::<Vec<_>>();

        match merge_blocks(&blocks, false) {
            super::merging::MergeBlockResult::Full(block) => {
                self.blocks = vec![block];
            }
            super::merging::MergeBlockResult::Partial {
                mut merged,
                mut failed,
            } => {
                merged.append(&mut failed);
                self.blocks = merged;
                Block::sort(&mut self.blocks);
            }
            super::merging::MergeBlockResult::Fail => {}
        }

        self
    }
}
