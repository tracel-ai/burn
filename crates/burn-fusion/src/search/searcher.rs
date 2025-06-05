use crate::{NumOperations, stream::store::ExecutionStrategy};

use super::{Block, OptimizationFound, merging::merge_blocks};

/// What we know here is that every block is independent at that time and can be executed
/// in any order.
///
/// The contract is that the length of operations executed must include all operations.
pub struct Searcher<O> {
    blocks: Vec<Block<O>>,
    resolved: Vec<bool>,
    last_checked: usize,
}

pub enum SearchError {
    TooMuchFallbacksOperations,
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

    pub fn search(mut self) -> Result<OptimizationFound<O>, SearchError> {
        self = self.merging_pass();

        let mut strategies = Vec::with_capacity(self.blocks.len());
        let mut blocks = Vec::new();
        let mut num_optimized = 0;
        let mut ordering = Vec::new();

        core::mem::swap(&mut blocks, &mut self.blocks);

        for block in blocks {
            let last_index = block.end_pos;
            let mut block_optimization = block.optimize();
            let opt_size = block_optimization.ordering.len();

            for pos in block_optimization.ordering.iter() {
                self.update_check(*pos);
            }

            if self.last_checked != num_optimized + opt_size {
                if num_optimized > 0 {
                    // Don't include that block and need furthur exploring.
                    break;
                } else {
                    match block_optimization.strategy {
                        ExecutionStrategy::Optimization(opt) => {
                            let fallbacks = self.add_missing_ops(last_index);

                            let strategy = if fallbacks.is_empty() {
                                num_optimized += opt_size;
                                ordering.append(&mut block_optimization.ordering);
                                ExecutionStrategy::Optimization(opt)
                            } else {
                                let (strategy, mut positions, opt_len) = self.optimize_fallback(
                                    opt,
                                    fallbacks,
                                    block_optimization.ordering,
                                )?;

                                num_optimized += opt_len;
                                ordering.append(&mut positions);
                                strategy
                            };
                            // assert_eq!(self.last_checked, num_optimized, "Num optimized");

                            strategies.push(Box::new(strategy));
                            break;
                        }
                        ExecutionStrategy::Operations(size) => {
                            let fallbacks = self.add_missing_ops(last_index);
                            ordering.append(&mut block_optimization.ordering);
                            ordering.append(&mut fallbacks.clone());
                            num_optimized += size;
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
                ordering.append(&mut block_optimization.ordering);
            }

            strategies.push(Box::new(block_optimization.strategy));
        }

        Ok(if strategies.len() > 1 {
            OptimizationFound {
                strategy: ExecutionStrategy::Composed(strategies),
                num_operations: num_optimized,
                ordering,
            }
        } else {
            OptimizationFound {
                strategy: *strategies.remove(0),
                num_operations: num_optimized,
                ordering,
            }
        })
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

    fn add_missing_ops(&mut self, last: usize) -> Vec<usize> {
        let mut fallbacks = Vec::new();

        for i in self.last_checked..last {
            if !self.resolved[i] {
                fallbacks.push(i);
                self.resolved[i] = true;
            }
            self.last_checked += 1;
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
            super::merging::MergeBlocksResult::Full(block) => {
                self.blocks = vec![block];
            }
            super::merging::MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                merged.append(&mut failed);
                self.blocks = merged;
                Block::sort(&mut self.blocks);
            }
            super::merging::MergeBlocksResult::Fail => {}
        }

        self
    }

    pub fn optimize_fallback(
        &self,
        opt: O,
        fallbacks: Vec<usize>,
        positions: Vec<usize>,
    ) -> Result<(ExecutionStrategy<O>, Vec<usize>, usize), SearchError> {
        if fallbacks.len() <= 1 {
            Ok(self.optimize_fallback_no_change(opt, fallbacks, positions))
        } else {
            Err(SearchError::TooMuchFallbacksOperations)
        }
    }

    fn optimize_fallback_no_change(
        &self,
        opt: O,
        fallbacks: Vec<usize>,
        mut positions: Vec<usize>,
    ) -> (ExecutionStrategy<O>, Vec<usize>, usize) {
        let mut num_optimized = opt.len();
        num_optimized += fallbacks.len();

        positions.append(&mut fallbacks.clone());

        (
            ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks),
            positions,
            num_optimized,
        )
    }
}
