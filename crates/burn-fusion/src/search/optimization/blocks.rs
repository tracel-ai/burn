use std::sync::Arc;

use crate::{
    NumOperations,
    search::{
        Block, BlockOptimization,
        merging::{MergeBlocksResult, merge_blocks},
    },
    stream::store::ExecutionStrategy,
};

/// Try to optimize a list of [blocks](Block) into a [block optimization](BlockOptimization).
///
/// # Notes
///
/// What we know here is that every block is independent at that time and can be executed
/// in any order.
///
/// The contract is that the length of operations executed must include all operations. If we don't
/// find an optimization that can be executed with that constraint, we return a
/// [BlocksOptimizerResult::WithHoles].
pub struct BlocksOptimizer<O> {
    blocks: Vec<Block<O>>,
    resolved: Vec<bool>,
    last_checked: usize,
}

/// When we can't find a proper optimization for the provided list of [blocks](Block).
pub enum BlocksOptimizerResult<O> {
    /// When an optimization fill the hole stream.
    Full(BlockOptimization<O>),
    /// The optimization found with the holes indices.
    WithHoles {
        strategies: Vec<Box<ExecutionStrategy<O>>>,
        ordering: Vec<usize>,
        holes: Vec<usize>,
    },
}

enum BlockOptimizationStep<O> {
    Contiguous {
        strategy: ExecutionStrategy<O>,
    },
    /// Only happen when we fallback on executing a single operation.
    Operation {
        strategy: ExecutionStrategy<O>,
    },
    WithHoles {
        strategy: ExecutionStrategy<O>,
        holes: Vec<usize>,
    },
    Stop,
}

impl<O: NumOperations> BlocksOptimizer<O> {
    /// Create a new optimizer with the given blocks.
    pub fn new(blocks: Vec<Block<O>>) -> Self {
        let num_ops: usize = blocks.iter().map(|g| g.end_pos).max().unwrap();

        Self {
            blocks,
            resolved: vec![false; num_ops],
            last_checked: 0,
        }
    }

    /// Optimizes the blocks.
    ///
    /// The strategy is quite simple. We try to merge as much [blocks](Block) together as we can,
    /// then we iterate over them in order composing optimizations with the remaining blocks, all
    /// while minimizing fallbacks operations to avoid having holes in the optimization stream.
    pub fn optimize(mut self) -> BlocksOptimizerResult<O> {
        self = self.merging_pass();

        let mut strategies = Vec::with_capacity(self.blocks.len());
        let mut ordering = Vec::new();
        let mut blocks = Vec::new();
        core::mem::swap(&mut blocks, &mut self.blocks);

        for block in blocks {
            match self.optimize_block(block, &mut ordering) {
                BlockOptimizationStep::Contiguous { strategy } => {
                    strategies.push(Box::new(strategy));
                }
                BlockOptimizationStep::Operation { strategy } => {
                    strategies.push(Box::new(strategy));
                    break;
                }
                BlockOptimizationStep::WithHoles { strategy, holes } => {
                    strategies.push(Box::new(strategy));

                    return BlocksOptimizerResult::WithHoles {
                        strategies,
                        ordering,
                        holes,
                    };
                }
                BlockOptimizationStep::Stop => {
                    break;
                }
            }
        }

        let optimization = match strategies.len() > 1 {
            true => BlockOptimization {
                strategy: ExecutionStrategy::Composed(strategies),
                ordering,
            },
            false => BlockOptimization {
                strategy: *strategies.remove(0),
                ordering,
            },
        };

        BlocksOptimizerResult::Full(optimization)
    }

    /// Optimize a single block.
    fn optimize_block(
        &mut self,
        block: Block<O>,
        ordering: &mut Vec<usize>,
    ) -> BlockOptimizationStep<O> {
        let last_index = block.end_pos;
        let mut block_optimization = block.optimize();
        let opt_size = block_optimization.ordering.len();

        for pos in block_optimization.ordering.iter() {
            self.update_check(*pos);
        }

        if self.last_checked != ordering.len() + opt_size {
            if !ordering.is_empty() {
                // Don't include that block and need further exploring.
                return BlockOptimizationStep::Stop;
            }

            return self.optimize_holes(block_optimization, last_index, ordering);
        }

        ordering.append(&mut block_optimization.ordering);
        BlockOptimizationStep::Contiguous {
            strategy: block_optimization.strategy,
        }
    }

    /// The provided optimization has holes.
    fn optimize_holes(
        &mut self,
        mut optimization: BlockOptimization<O>,
        last_index: usize,
        ordering_global: &mut Vec<usize>,
    ) -> BlockOptimizationStep<O> {
        match optimization.strategy {
            ExecutionStrategy::Optimization { opt, ordering } => {
                ordering_global.append(&mut optimization.ordering);
                let holes = self.find_holes(last_index);

                if holes.is_empty() {
                    let strategy = ExecutionStrategy::Optimization { opt, ordering };
                    BlockOptimizationStep::Contiguous { strategy }
                } else {
                    let strategy = ExecutionStrategy::Optimization { opt, ordering };
                    BlockOptimizationStep::WithHoles { strategy, holes }
                }
            }
            ExecutionStrategy::Operations { ordering } => {
                let min = ordering.iter().min().unwrap();
                ordering_global.push(*min);

                let strategy = ExecutionStrategy::Operations {
                    ordering: Arc::new(vec![*min]),
                };
                BlockOptimizationStep::Operation { strategy }
            }
            _ => unreachable!(),
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

    fn find_holes(&mut self, last: usize) -> Vec<usize> {
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

    /// Try to merge blocks together.
    fn merging_pass(mut self) -> Self {
        if self.blocks.len() == 1 {
            return self;
        }

        Block::sort(&mut self.blocks);
        let blocks = self.blocks.iter().collect::<Vec<_>>();

        match merge_blocks(&blocks, false) {
            MergeBlocksResult::Full(block) => {
                self.blocks = vec![block];
            }
            MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                merged.append(&mut failed);
                self.blocks = merged;
                Block::sort(&mut self.blocks);
            }
            MergeBlocksResult::Fail => {}
        }

        self
    }
}
