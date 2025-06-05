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
/// find an optimization that can be executed with that constraint, we return an
/// [error](BlocksOptimizerError).
pub struct BlocksOptimizer<O> {
    blocks: Vec<Block<O>>,
    resolved: Vec<bool>,
    last_checked: usize,
}

/// When we can't find a proper optimization for the provided list of [blocks](Block).
pub enum BlocksOptimizerError {
    /// When there is too much fallback operations to execute in order to not have wholes in the
    /// execution stream.
    TooMuchFallbacksOperations,
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
    /// while minimizing fallbacks operations to avoid having wholes in the optimization stream.
    pub fn optimize(mut self) -> Result<BlockOptimization<O>, BlocksOptimizerError> {
        self = self.merging_pass();

        let mut strategies = Vec::with_capacity(self.blocks.len());
        let mut ordering = Vec::new();
        let mut blocks = Vec::new();
        core::mem::swap(&mut blocks, &mut self.blocks);

        for block in blocks {
            match self.optimize_block(block, &mut ordering)? {
                Some(strategy) => {
                    strategies.push(Box::new(strategy));
                }
                None => break,
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

        Ok(optimization)
    }

    /// Optimize a single block.
    fn optimize_block(
        &mut self,
        block: Block<O>,
        ordering: &mut Vec<usize>,
    ) -> Result<Option<ExecutionStrategy<O>>, BlocksOptimizerError> {
        let last_index = block.end_pos;
        let mut block_optimization = block.optimize();
        let opt_size = block_optimization.ordering.len();

        for pos in block_optimization.ordering.iter() {
            self.update_check(*pos);
        }

        if self.last_checked != ordering.len() + opt_size {
            if !ordering.is_empty() {
                // Don't include that block and need furthur exploring.
                return Ok(None);
            }

            let strategy = self.optimize_wholes(block_optimization, last_index, ordering)?;
            return Ok(Some(strategy));
        }

        ordering.append(&mut block_optimization.ordering);
        Ok(Some(block_optimization.strategy))
    }

    /// The provided optimization has wholes.
    fn optimize_wholes(
        &mut self,
        mut optimization: BlockOptimization<O>,
        last_index: usize,
        ordering: &mut Vec<usize>,
    ) -> Result<ExecutionStrategy<O>, BlocksOptimizerError> {
        let strategy = match optimization.strategy {
            ExecutionStrategy::Optimization(opt) => {
                let fallbacks = self.add_missing_ops(last_index);

                if fallbacks.is_empty() {
                    ordering.append(&mut optimization.ordering);
                    ExecutionStrategy::Optimization(opt)
                } else {
                    let mut optimization =
                        self.optimize_wholes_opt(opt, fallbacks, optimization.ordering)?;

                    ordering.append(&mut optimization.ordering);
                    optimization.strategy
                }
            }
            ExecutionStrategy::Operations(size) => {
                let fallbacks = self.add_missing_ops(last_index);
                ordering.append(&mut optimization.ordering);
                ordering.append(&mut fallbacks.clone());

                ExecutionStrategy::Operations(fallbacks.len() + size)
            }
            _ => unreachable!(),
        };

        Ok(strategy)
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

    /// Optimize the wholes when an optimization was found.
    fn optimize_wholes_opt(
        &self,
        opt: O,
        fallbacks: Vec<usize>,
        ordering: Vec<usize>,
    ) -> Result<BlockOptimization<O>, BlocksOptimizerError> {
        if fallbacks.len() <= 1 {
            Ok(self.optimize_wholes_with_fallbacks(opt, fallbacks, ordering))
        } else {
            Err(BlocksOptimizerError::TooMuchFallbacksOperations)
        }
    }

    /// We only execute operations alongside the optimization.
    fn optimize_wholes_with_fallbacks(
        &self,
        opt: O,
        fallbacks: Vec<usize>,
        mut ordering: Vec<usize>,
    ) -> BlockOptimization<O> {
        ordering.append(&mut fallbacks.clone());

        BlockOptimization::new(
            ExecutionStrategy::OptimizationWithFallbacks(opt, fallbacks),
            ordering,
        )
    }
}
