use burn_std::config::{fusion::FusionLogLevel, log_fusion};

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
    num_ops: usize,
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

impl<O: NumOperations> BlocksOptimizer<O> {
    /// Create a new optimizer with the given blocks.
    pub fn new(blocks: Vec<Block<O>>) -> Self {
        let num_ops: usize = blocks.iter().map(|g| g.end_pos).max().unwrap();

        Self { blocks, num_ops }
    }

    /// Optimizes the blocks.
    ///
    /// Strategy:
    /// 1. Try to merge blocks together — independent blocks have no data
    ///    dependency, so reordering across a merge is safe.
    /// 2. Ask every resulting block for its best optimization (or the
    ///    fallback [Operations](ExecutionStrategy::Operations) if no builder
    ///    matched) and concatenate the strategies.
    /// 3. An *interior* unresolved position — an op that no block's final
    ///    ordering covered, but which sits before some other resolved
    ///    position — is a hole that must be filled by a second optimization
    ///    pass. Trailing unresolved positions, on the other hand, are the
    ///    natural tail of a drained block and are left in the queue for the
    ///    processor to handle in the next round.
    pub fn optimize(mut self) -> BlocksOptimizerResult<O> {
        self = self.merging_pass();

        let num_ops = self.num_ops;
        let blocks = core::mem::take(&mut self.blocks);

        let mut strategies: Vec<Box<ExecutionStrategy<O>>> = Vec::with_capacity(blocks.len());
        let mut ordering = Vec::new();
        let mut resolved = vec![false; num_ops];

        for block in blocks {
            let mut block_opt = block.optimize();
            for pos in block_opt.ordering.iter() {
                resolved[*pos] = true;
            }
            ordering.append(&mut block_opt.ordering);
            strategies.push(Box::new(block_opt.strategy));
        }

        // An unresolved position is a hole only if some position *after* it
        // is resolved (it's interleaved, not trailing). A trailing run of
        // unresolved positions is a drained tail — left for the processor.
        let last_resolved_end = resolved
            .iter()
            .rposition(|&r| r)
            .map(|i| i + 1)
            .unwrap_or(0);
        let holes: Vec<usize> = (0..last_resolved_end).filter(|i| !resolved[*i]).collect();

        let num_strategies = strategies.len();
        log_fusion(FusionLogLevel::Basic, move || {
            if num_strategies > 1 {
                format!("selected composed strategy ({num_strategies} sub-strategies)")
            } else {
                "selected single strategy".to_string()
            }
        });

        if holes.is_empty() {
            let strategy = if strategies.len() > 1 {
                ExecutionStrategy::Composed(strategies)
            } else {
                *strategies.remove(0)
            };
            BlocksOptimizerResult::Full(BlockOptimization::new(strategy, ordering))
        } else {
            BlocksOptimizerResult::WithHoles {
                strategies,
                ordering,
                holes,
            }
        }
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
