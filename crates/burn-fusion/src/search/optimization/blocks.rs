use burn_std::config::{fusion::FusionLogLevel, log_fusion};

use crate::{
    NumOperations,
    search::{
        Block, BlockOptimization, MergeGuard, topological_order,
        merging::{MergeBlocksResult, merge_blocks},
    },
    stream::store::ExecutionStrategy,
};

/// Try to optimize a list of [blocks](Block) into a [block optimization](BlockOptimization).
///
/// # Notes
///
/// The blocks form a dependency DAG: a block may depend on another when it consumes a tensor the
/// other produces. Blocks with no dependency between them can still be merged/reordered freely;
/// dependent blocks must be emitted in a topological order (dependencies first).
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
    /// 1. Try to merge blocks together — the merge guard rejects any contraction that would
    ///    create a dependency cycle, so a surviving merge is always order-safe.
    /// 2. Emit the blocks in a topological order (dependencies first) and ask every block for its
    ///    best optimization (or the fallback [Operations](ExecutionStrategy::Operations) if no
    ///    builder matched), concatenating the strategies.
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

        // Emit blocks in a valid execution order: a dependency must run before its dependents.
        // The set is acyclic (register keeps it so, and `merging_pass` only accepts cycle-free
        // merges), so `topological_order` always succeeds; fall back to the current order if not.
        let order =
            topological_order(&blocks).unwrap_or_else(|| (0..blocks.len()).collect::<Vec<_>>());
        let mut slots: Vec<Option<Block<O>>> = blocks.into_iter().map(Some).collect();
        let blocks: Vec<Block<O>> = order
            .into_iter()
            .map(|i| slots[i].take().expect("each block taken once"))
            .collect();

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

        // Seed constituents so the guard can reason about the original dependency DAG through the
        // recursive merging, then reject any contraction that would create a cycle.
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.seed_constituent(i);
        }
        let blocks = self.blocks.iter().collect::<Vec<_>>();
        let guard = MergeGuard::new(&blocks);

        match merge_blocks(&blocks, false, &guard) {
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
