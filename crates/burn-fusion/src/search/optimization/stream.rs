use super::blocks::BlocksOptimizer;
use crate::{
    NumOperations, OperationFuser,
    search::{
        Block, BlockOptimization, RegistrationResult,
        merging::{MergeBlocksResult, merge_blocks},
        optimization::blocks::BlocksOptimizerResult,
    },
    stream::{execution::op_kind, store::ExecutionStrategy},
};
use burn_ir::OperationIr;
use burn_std::config::{config, fusion::FusionLogLevel, log_fusion};

/// Optimize a stream of [operations](OperationIr) using a list of [builders](OptimizationBuilder).
pub struct StreamOptimizer<O> {
    builders: Vec<Box<dyn OperationFuser<O>>>,
    blocks: Vec<Block<O>>,
    length: usize,
    stopped: bool,
    max_blocks: Option<usize>,
}

impl<O: NumOperations> StreamOptimizer<O> {
    /// Create a new stream optimizer.
    pub fn new(builders: Vec<Box<dyn OperationFuser<O>>>) -> Self {
        // Too high and it may break the fusion cache always retriggering explorations.
        let max_blocks = Some(config().fusion().beam_search.max_blocks);
        Self {
            builders,
            blocks: Vec::new(),
            length: 0,
            stopped: false,
            max_blocks,
        }
    }

    /// Register a new [operation](OperationIr) in the optimizer.
    ///
    /// You can use the function [Self::still_optimizing] to know if the operations are actually
    /// being registered.
    pub fn register(&mut self, operation: &OperationIr) {
        if self.stopped {
            let length = self.length;
            log_fusion(FusionLogLevel::Full, || {
                format!(
                    "[stream] {} dropped (optimizer stopped at op {length})",
                    op_kind(operation)
                )
            });
            return;
        }

        if self.blocks.is_empty() {
            self.on_new_block(operation);
            self.length += 1;
            return;
        }

        match self.merge_blocks(operation, false) {
            MergeBlockStep::Full | MergeBlockStep::NoNeed => {}
            step @ (MergeBlockStep::Fail | MergeBlockStep::Partial) => {
                // With the given operation, blocks are no longer independent.
                let reason = match step {
                    MergeBlockStep::Fail => "merge failed",
                    MergeBlockStep::Partial => "merge partial",
                    _ => unreachable!(),
                };
                let num_blocks = self.blocks.len();
                let length = self.length;
                log_fusion(FusionLogLevel::Medium, || {
                    format!(
                        "[stream] stopped ({reason}) at op {length} ({}); {num_blocks} blocks",
                        op_kind(operation)
                    )
                });
                // A block that couldn't be merged with a sibling can still fuse with this very
                // operation — just in the next round. Defer such creation blocks instead of
                // flushing them here (see [`Self::defer_fusable_creation_blocks`]).
                self.defer_fusable_creation_blocks(operation);
                self.stopped = true;
                return;
            }
        }

        if let Some(max_blocks) = self.max_blocks {
            if self.register_max_block(operation, max_blocks) {
                self.length += 1;
            } else {
                let length = self.length;
                log_fusion(FusionLogLevel::Medium, || {
                    format!(
                        "[stream] stopped (max_blocks={max_blocks} reached) at op {length} ({})",
                        op_kind(operation)
                    )
                });
                self.stopped = true;
            }
            return;
        }

        let added_count = self.register_inner(operation, false);
        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            self.log_accepted(operation, added_count);
        }

        self.length += 1;
    }

    /// Optimize the current stream on the given [operations](OperationIr).
    ///
    /// # Notes
    ///
    /// The operations provided are the same as the ones used in the [register](Self::register)
    /// method, this simply remove the need for the current type to also keep track of the list of
    /// operations.
    pub fn optimize(&self, operations: &[OperationIr]) -> BlockOptimization<O> {
        let result = BlocksOptimizer::new(self.blocks.clone()).optimize();

        match result {
            BlocksOptimizerResult::Full(block_optimization) => block_optimization,
            BlocksOptimizerResult::WithHoles {
                mut strategies,
                mut ordering,
                mut holes,
            } => {
                loop {
                    let mut search = self.new_empty_search();

                    let mut operations_holes = Vec::with_capacity(holes.len());

                    for index in holes.iter() {
                        let op = &operations[*index];
                        operations_holes.push(op.clone());
                        search.register(op);
                    }

                    let mut optimization_of_holes = search.optimize(&operations_holes);

                    optimization_of_holes.map_ordering(&holes);

                    strategies.push(Box::new(optimization_of_holes.strategy));
                    holes.drain(0..optimization_of_holes.ordering.len());
                    ordering.append(&mut optimization_of_holes.ordering);

                    if holes.is_empty() {
                        break;
                    }
                }

                BlockOptimization::new(ExecutionStrategy::Composed(strategies), ordering)
            }
        }
    }

    /// Reset the state of the optimizer.
    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.blocks.clear();
        self.stopped = false;
    }

    /// Returns if some optimizations are still possible within the stream.
    pub fn still_optimizing(&self) -> bool {
        if self.stopped {
            return false;
        }
        if self.blocks.is_empty() {
            return true;
        }

        let mut num_stopped = 0;

        for block in self.blocks.iter() {
            if !block.still_optimizing() {
                num_stopped += 1
            }
        }

        num_stopped < self.blocks.len()
    }

    fn register_max_block(&mut self, operation: &OperationIr, max_blocks: usize) -> bool {
        if max_blocks == 1 {
            // With a single block every op is force-fused into it. A no-input creation op
            // (`zeros`/`empty`/`full`/…) that can't fuse into the current block would close it and
            // be flushed *with* the block — trapping its buffer so the consumer that arrives next
            // round reads it as a global input instead of fusing on-write (e.g. RoPE's `zeros`
            // wedged between the `neg` and its `slice_assign`). Defer it instead: it re-enters the
            // next round, starts a fresh block, and fuses with its consumer. Creation ops have no
            // inputs, so nothing already in the block depends on it and re-emitting it is free.
            if let Some(block) = self.blocks.first()
                && operation.inputs().next().is_none()
                && !block.would_fuse(operation, self.length)
            {
                return false;
            }
            // Register in the single block with a force.
            self.register_inner(operation, true);
            return true;
        }
        let added_count = self.register_inner(operation, false);

        if added_count > 0 {
            self.log_accepted(operation, added_count);
            return true;
        }

        if added_count == 0 && self.blocks.len() < max_blocks {
            self.on_new_block(operation);
            return true;
        }

        self.merge_blocks(operation, true);

        if self.blocks.len() >= max_blocks {
            self.stopped = true;
            return false;
        }

        let added_count = self.register_inner(operation, false);

        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            self.log_accepted(operation, added_count);
        }

        true
    }

    fn log_accepted(&self, operation: &OperationIr, added_count: usize) {
        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → accepted in {added_count}/{num_blocks} block(s)",
                op_kind(operation)
            )
        });
    }

    fn register_inner(&mut self, operation: &OperationIr, force: bool) -> usize {
        let mut added_count = 0;
        for block in self.blocks.iter_mut() {
            match block.register(operation, self.length, force) {
                RegistrationResult::Accepted => {
                    added_count += 1;
                }
                RegistrationResult::NotPartOfTheGraph => {}
            }
        }
        added_count
    }

    fn new_empty_search(&self) -> Self {
        Self::new(
            self.builders
                .iter()
                .map(|b| {
                    let mut b = b.clone_dyn();
                    b.reset();
                    b
                })
                .collect(),
        )
    }

    fn merge_blocks(&mut self, operation: &OperationIr, all: bool) -> MergeBlockStep {
        let nodes = operation.nodes();
        let mut block_merges = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            if all || block.contains_tensors(&nodes) {
                block_merges.push(i);
            }
        }

        if block_merges.len() <= 1 {
            return MergeBlockStep::NoNeed;
        }

        let blocks_to_merge = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match block_merges.contains(&i) {
                true => Some(g),
                false => None,
            })
            .collect::<Vec<_>>();

        let merged = merge_blocks(&blocks_to_merge, false);

        let mut clear_blocks = || {
            let mut indices = block_merges.to_vec();
            indices.sort();

            for g in indices.into_iter().rev() {
                self.blocks.remove(g);
            }
        };

        match merged {
            MergeBlocksResult::Full(block) => {
                clear_blocks();
                self.blocks.push(block);
                Block::sort(&mut self.blocks);
                MergeBlockStep::Full
            }
            MergeBlocksResult::Partial {
                mut merged,
                mut failed,
            } => {
                clear_blocks();
                self.blocks.append(&mut merged);
                self.blocks.append(&mut failed);
                Block::sort(&mut self.blocks);
                MergeBlockStep::Partial
            }
            MergeBlocksResult::Fail => MergeBlockStep::Fail,
        }
    }

    /// When `operation` couples blocks that can't be merged, the stream stops and flushes every
    /// current block. But a block whose ops are all input-less creation ops (e.g. a `zeros`/`empty`
    /// that `operation` writes into) can still fuse with `operation` itself — the merge only failed
    /// against an *incompatible sibling* it happened to also touch (a differently-shaped `value`
    /// producer, say). Flushing it here materializes the buffer and forces `operation` to read it
    /// back as a global input.
    ///
    /// Instead, drop such blocks from this round so they re-enter the next round together with
    /// `operation` and fuse on-write. Creation blocks have no inputs, so re-emitting them next
    /// round is free and cannot disturb tensor lifetimes. Only blocks that `operation` would
    /// genuinely fuse with are dropped (so a `slice_assign`'s indexed `value` producer — which can
    /// never become a fused local — is still flushed), and at least one block is always kept so the
    /// round makes progress.
    fn defer_fusable_creation_blocks(&mut self, operation: &OperationIr) {
        let nodes = operation.nodes();
        let order = self.length;

        let deferrable = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| {
                block.contains_tensors(&nodes)
                    && block.is_creation_only()
                    && block.would_fuse(operation, order)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();

        // Keep at least one block so the round still flushes progress (an empty block list would
        // make `optimize` operate on nothing).
        if deferrable.is_empty() || deferrable.len() >= self.blocks.len() {
            return;
        }

        for index in deferrable.into_iter().rev() {
            self.blocks.remove(index);
        }
    }

    fn on_new_block(&mut self, operation: &OperationIr) {
        let mut block = Block::new(&self.builders);
        block.register(operation, self.length, true);
        self.blocks.push(block);

        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → new block (total: {num_blocks})",
                op_kind(operation)
            )
        });
    }
}

enum MergeBlockStep {
    Full,
    Partial,
    Fail,
    NoNeed,
}
