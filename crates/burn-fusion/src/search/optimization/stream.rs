use super::blocks::BlocksOptimizer;
use crate::{
    NumOperations, OperationFuser,
    search::{
        Block, BlockOptimization, OperationNode, RegistrationResult,
        graph::{Dag, GraphNode, is_valid_execution_order},
        merging::{MergeBlocksResult, merge_blocks},
        optimization::blocks::BlocksOptimizerResult,
    },
    stream::{execution::op_kind, store::ExecutionStrategy},
};
use burn_ir::{OperationIr, TensorId, TensorStatus};
use burn_std::config::{config, fusion::FusionLogLevel, log_fusion};
use std::collections::HashSet;
use std::sync::Arc;

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
            MergeBlockStep::Fail | MergeBlockStep::Partial => {
                // The operation ties together blocks that couldn't be merged into one. Instead of
                // giving up on the segment, give the operation its own block that *depends* on the
                // blocks it reads from — the block set becomes a dependency DAG.
                self.on_dependent_op(operation);
                if self.stopped {
                    return;
                }
                self.length += 1;
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

        let out = match result {
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

                    // Append the re-optimized holes as their own chunk; `repair_order` (below) puts
                    // every chunk into a hazard-respecting execution order, so the placement here
                    // doesn't need to be correct — only complete.
                    let consumed = optimization_of_holes.ordering.len();
                    strategies.push(Box::new(optimization_of_holes.strategy));
                    ordering.append(&mut optimization_of_holes.ordering);
                    holes.drain(0..consumed);

                    if holes.is_empty() {
                        break;
                    }
                }

                BlockOptimization::new(ExecutionStrategy::Composed(strategies), ordering)
            }
        };

        repair_order(out, operations)
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

        // Seed each block with its index, then build the cycle guard over ALL blocks — a block
        // that "sits between" two merge candidates need not itself be a candidate, so the guard
        // must see the full dependency graph, not just the subset being merged.
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.seed_constituent(i);
        }
        let all_blocks = self.blocks.iter().collect::<Vec<_>>();
        let guard = Dag::new(&all_blocks).reachability();

        let blocks_to_merge = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match block_merges.contains(&i) {
                true => Some(g),
                false => None,
            })
            .collect::<Vec<_>>();

        let merged = merge_blocks(&blocks_to_merge, false, &guard);

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

    /// Give the operation its own block that depends on the blocks it reads from.
    ///
    /// The new block owns the operation's outputs and records its upstream inputs as external —
    /// those external inputs form the dependency edges. A freshly created block is a pure sink, so
    /// it cannot create a cycle, except through an in-place output that re-produces an upstream
    /// tensor; that case is caught by the defensive acyclicity check.
    fn on_dependent_op(&mut self, operation: &OperationIr) {
        // Dependent blocks count toward the cap. With no room left, try to free a slot by
        // force-merging mergeable blocks; if that fails, stop the segment (the op is deferred).
        if let Some(max_blocks) = self.max_blocks
            && self.blocks.len() >= max_blocks
        {
            self.merge_blocks(operation, true);

            if self.blocks.len() >= max_blocks {
                let length = self.length;
                log_fusion(FusionLogLevel::Medium, || {
                    format!(
                        "[stream] stopped (max_blocks={max_blocks} reached on dependency) at op {length} ({})",
                        op_kind(operation)
                    )
                });
                self.stopped = true;
                return;
            }
        }

        let mut block = Block::new(&self.builders);
        block.register(operation, self.length, true);
        self.blocks.push(block);

        if !Dag::new(&self.blocks).is_acyclic() {
            self.blocks.pop();
            let length = self.length;
            log_fusion(FusionLogLevel::Medium, || {
                format!(
                    "[stream] stopped (unresolvable dependency cycle) at op {length} ({})",
                    op_kind(operation)
                )
            });
            self.stopped = true;
            return;
        }

        let length = self.length;
        let num_blocks = self.blocks.len();
        log_fusion(FusionLogLevel::Full, || {
            format!(
                "[stream] op {length} {} → new dependent block (total: {num_blocks})",
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

/// Reorder the top-level strategy chunks so the execution order respects tensor handle lifetimes.
///
/// Blocks and re-optimized holes are placed by separate, incremental heuristics that can't see the
/// whole picture (a hole may depend on another hole spliced later). This final pass treats each
/// top-level strategy as an atomic [Chunk] node and orders the chunks topologically (see
/// [GraphNode] for the hazard rules deriving the edges). Ties keep the earliest stream position
/// first, reproducing the historical order when there are no hazards.
///
/// If the chunk graph has a cycle — the chunking can't be linearized as atomic units — unfused
/// chunks are [split](repair_order_split) to break the cycle before giving up on fusion.
fn repair_order<O>(opt: BlockOptimization<O>, operations: &[OperationIr]) -> BlockOptimization<O> {
    let (strategies, ordering) = match opt.strategy {
        ExecutionStrategy::Composed(items) => (items, opt.ordering),
        // A single strategy has nothing to reorder across, but a merge can still leave its ops
        // internally out of stream order — validate, and unfuse in stream order if it's broken.
        single => {
            if ordering_is_valid(&opt.ordering, operations) {
                return BlockOptimization::new(single, opt.ordering);
            }
            return unfused_stream_order(opt.ordering);
        }
    };

    // Split the concatenated ordering back into one chunk per top-level strategy.
    let mut chunks = Vec::with_capacity(strategies.len());
    let mut offset = 0;
    for strategy in &strategies {
        let len = strategy_len(strategy);
        chunks.push(Chunk::new(
            ordering[offset..offset + len].to_vec(),
            operations,
        ));
        offset += len;
    }

    let order = match Dag::new(&chunks).topological_order() {
        Some(order) => order,
        // The chunks can't be linearized as atomic units. Unfused chunks are free to split
        // (their ops execute individually), which may break the cycle while every fused
        // optimization survives intact.
        None => return repair_order_split(strategies, chunks, operations),
    };

    assemble(strategies, &chunks, &order, operations)
}

/// Retry [repair_order] with every [Operations](ExecutionStrategy::Operations) chunk split into
/// single-operation chunks, keeping fused optimizations atomic. Splitting only removes ordering
/// constraints, so this resolves any cycle that isn't between fused chunks themselves.
fn repair_order_split<O>(
    strategies: Vec<Box<ExecutionStrategy<O>>>,
    chunks: Vec<Chunk>,
    operations: &[OperationIr],
) -> BlockOptimization<O> {
    let mut split_strategies = Vec::new();
    let mut split_chunks = Vec::new();
    for (strategy, chunk) in strategies.into_iter().zip(chunks) {
        match *strategy {
            ExecutionStrategy::Operations { .. } => {
                for position in chunk.positions {
                    split_strategies.push(Box::new(ExecutionStrategy::Operations {
                        ordering: Arc::new(vec![position]),
                    }));
                    split_chunks.push(Chunk::new(vec![position], operations));
                }
            }
            strategy => {
                split_strategies.push(Box::new(strategy));
                split_chunks.push(chunk);
            }
        }
    }

    match Dag::new(&split_chunks).topological_order() {
        Some(order) => assemble(split_strategies, &split_chunks, &order, operations),
        // Fused chunks are mutually dependent: the segment isn't executable as built — run
        // everything unfused in stream order, which is always valid.
        None => unfused_stream_order(
            split_chunks
                .into_iter()
                .flat_map(|chunk| chunk.positions)
                .collect(),
        ),
    }
}

/// Emit the strategies in the given chunk order, validating the final operation-level ordering.
fn assemble<O>(
    strategies: Vec<Box<ExecutionStrategy<O>>>,
    chunks: &[Chunk],
    order: &[usize],
    operations: &[OperationIr],
) -> BlockOptimization<O> {
    let mut slots: Vec<Option<Box<ExecutionStrategy<O>>>> =
        strategies.into_iter().map(Some).collect();
    let mut new_strategies = Vec::with_capacity(order.len());
    let mut new_ordering = Vec::new();
    for &k in order {
        new_strategies.push(slots[k].take().expect("each chunk taken once"));
        new_ordering.extend_from_slice(&chunks[k].positions);
    }

    // A chunk that is internally mis-ordered can still violate handle lifetimes even after the
    // chunk-level sort. If the result isn't executable, fall back to running every operation
    // unfused in stream order.
    if !ordering_is_valid(&new_ordering, operations) {
        return unfused_stream_order(new_ordering);
    }

    BlockOptimization::new(ExecutionStrategy::Composed(new_strategies), new_ordering)
}

/// Run every operation unfused, in ascending stream order — always a valid execution order.
fn unfused_stream_order<O>(mut positions: Vec<usize>) -> BlockOptimization<O> {
    positions.sort_unstable();
    let ordering = Arc::new(positions.clone());
    BlockOptimization::new(ExecutionStrategy::Operations { ordering }, positions)
}

/// Whether the execution order respects tensor handle lifetimes: every operation's inputs must be
/// live when it runs (see [is_valid_execution_order]).
fn ordering_is_valid(ordering: &[usize], operations: &[OperationIr]) -> bool {
    let nodes = operations
        .iter()
        .enumerate()
        .map(|(position, operation)| OperationNode {
            position,
            operation,
        })
        .collect::<Vec<_>>();

    is_valid_execution_order(&nodes, ordering)
}

/// Number of stream positions a strategy covers (its share of the concatenated ordering).
fn strategy_len<O>(strategy: &ExecutionStrategy<O>) -> usize {
    match strategy {
        ExecutionStrategy::Optimization { ordering, .. } => ordering.len(),
        ExecutionStrategy::Operations { ordering } => ordering.len(),
        ExecutionStrategy::Composed(items) => items.iter().map(|s| strategy_len(s)).sum(),
    }
}

/// One top-level strategy of a composed optimization, viewed as a single atomic [GraphNode]
/// covering the stream positions of its operations.
struct Chunk {
    positions: Vec<usize>,
    produced: HashSet<TensorId>,
    read: HashSet<TensorId>,
    freed: HashSet<TensorId>,
}

impl Chunk {
    fn new(positions: Vec<usize>, operations: &[OperationIr]) -> Self {
        let mut produced = HashSet::new();
        for &position in &positions {
            for tensor in operations[position].outputs() {
                produced.insert(tensor.id);
            }
        }

        let mut read = HashSet::new();
        let mut freed = HashSet::new();
        for &position in &positions {
            for tensor in operations[position].inputs() {
                // A ReadWrite read frees the tensor even when this chunk also produced it — the
                // write-after-read hazard against readers in *other* chunks still applies
                // (mirrors [Block::register_op]).
                if let TensorStatus::ReadWrite = tensor.status {
                    freed.insert(tensor.id);
                }
                if produced.contains(&tensor.id) {
                    continue;
                }
                read.insert(tensor.id);
            }
        }

        Self {
            positions,
            produced,
            read,
            freed,
        }
    }
}

impl GraphNode for Chunk {
    type Resource = TensorId;

    fn produced(&self) -> impl Iterator<Item = TensorId> {
        self.produced.iter().copied()
    }

    fn read(&self) -> impl Iterator<Item = TensorId> {
        self.read.iter().copied()
    }

    fn freed(&self) -> impl Iterator<Item = TensorId> {
        self.freed.iter().copied()
    }

    fn position(&self) -> usize {
        self.positions.iter().copied().min().unwrap_or(0)
    }
}

/// Tests for the [repair_order] pass, driving it directly with hand-built chunkings.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::execution::tests::TestOptimization;
    use burn_backend::{DType, Shape};
    use burn_ir::{BinaryOpIr, NumericOperationIr, TensorIr};

    fn tensor(id: u64) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::new([32, 32]),
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        }
    }

    fn add(lhs: u64, rhs: u64, out: u64) -> OperationIr {
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor(lhs),
                rhs: tensor(rhs),
                out: tensor(out),
            }),
        )
    }

    /// Like [add] but reads `lhs` with `ReadWrite` status (last use — frees the tensor).
    fn add_rw(lhs: u64, rhs: u64, out: u64) -> OperationIr {
        let mut lhs = tensor(lhs);
        lhs.status = TensorStatus::ReadWrite;
        OperationIr::NumericFloat(
            DType::F32,
            NumericOperationIr::Add(BinaryOpIr {
                lhs,
                rhs: tensor(rhs),
                out: tensor(out),
            }),
        )
    }

    fn fused(ordering: Vec<usize>) -> ExecutionStrategy<TestOptimization> {
        ExecutionStrategy::Optimization {
            opt: TestOptimization::new(0, ordering.len()),
            ordering: Arc::new(ordering),
            score: 0,
        }
    }

    fn unfused(ordering: Vec<usize>) -> ExecutionStrategy<TestOptimization> {
        ExecutionStrategy::Operations {
            ordering: Arc::new(ordering),
        }
    }

    fn composed(
        parts: Vec<ExecutionStrategy<TestOptimization>>,
    ) -> ExecutionStrategy<TestOptimization> {
        ExecutionStrategy::Composed(parts.into_iter().map(Box::new).collect())
    }

    /// A chunk producing AND freeing the same tensor still constrains readers in other chunks:
    /// the free must be a write-after-read edge. With the edge, the chunk-level cycle is
    /// detected and the unfused chunk is split around the fused one, salvaging the fusion;
    /// without it, the invalid order is only caught by the final validation, which unfuses the
    /// whole segment.
    #[test]
    fn splits_unfused_produce_free_chunk_instead_of_unfusing_everything() {
        let ops = vec![
            add(1, 2, 10),     // 0: produces 10 (unfused chunk)
            add(10, 3, 100),   // 1: fused, reads 10
            add(100, 4, 101),  // 2: fused
            add_rw(10, 5, 11), // 3: frees 10 (same unfused chunk as op 0)
        ];
        let opt = BlockOptimization::new(
            composed(vec![fused(vec![1, 2]), unfused(vec![0, 3])]),
            vec![1, 2, 0, 3],
        );

        let repaired = repair_order(opt, &ops);

        // The fused pair survives; the produce/free chunk is split around it.
        assert_eq!(repaired.ordering, vec![0, 1, 2, 3]);
        assert_eq!(
            repaired.strategy,
            composed(vec![unfused(vec![0]), fused(vec![1, 2]), unfused(vec![3])]),
        );
    }
}
