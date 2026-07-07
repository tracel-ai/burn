use crate::{
    FuserStatus, NumOperations, OperationFuser,
    search::graph::{GraphNode, SubGraph, is_valid_execution_order},
    stream::store::ExecutionStrategy,
};
use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
use std::{collections::HashSet, sync::Arc};

/// A block represents a list of operations, not necessarily in the same order as the execution
/// stream.
///
/// The start and end position of the relative execution stream are tracked in the block alongside
/// the ordering.
pub struct Block<O> {
    builders: Vec<Box<dyn OperationFuser<O>>>,
    operations: Vec<OperationIr>,
    ids: HashSet<TensorId>,
    /// Tensor ids produced (as an output) by an operation of this block.
    produced: HashSet<TensorId>,
    /// Tensor ids consumed by this block but produced elsewhere (external inputs).
    read: HashSet<TensorId>,
    /// Tensor ids this block reads with [ReadWrite](TensorStatus::ReadWrite) status — i.e. it is
    /// the last use and frees/reuses the buffer in place.
    freed: HashSet<TensorId>,
    /// The original blocks this block subsumes.
    ///
    /// Seeded to a single node by the optimizer before a merge pass and unioned on
    /// [merge](Self::merge), so the [Reachability](crate::search::graph::Reachability) guard can
    /// reason about the original dependency graph through the recursive merging.
    constituents: SubGraph,
    ordering: Vec<usize>,
    /// The start position in the relative execution stream.
    pub start_pos: usize,
    /// The end position in the relative execution stream.
    pub end_pos: usize,
}

/// The result of [registering](Block::register) an [operation](OperationIr).
pub enum RegistrationResult {
    /// If the [operation](OperationIr) is correctly registered.
    Accepted,
    /// If the [operation](OperationIr) isn't part of the graph.
    ///
    /// In this case the operation isn't registered.
    NotPartOfTheGraph,
}

/// The optimization found for a [block](Block).
#[derive(Debug, new)]
pub struct BlockOptimization<O> {
    /// The [execution strategy](ExecutionStrategy) to be used to execute the [block](Block).
    pub strategy: ExecutionStrategy<O>,
    /// The ordering of each operation in the relative execution stream.
    pub ordering: Vec<usize>,
}

impl<O> Block<O> {
    /// The original blocks this block subsumes.
    pub fn constituents(&self) -> &SubGraph {
        &self.constituents
    }

    /// Seed this block's [constituents](Self::constituents) to a single original block index.
    pub fn seed_constituent(&mut self, index: usize) {
        self.constituents = SubGraph::single(index);
    }
}

/// A single operation at its stream position, viewed as a [GraphNode].
pub struct OperationNode<'a> {
    /// The operation.
    pub operation: &'a OperationIr,
    /// The stream position of the operation.
    pub position: usize,
}

impl GraphNode for OperationNode<'_> {
    type Resource = TensorId;

    fn produced(&self) -> impl Iterator<Item = TensorId> {
        self.operation.outputs().map(|tensor| tensor.id)
    }

    fn read(&self) -> impl Iterator<Item = TensorId> {
        self.operation.inputs().map(|tensor| tensor.id)
    }

    fn freed(&self) -> impl Iterator<Item = TensorId> {
        self.operation
            .inputs()
            .filter(|tensor| matches!(tensor.status, TensorStatus::ReadWrite))
            .map(|tensor| tensor.id)
    }

    fn position(&self) -> usize {
        self.position
    }
}

/// The dependency edges between blocks are derived from what each block reads, produces, and
/// frees — see [GraphNode].
impl<O> GraphNode for Block<O> {
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

    fn produces(&self, resource: TensorId) -> bool {
        self.produced.contains(&resource)
    }

    fn reads(&self, resource: TensorId) -> bool {
        self.read.contains(&resource)
    }

    fn position(&self) -> usize {
        self.start_pos
    }
}

impl<O: NumOperations> Block<O> {
    /// Create a new block that will be optimized with the provided [optimization builders](OptimizationBuilder).
    pub fn new(builders: &[Box<dyn OperationFuser<O>>]) -> Self {
        Self {
            builders: builders.iter().map(|o| o.clone_dyn()).collect(),
            operations: Vec::new(),
            ids: HashSet::new(),
            produced: HashSet::new(),
            read: HashSet::new(),
            freed: HashSet::new(),
            constituents: SubGraph::empty(),
            ordering: Vec::new(),
            start_pos: usize::MAX,
            end_pos: usize::MIN,
        }
    }

    /// Sort the [blocks](Block) based on the start position.
    pub fn sort(blocks: &mut [Self]) {
        blocks.sort_by_key(|a| a.start_pos);
    }

    /// Optimize the block.
    pub fn optimize(mut self) -> BlockOptimization<O> {
        match find_best_optimization_index(&mut self.builders) {
            BestOptimization::Found { index, score } => {
                let opt = self.builders[index].finish();
                let opt_len = opt.len();
                if opt_len < self.operations.len() {
                    self.ordering.drain(opt_len..);
                }

                let strategy = ExecutionStrategy::Optimization {
                    ordering: Arc::new(self.ordering.clone()),
                    opt,
                    score,
                };
                BlockOptimization::new(strategy, self.ordering)
            }
            BestOptimization::NotFound => {
                let strategy = ExecutionStrategy::Operations {
                    ordering: Arc::new(self.ordering.clone()),
                };
                BlockOptimization::new(strategy, self.ordering)
            }
        }
    }

    /// Returns if the block contains any of the provided [tensors](TensorIr).
    pub fn contains_tensors(&self, tensors: &[&TensorIr]) -> bool {
        for node in tensors {
            if self.ids.contains(&node.id) {
                return true;
            }
        }

        false
    }

    /// Merge the current block with the other one and returns if the operation is successful.
    ///
    /// # Warning
    ///
    /// This will modify the current block even if the other block isn't correctly merged.
    pub fn merge(&mut self, other: &Block<O>) -> bool {
        // A block executes as one contiguous unit in registration order (fused kernels replay
        // the fusion order). Appending the other block's operations can place a consumer before
        // its producer — or a free before a read — when the blocks depend on each other. Reject
        // such merges before mutating anything; the caller can retry in the other direction.
        if !self.can_append(other) {
            return false;
        }

        let self_ready = self.has_ready_optimization();
        let other_ready = other.has_ready_optimization();

        // Absorb the other block's provenance so the merge guard sees the combined set on any
        // subsequent `can_contract` check.
        self.constituents.union_with(&other.constituents);

        for (op, pos) in other.operations.iter().zip(&other.ordering) {
            self.register(op, *pos, true);
        }

        // If the merged block can still be improved, keep it — that's the
        // usual lazy-optimization signal.
        if self.still_optimizing() {
            return true;
        }

        // Otherwise, only accept the merge when it *creates* a ready fusion
        // that didn't exist separately. If either side already had a ready
        // fusion before the merge, merging would collapse them and hide one
        // of the fusions — keep the blocks separate instead.
        !self_ready && !other_ready && self.has_ready_optimization()
    }

    fn has_ready_optimization(&self) -> bool {
        self.builders.iter().any(|b| b.properties().ready)
    }

    /// Whether appending the other block's operations after this block's — the registration
    /// order a [merge](Self::merge) in this direction produces — respects every tensor lifetime
    /// (no read before the producing operation, no read after the freeing operation).
    ///
    /// This is the validity check `merge` performs, exposed so callers can test a merge
    /// direction *before* paying for a deep clone of the base block.
    pub fn can_append<'a>(&'a self, other: &'a Block<O>) -> bool {
        is_valid_execution_order(self.operation_nodes().chain(other.operation_nodes()))
    }

    /// The block's operations in registration order, viewed as [graph nodes](OperationNode).
    fn operation_nodes(&self) -> impl Iterator<Item = OperationNode<'_>> + Clone {
        self.operations
            .iter()
            .zip(&self.ordering)
            .map(|(operation, &position)| OperationNode {
                operation,
                position,
            })
    }

    /// Register an [operation](OperationIr) in the current block.
    ///
    /// You need to provide the order of the operation as well as a force flag.
    ///
    /// When the force flag is true, the builder will always accept the operation, otherwise it
    /// might refuse it if the operation [isn't part of the graph](RegistrationResult::NotPartOfTheGraph).
    ///
    /// Forcing is useful to fuse operations that are part of different graphs, but included
    /// in the same optimization.
    pub fn register(
        &mut self,
        operation: &OperationIr,
        order: usize,
        force: bool,
    ) -> RegistrationResult {
        if self.ids.is_empty() {
            self.register_op(operation, order);
            return RegistrationResult::Accepted;
        }
        let mut contains = false;
        for node in operation.nodes() {
            contains = self.ids.contains(&node.id);

            if contains {
                break;
            }
        }

        if !contains && !force {
            return RegistrationResult::NotPartOfTheGraph;
        }

        self.register_op(operation, order);
        RegistrationResult::Accepted
    }

    /// If the block can still be optimized further.
    pub fn still_optimizing(&self) -> bool {
        let mut num_stopped = 0;

        for optimization in self.builders.iter() {
            if let FuserStatus::Closed = optimization.status() {
                num_stopped += 1
            }
        }

        num_stopped < self.builders.len()
    }

    fn register_op(&mut self, operation: &OperationIr, pos: usize) {
        self.operations.push(operation.clone());
        self.ordering.push(pos);

        if pos < self.start_pos {
            self.start_pos = pos;
        }
        if pos + 1 > self.end_pos {
            self.end_pos = pos + 1;
        }

        for builder in self.builders.iter_mut() {
            builder.fuse(operation);
        }

        for node in operation.nodes() {
            self.ids.insert(node.id);
        }

        // Maintain the produced / external-read sets incrementally. A consumed id counts as an
        // external read only while nothing in the block has produced it; producing an id makes
        // it internal (and clears any earlier external record). This is order-independent, so it
        // stays correct when `merge` folds ops in a non-causal order.
        for node in operation.inputs() {
            if !self.produced.contains(&node.id) {
                self.read.insert(node.id);
            }
            if let TensorStatus::ReadWrite = node.status {
                self.freed.insert(node.id);
            }
        }
        for node in operation.outputs() {
            self.produced.insert(node.id);
            self.read.remove(&node.id);
        }
    }
}

impl<O> BlockOptimization<O> {
    /// Maps the ordering of the current block optimization using the given mapping.
    pub fn map_ordering(&mut self, mapping: &[usize]) {
        for i in self.ordering.iter_mut() {
            *i = mapping[*i];
        }
        self.strategy.map_ordering(mapping);
    }
}

impl<O> ExecutionStrategy<O> {
    /// Maps the ordering of the current execution strategy using the given mapping.
    pub fn map_ordering(&mut self, mapping: &[usize]) {
        match self {
            ExecutionStrategy::Optimization { ordering, .. } => {
                let mut ordering_mapped = ordering.to_vec();

                for o in ordering_mapped.iter_mut() {
                    *o = mapping[*o];
                }
                *ordering = Arc::new(ordering_mapped);
            }
            ExecutionStrategy::Operations { ordering } => {
                let mut ordering_mapped = ordering.to_vec();

                for o in ordering_mapped.iter_mut() {
                    *o = mapping[*o];
                }

                *ordering = Arc::new(ordering_mapped);
            }
            ExecutionStrategy::Composed(items) => {
                for item in items.iter_mut() {
                    item.map_ordering(mapping);
                }
            }
        }
    }
}

enum BestOptimization {
    NotFound,
    Found { index: usize, score: u64 },
}

fn find_best_optimization_index<O>(
    optimizations: &mut [Box<dyn OperationFuser<O>>],
) -> BestOptimization {
    let mut best_index = BestOptimization::NotFound;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = optimization.properties();

        // A score of zero is worse than fusing.
        if properties.ready && properties.score > best_score {
            best_index = BestOptimization::Found {
                index: i,
                score: properties.score,
            };
            best_score = properties.score;
        }
    }

    best_index
}

impl<O> PartialEq for Block<O> {
    fn eq(&self, other: &Self) -> bool {
        // Since the ordering can be seen as operation ids, we can use it to compare
        // blocks.
        let mut sorted_a = self.ordering.clone();
        let mut sorted_b = other.ordering.clone();
        sorted_a.sort();
        sorted_b.sort();

        sorted_a == sorted_b
    }
}

impl<O> core::fmt::Debug for Block<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Block {{ pos: [{:?}, {:?}; {:?}] }}",
            self.start_pos,
            self.end_pos,
            self.ordering.len(),
        ))
    }
}

impl<O> Clone for Block<O> {
    fn clone(&self) -> Self {
        Self {
            builders: self.builders.iter().map(|b| b.clone_dyn()).collect(),
            operations: self.operations.clone(),
            ids: self.ids.clone(),
            produced: self.produced.clone(),
            read: self.read.clone(),
            freed: self.freed.clone(),
            constituents: self.constituents.clone(),
            ordering: self.ordering.clone(),
            start_pos: self.start_pos,
            end_pos: self.end_pos,
        }
    }
}

/// Integration of [Block] with the [graph](crate::search::graph) algorithms: dependency edges
/// between blocks are derived from the tensors they read, produce, and free.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::graph::Dag;
    use crate::search::testing::{add, add_rw};
    use crate::stream::execution::tests::TestOptimization;

    /// A block owning the given `(operation, position)` pairs. No builders needed — the
    /// dependency analysis only reads the data-flow sets and `start_pos`.
    fn block(ops: &[(OperationIr, usize)]) -> Block<TestOptimization> {
        let builders: Vec<Box<dyn OperationFuser<TestOptimization>>> = Vec::new();
        let mut block = Block::new(&builders);
        for (op, pos) in ops {
            block.register(op, *pos, true);
        }
        block
    }

    #[test]
    fn dependency_beats_start_pos_in_topological_order() {
        // Block 0 (start_pos 0) consumes tensor 50, which block 1 (start_pos 5) produces.
        let blocks = vec![
            block(&[(add(50, 1, 2), 0)]), // Consumes 50 -> depends on block 1.
            block(&[(add(9, 8, 50), 5)]), // Produces 50.
        ];

        assert_eq!(Dag::new(&blocks).topological_order(), Some(vec![1, 0]));
    }

    #[test]
    fn mutual_dependency_between_blocks_is_a_cycle() {
        // Block 0 produces 100 (consumed by block 1) and consumes 200 (produced by block 1).
        let blocks = vec![
            block(&[(add(1, 2, 100), 0), (add(200, 3, 101), 1)]),
            block(&[(add(100, 4, 200), 2)]),
        ];

        assert!(!Dag::new(&blocks).is_acyclic());
    }

    #[test]
    fn freeing_block_runs_after_reading_block() {
        // Both blocks read tensor 100 (produced elsewhere): block 0 read-only, block 1 frees it
        // (ReadWrite). The freer must run after the reader.
        let blocks = vec![
            block(&[(add(100, 1, 2), 0)]),    // Reads 100 read-only.
            block(&[(add_rw(100, 3, 4), 1)]), // Frees 100.
        ];

        assert_eq!(Dag::new(&blocks).topological_order(), Some(vec![0, 1]));
    }

    #[test]
    fn merge_guard_blocks_contraction_around_intermediate_block() {
        // A(0) -> C(1) -> B(2): C sits between A and B, so merging A and B would create a cycle.
        let mut blocks = vec![
            block(&[(add(1, 2, 100), 0)]),   // A: produces 100.
            block(&[(add(100, 3, 200), 1)]), // C: consumes 100, produces 200.
            block(&[(add(200, 4, 300), 2)]), // B: consumes 200.
        ];
        for (i, block) in blocks.iter_mut().enumerate() {
            block.seed_constituent(i);
        }
        let guard = Dag::new(&blocks).reachability();

        // A and B cannot merge (C is between them)...
        assert!(!guard.can_contract(blocks[0].constituents(), blocks[2].constituents()));
        // ...but each direct edge is fine.
        assert!(guard.can_contract(blocks[0].constituents(), blocks[1].constituents()));
        assert!(guard.can_contract(blocks[1].constituents(), blocks[2].constituents()));
    }

    #[test]
    fn merge_guard_refuses_unseeded_blocks() {
        let blocks = vec![block(&[(add(1, 2, 3), 0)]), block(&[(add(4, 5, 6), 1)])];
        let guard = Dag::new(&blocks).reachability();

        assert!(!guard.can_contract(blocks[0].constituents(), blocks[1].constituents()));
    }

    /// A block with a builder open to any prefix of `pattern`, owning the given ops.
    ///
    /// The builder keeps the block `still_optimizing` as long as the registered ops follow the
    /// pattern, so merge acceptance is driven by the pattern and the tests below can isolate the
    /// internal-order validation.
    fn block_with_pattern(
        pattern: &[OperationIr],
        ops: &[(OperationIr, usize)],
    ) -> Block<TestOptimization> {
        use crate::stream::execution::tests::TestOptimizationBuilder;

        let builders: Vec<Box<dyn OperationFuser<TestOptimization>>> =
            vec![Box::new(TestOptimizationBuilder::new(0, pattern.to_vec()))];
        let mut block = Block::new(&builders);
        for (op, pos) in ops {
            block.register(op, *pos, true);
        }
        block
    }

    #[test]
    fn merge_rejects_consumer_fused_before_producer() {
        let x = add(1, 2, 10);
        let y = add(50, 3, 11); // Reads 50...
        let z = add(4, 5, 50); // ...which this op produces.

        // The pattern accepts the forward merge order [x, y, z], so only the internal-order
        // validation can reject it: y would be fused before the op producing its input.
        let pattern = [x.clone(), y.clone(), z.clone(), x.clone()];
        let mut base = block_with_pattern(&pattern, &[(x, 0), (y, 1)]);
        let producer = block_with_pattern(&pattern, &[(z, 2)]);

        assert!(!base.merge(&producer));
    }

    #[test]
    fn merge_accepts_producer_fused_before_consumer() {
        let x = add(1, 2, 10);
        let y = add(50, 3, 11);
        let z = add(4, 5, 50);

        // Same blocks as above, merged in the other direction: [z, x, y] is causal.
        let pattern = [z.clone(), x.clone(), y.clone(), x.clone()];
        let consumers = block_with_pattern(&pattern, &[(x, 0), (y, 1)]);
        let mut base = block_with_pattern(&pattern, &[(z, 2)]);

        assert!(base.merge(&consumers));
    }

    #[test]
    fn merge_rejects_free_fused_before_read() {
        let reader = add(100, 1, 2); // Reads 100.
        let freer = add_rw(100, 3, 4); // Frees 100.

        // Fusing the free before the read would release the tensor under the reader.
        let pattern = [freer.clone(), reader.clone(), reader.clone()];
        let mut base = block_with_pattern(&pattern, &[(freer, 1)]);
        let other = block_with_pattern(&pattern, &[(reader, 0)]);

        assert!(!base.merge(&other));
    }
}
