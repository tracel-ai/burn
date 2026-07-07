use std::collections::HashSet;

use burn_ir::TensorId;

use super::Block;

/// Whether `a` depends on `b` (so `b` must execute before `a`). Two reasons:
///
/// - **read-after-write**: `a` consumes a tensor `b` produces.
/// - **write-after-read (anti-dependency)**: `a` reads a tensor with `ReadWrite` status — it is the
///   last use and frees/reuses the buffer — while `b` also reads that tensor. Reordering `a` before
///   `b` would free the buffer out from under `b`. This is the hazard that keeps in-place ops
///   correct once operations are reordered across blocks.
fn depends_on<O>(a: &Block<O>, b: &Block<O>) -> bool {
    intersects(a.inputs_external(), b.produced()) || intersects(a.reads_rw(), b.inputs_external())
}

fn intersects(a: &HashSet<TensorId>, b: &HashSet<TensorId>) -> bool {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().any(|id| large.contains(id))
}

/// A valid execution order (dependencies first) of the blocks, or `None` if the block
/// dependency graph has a cycle and therefore cannot be linearized.
///
/// Ready nodes are emitted by ascending `start_pos` (then index) so that an edge-free graph
/// reproduces the historical `start_pos` ordering.
pub fn topological_order<O>(blocks: &[Block<O>]) -> Option<Vec<usize>> {
    let n = blocks.len();

    // `indegree[i]` = number of distinct blocks `i` depends on.
    // `successors[j]` = blocks that depend on `j` (and whose indegree drops when `j` is emitted).
    let mut indegree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if depends_on(&blocks[i], &blocks[j]) {
                indegree[i] += 1;
                successors[j].push(i);
            }
        }
    }

    let mut order = Vec::with_capacity(n);
    let mut done = vec![false; n];

    for _ in 0..n {
        // Pick the ready node (indegree 0, not yet emitted) with the smallest `start_pos`.
        let mut pick: Option<usize> = None;
        for k in 0..n {
            if done[k] || indegree[k] != 0 {
                continue;
            }
            match pick {
                None => pick = Some(k),
                Some(p) => {
                    if (blocks[k].start_pos, k) < (blocks[p].start_pos, p) {
                        pick = Some(k);
                    }
                }
            }
        }

        // No ready node while blocks remain => cycle.
        let k = pick?;
        done[k] = true;
        order.push(k);
        for &s in &successors[k] {
            indegree[s] -= 1;
        }
    }

    Some(order)
}

/// Whether the block dependency graph cannot be linearized (a dependency that can't be resolved
/// because of the reordering of operations).
pub fn has_cycle<O>(blocks: &[Block<O>]) -> bool {
    topological_order(blocks).is_none()
}

/// Reachability oracle over an *original* set of blocks, keyed by [constituent](Block::constituents)
/// bitmasks.
///
/// Built once from the pre-merge blocks (each seeded with a single distinct bit), it answers
/// whether contracting two working blocks — whose constituents may already be unions of several
/// original blocks — would create a dependency cycle.
pub struct MergeGuard {
    /// `ancestors[i]` = bits of original blocks that must run before block `i` (transitive).
    ancestors: Vec<u64>,
    /// `descendants[i]` = bits of original blocks that must run after block `i` (transitive).
    descendants: Vec<u64>,
    /// `false` when the block count exceeds the 64-bit mask capacity; the guard then refuses
    /// every merge (conservative but always correct).
    supported: bool,
}

impl MergeGuard {
    /// Build the guard from the current blocks. Each block at position `i` is treated as the
    /// original block with bit `i` (matching [Block::seed_constituent]).
    pub fn new<O>(blocks: &[&Block<O>]) -> Self {
        let n = blocks.len();
        if n > 64 {
            return Self {
                ancestors: Vec::new(),
                descendants: Vec::new(),
                supported: false,
            };
        }

        let mut ancestors = vec![0u64; n];
        let mut descendants = vec![0u64; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if depends_on(blocks[i], blocks[j]) {
                    // `i` depends on `j`: `j` before `i`.
                    ancestors[i] |= 1 << j;
                    descendants[j] |= 1 << i;
                }
            }
        }

        transitive_closure(&mut ancestors);
        transitive_closure(&mut descendants);

        Self {
            ancestors,
            descendants,
            supported: true,
        }
    }

    /// Whether merging the two blocks identified by their constituent masks is cycle-safe.
    ///
    /// Merging is illegal iff some original block outside the union sits *between* the two — it
    /// is both an ancestor and a descendant of the union, so contracting them into one node
    /// would force that block to run both before and after itself.
    pub fn can_merge(&self, a: u64, b: u64) -> bool {
        // Unseeded blocks (mask 0) or an unsupported graph: refuse conservatively.
        if !self.supported || a == 0 || b == 0 {
            return false;
        }

        let s = a | b;
        let mut anc = 0u64;
        let mut desc = 0u64;
        let mut bits = s;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            anc |= self.ancestors[i];
            desc |= self.descendants[i];
        }

        (anc & desc & !s) == 0
    }
}

/// In-place transitive closure of a reachability bitset array (`reach[i]` gains the reachable
/// set of every node it can already reach, to a fixpoint).
fn transitive_closure(reach: &mut [u64]) {
    loop {
        let mut changed = false;
        for i in 0..reach.len() {
            let mut m = reach[i];
            let mut bits = reach[i];
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                m |= reach[j];
            }
            if m != reach[i] {
                reach[i] = m;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::execution::tests::TestOptimization;
    use crate::{OperationFuser, search::Block};
    use burn_backend::{DType, Shape};
    use burn_ir::{BinaryOpIr, NumericOperationIr, OperationIr, TensorId, TensorIr, TensorStatus};

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

    /// Like [add] but reads `lhs` with `ReadWrite` status (it is freed / reused in place).
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

    /// A block owning the given `(operation, position)` pairs. No builders needed — the dependency
    /// helpers only read the produced / external-input sets and `start_pos`.
    fn block(ops: &[(OperationIr, usize)]) -> Block<TestOptimization> {
        let builders: Vec<Box<dyn OperationFuser<TestOptimization>>> = Vec::new();
        let mut b = Block::new(&builders);
        for (op, pos) in ops {
            b.register(op, *pos, true);
        }
        b
    }

    fn guard(blocks: &mut [Block<TestOptimization>]) -> MergeGuard {
        for (i, b) in blocks.iter_mut().enumerate() {
            b.seed_constituent(i);
        }
        let refs = blocks.iter().collect::<Vec<_>>();
        MergeGuard::new(&refs)
    }

    #[test]
    fn topological_order_independent_uses_start_pos() {
        // Two blocks, no shared tensors -> no edges -> ordered by start_pos.
        let blocks = vec![
            block(&[(add(200, 201, 202), 2)]),
            block(&[(add(100, 101, 102), 0)]),
        ];
        assert_eq!(topological_order(&blocks), Some(vec![1, 0]));
        assert!(!has_cycle(&blocks));
    }

    #[test]
    fn topological_order_respects_dependency_over_start_pos() {
        // Block 0 (start_pos 0) consumes tensor 50, which block 1 (start_pos 5) produces.
        // Block 0 therefore depends on block 1 and must run *after* it, despite the smaller
        // start_pos.
        let blocks = vec![
            block(&[(add(50, 1, 2), 0)]),  // consumes 50 -> depends on block 1
            block(&[(add(9, 8, 50), 5)]),  // produces 50
        ];
        assert_eq!(topological_order(&blocks), Some(vec![1, 0]));
    }

    #[test]
    fn has_cycle_detects_mutual_dependency() {
        // Block 0 produces 100 (consumed by block 1) and consumes 200 (produced by block 1).
        let blocks = vec![
            block(&[(add(1, 2, 100), 0), (add(200, 3, 101), 1)]),
            block(&[(add(100, 4, 200), 2)]),
        ];
        assert!(has_cycle(&blocks));
        assert_eq!(topological_order(&blocks), None);
    }

    #[test]
    fn merge_guard_allows_independent_and_direct_edge() {
        // Block 1 depends on block 0 (consumes 102). A direct edge is still mergeable.
        let mut blocks = vec![
            block(&[(add(100, 101, 102), 0)]),
            block(&[(add(102, 103, 104), 1)]),
        ];
        let guard = guard(&mut blocks);
        assert!(guard.can_merge(blocks[0].constituents(), blocks[1].constituents()));
    }

    #[test]
    fn merge_guard_blocks_intermediate_node() {
        // A(0) -> C(1) -> B(2): C sits between A and B, so merging A and B would create a cycle.
        let mut blocks = vec![
            block(&[(add(1, 2, 100), 0)]),   // A: produces 100
            block(&[(add(100, 3, 200), 1)]), // C: consumes 100, produces 200
            block(&[(add(200, 4, 300), 2)]), // B: consumes 200
        ];
        let guard = guard(&mut blocks);
        // A and B cannot merge (C is between them)...
        assert!(!guard.can_merge(blocks[0].constituents(), blocks[2].constituents()));
        // ...but each direct edge is fine.
        assert!(guard.can_merge(blocks[0].constituents(), blocks[1].constituents()));
        assert!(guard.can_merge(blocks[1].constituents(), blocks[2].constituents()));
    }

    #[test]
    fn war_edge_orders_freer_after_reader() {
        // Both blocks read tensor 100 (produced elsewhere): block 0 read-only, block 1 frees it
        // (ReadWrite). The freer must run after the reader, so block 1 depends on block 0.
        let blocks = vec![
            block(&[(add(100, 1, 2), 0)]),    // reads 100 read-only
            block(&[(add_rw(100, 3, 4), 1)]), // frees 100
        ];
        assert!(depends_on(&blocks[1], &blocks[0]));
        assert!(!depends_on(&blocks[0], &blocks[1]));
        // A topological order must place the reader (0) before the freer (1).
        assert_eq!(topological_order(&blocks), Some(vec![0, 1]));
    }

    #[test]
    fn merge_guard_refuses_unseeded_blocks() {
        // Blocks that were never seeded (mask 0) are refused conservatively.
        let blocks = vec![block(&[(add(1, 2, 3), 0)]), block(&[(add(4, 5, 6), 1)])];
        let refs = blocks.iter().collect::<Vec<_>>();
        let guard = MergeGuard::new(&refs);
        assert!(!guard.can_merge(0, 0));
    }
}
