use super::{GraphNode, SubGraph};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A dependency graph over a fixed list of nodes, with edges derived from the nodes' data-flow
/// sets (see [GraphNode]).
///
/// The graph is not guaranteed to be acyclic — a cycle means the nodes cannot be linearized and
/// is reported by [topological_order](Self::topological_order) returning `None`.
pub struct Dag {
    /// `dependencies[i]` = nodes that must execute before node `i` (direct edges only).
    dependencies: Vec<SubGraph>,
    /// `dependents[j]` = nodes that directly depend on node `j` (transpose of `dependencies`).
    dependents: Vec<SubGraph>,
    /// Tie-break position of each node (see [GraphNode::position]).
    positions: Vec<usize>,
}

impl Dag {
    /// Build the dependency graph of the given nodes.
    pub fn new<N: GraphNode>(nodes: &[N]) -> Self {
        let n = nodes.len();
        let mut dependencies = vec![SubGraph::empty(); n];
        let mut dependents = vec![SubGraph::empty(); n];

        for (i, node) in nodes.iter().enumerate() {
            for (j, other) in nodes.iter().enumerate() {
                if i != j && depends_on(node, other) {
                    dependencies[i].insert(j);
                    dependents[j].insert(i);
                }
            }
        }

        Self {
            dependencies,
            dependents,
            positions: nodes.iter().map(|n| n.position()).collect(),
        }
    }

    /// The number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    /// Whether node `i` directly depends on node `j` (`j` must execute before `i`).
    #[cfg(test)]
    pub fn depends_on(&self, i: usize, j: usize) -> bool {
        self.dependencies[i].contains(j)
    }

    /// A valid execution order of the nodes (dependencies first), or `None` if the graph has a
    /// cycle and cannot be linearized.
    ///
    /// Among the nodes ready at each step, the one with the smallest `(position, index)` is
    /// emitted first, so an edge-free graph reproduces the original program order.
    pub fn topological_order(&self) -> Option<Vec<usize>> {
        let n = self.len();
        let mut pending: Vec<usize> = self.dependencies.iter().map(SubGraph::len).collect();

        // Min-heap of the nodes ready to execute, keyed by `(position, index)`.
        let mut ready: BinaryHeap<Reverse<(usize, usize)>> = (0..n)
            .filter(|&i| pending[i] == 0)
            .map(|i| Reverse((self.positions[i], i)))
            .collect();

        let mut order = Vec::with_capacity(n);
        while let Some(Reverse((_, i))) = ready.pop() {
            order.push(i);
            for dependent in self.dependents[i].iter() {
                pending[dependent] -= 1;
                if pending[dependent] == 0 {
                    ready.push(Reverse((self.positions[dependent], dependent)));
                }
            }
        }

        // A node still pending at the end sits on a cycle.
        (order.len() == n).then_some(order)
    }

    /// Whether the graph can be linearized.
    pub fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }

    /// Compute the transitive [Reachability] of the graph.
    pub fn reachability(&self) -> Reachability {
        let n = self.len();
        let mut ancestors = self.dependencies.clone();

        // A node absorbs the (final) ancestor sets of its direct dependencies. In execution
        // order every dependency is processed before its dependents, so a single pass is
        // complete. Taking the set out and putting it back sidesteps the aliasing between
        // `ancestors[i]` and `ancestors[j]` without any clone (`j != i`: no self-edges).
        match self.topological_order() {
            Some(order) => {
                for i in order {
                    let mut acc = core::mem::take(&mut ancestors[i]);
                    for j in self.dependencies[i].iter() {
                        acc.union_with(&ancestors[j]);
                    }
                    ancestors[i] = acc;
                }
            }
            // On a cycle no linear pass exists (defensive — the merge guards keep the block set
            // acyclic): relax the direct edges to a fixpoint instead. Unions only grow, so a
            // length change is an exact change signal.
            None => loop {
                let mut changed = false;
                for i in 0..n {
                    let mut acc = core::mem::take(&mut ancestors[i]);
                    let before = acc.len();
                    for j in self.dependencies[i].iter() {
                        acc.union_with(&ancestors[j]);
                    }
                    changed |= acc.len() != before;
                    ancestors[i] = acc;
                }
                if !changed {
                    break;
                }
            },
        }

        // Descendants are the transpose of ancestors.
        let mut descendants = vec![SubGraph::empty(); n];
        for (i, ancestors) in ancestors.iter().enumerate() {
            for j in ancestors.iter() {
                descendants[j].insert(i);
            }
        }

        Reachability {
            ancestors,
            descendants,
        }
    }
}

/// Transitive reachability over a [Dag].
///
/// Answers whether contracting a set of nodes into a single node keeps the graph acyclic, which
/// is the legality condition for merging fusion blocks that may depend on each other.
pub struct Reachability {
    /// `ancestors[i]` = nodes that must execute before node `i` (transitive).
    ancestors: Vec<SubGraph>,
    /// `descendants[i]` = nodes that must execute after node `i` (transitive).
    descendants: Vec<SubGraph>,
}

impl Reachability {
    /// Whether contracting the two subgraphs into a single node keeps the graph acyclic.
    ///
    /// Contraction is illegal iff some node outside `a ∪ b` is both an ancestor and a descendant
    /// of the union — it would have to execute both before and after the contracted node.
    ///
    /// Empty subgraphs are refused conservatively: they identify nodes the reachability was not
    /// built from.
    pub fn can_contract(&self, a: &SubGraph, b: &SubGraph) -> bool {
        if a.is_empty() || b.is_empty() {
            return false;
        }

        let mut union = a.clone();
        union.union_with(b);

        let mut ancestors = SubGraph::empty();
        let mut descendants = SubGraph::empty();
        for i in union.iter() {
            ancestors.union_with(&self.ancestors[i]);
            descendants.union_with(&self.descendants[i]);
        }

        let mut between = ancestors;
        between.intersect_with(&descendants);
        between.subtract(&union);
        between.is_empty()
    }
}

/// Whether `node` must execute after `other`: read-after-write (`node` reads a resource `other`
/// produces) or write-after-read (`node` frees a resource `other` reads).
///
/// Works directly on the nodes' own data-flow sets through the [GraphNode] membership queries —
/// no intermediate collection.
fn depends_on<N: GraphNode>(node: &N, other: &N) -> bool {
    node.read().any(|r| other.produces(r)) || node.freed().any(|r| other.reads(r))
}
