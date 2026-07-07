use super::{GraphNode, SubGraph};
use std::collections::HashSet;

/// A dependency graph over a fixed list of nodes, with edges derived from the nodes' data-flow
/// sets (see [GraphNode]).
///
/// The graph is not guaranteed to be acyclic — a cycle means the nodes cannot be linearized and
/// is reported by [topological_order](Self::topological_order) returning `None`.
pub struct Dag {
    /// `dependencies[i]` = nodes that must execute before node `i` (direct edges only).
    dependencies: Vec<SubGraph>,
    /// Tie-break position of each node (see [GraphNode::position]).
    positions: Vec<usize>,
}

impl Dag {
    /// Build the dependency graph of the given nodes.
    pub fn new<N: GraphNode>(nodes: &[N]) -> Self {
        let io = nodes.iter().map(NodeIo::new).collect::<Vec<_>>();

        let dependencies = io
            .iter()
            .map(|node| {
                let mut deps = SubGraph::empty();
                for (j, other) in io.iter().enumerate() {
                    if !core::ptr::eq(node, other) && node.depends_on(other) {
                        deps.insert(j);
                    }
                }
                deps
            })
            .collect();

        Self {
            dependencies,
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
        let mut emitted = SubGraph::empty();
        let mut order = Vec::with_capacity(n);

        for _ in 0..n {
            let next = (0..n)
                .filter(|&i| !emitted.contains(i))
                .filter(|&i| {
                    let mut waiting = self.dependencies[i].clone();
                    waiting.subtract(&emitted);
                    waiting.is_empty()
                })
                .min_by_key(|&i| (self.positions[i], i))?;

            emitted.insert(next);
            order.push(next);
        }

        Some(order)
    }

    /// Whether the graph can be linearized.
    pub fn is_acyclic(&self) -> bool {
        self.topological_order().is_some()
    }

    /// Compute the transitive [Reachability] of the graph.
    pub fn reachability(&self) -> Reachability {
        let n = self.len();
        let mut ancestors = self.dependencies.clone();

        // Propagate to a fixpoint: every node absorbs the ancestors of its ancestors.
        loop {
            let mut changed = false;
            for i in 0..n {
                let mut absorbed = ancestors[i].clone();
                for j in ancestors[i].iter().collect::<Vec<_>>() {
                    absorbed.union_with(&ancestors[j]);
                }
                if absorbed != ancestors[i] {
                    ancestors[i] = absorbed;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
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

/// The data-flow sets of a node, collected once so edge derivation is set intersections.
struct NodeIo<R> {
    produced: HashSet<R>,
    read: HashSet<R>,
    freed: HashSet<R>,
}

impl<R: Copy + Eq + core::hash::Hash> NodeIo<R> {
    fn new<N: GraphNode<Resource = R>>(node: &N) -> Self {
        Self {
            produced: node.produced().collect(),
            read: node.read().collect(),
            freed: node.freed().collect(),
        }
    }

    /// Whether `self` must execute after `other`: read-after-write (`self` reads a resource
    /// `other` produces) or write-after-read (`self` frees a resource `other` reads).
    fn depends_on(&self, other: &Self) -> bool {
        intersects(&self.read, &other.produced) || intersects(&self.freed, &other.read)
    }
}

fn intersects<R: Eq + core::hash::Hash>(a: &HashSet<R>, b: &HashSet<R>) -> bool {
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().any(|r| large.contains(r))
}
