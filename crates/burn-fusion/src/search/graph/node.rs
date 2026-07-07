use core::hash::Hash;

/// Data-flow description of a node in a dependency graph.
///
/// A node reads and produces resources (tensors, buffers, …) identified by
/// [Resource](Self::Resource). Dependencies between nodes are never declared explicitly: they are
/// derived from the resource sets by [Dag::new](super::Dag::new) using two hazard rules.
///
/// - **Read-after-write**: a node reading a resource depends on the node producing it.
/// - **Write-after-read**: a node freeing a resource depends on every other node reading it —
///   executing the freeing node first would release the resource out from under the readers.
pub trait GraphNode {
    /// Identifies a resource read, produced, or freed by a node.
    type Resource: Copy + Eq + Hash;

    /// The resources produced by this node.
    fn produced(&self) -> impl Iterator<Item = Self::Resource>;

    /// The resources this node reads that are produced by other nodes.
    fn read(&self) -> impl Iterator<Item = Self::Resource>;

    /// The resources this node reads for the last time, releasing the underlying storage
    /// (in-place reuse or deallocation).
    fn freed(&self) -> impl Iterator<Item = Self::Resource>;

    /// Whether this node produces the resource — the membership form of
    /// [produced](Self::produced). Nodes that already hold their resources in a set should
    /// override this with a direct lookup.
    fn produces(&self, resource: Self::Resource) -> bool {
        self.produced().any(|r| r == resource)
    }

    /// Whether this node reads the resource — the membership form of [read](Self::read).
    fn reads(&self, resource: Self::Resource) -> bool {
        self.read().any(|r| r == resource)
    }

    /// The position of the node in the original program order, used to break ties between
    /// independent nodes when ordering them.
    fn position(&self) -> usize;
}

impl<N: GraphNode> GraphNode for &N {
    type Resource = N::Resource;

    fn produced(&self) -> impl Iterator<Item = Self::Resource> {
        (*self).produced()
    }

    fn read(&self) -> impl Iterator<Item = Self::Resource> {
        (*self).read()
    }

    fn freed(&self) -> impl Iterator<Item = Self::Resource> {
        (*self).freed()
    }

    fn produces(&self, resource: Self::Resource) -> bool {
        (*self).produces(resource)
    }

    fn reads(&self, resource: Self::Resource) -> bool {
        (*self).reads(resource)
    }

    fn position(&self) -> usize {
        (*self).position()
    }
}
