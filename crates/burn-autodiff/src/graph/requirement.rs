use super::NodeRef;

/// Requirement for each tensor in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// Operations that require gradients.
    Grad,
    /// Operations that require gradients only for backprop.
    GradInBackward,
    /// Operations that don't need gradients, therefore not to be included in the graph.
    None,
}

impl Requirement {
    /// Returns true if gradients are not required.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
    /// Returns the right requirement from a list of nodes.
    pub fn from_nodes(nodes: &[NodeRef]) -> Self {
        if nodes.len() == 1 {
            return nodes[0].requirement.infer(&Requirement::None);
        }

        nodes
            .iter()
            .map(|node| node.requirement)
            .reduce(|acc, requirement| requirement.infer(&acc))
            .unwrap_or(Requirement::None)
    }

    fn infer(&self, other: &Self) -> Self {
        match self.is_none() && other.is_none() {
            true => Self::None,
            false => Self::GradInBackward,
        }
    }
}
