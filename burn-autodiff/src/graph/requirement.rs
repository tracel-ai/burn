use super::NodeRef;

/// Requirement enum for operations.
#[derive(Debug, Clone, Copy)]
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
        match self {
            Requirement::None => true,
            _ => false,
        }
    }

    /// Returns the right requirement from a list of nodes.
    pub fn from_nodes(nodes: &[NodeRef]) -> Self {
        nodes
            .iter()
            .map(|node| node.requirement)
            .reduce(|acc, requirement| requirement.infer(&acc))
            .unwrap_or(Requirement::None)
    }

    fn infer(&self, other: &Self) -> Self {
        match self {
            Self::Grad => return Self::GradInBackward,
            Self::GradInBackward => return Self::GradInBackward,
            Self::None => (),
        }

        match other {
            Self::Grad => Self::GradInBackward,
            Self::GradInBackward => Self::GradInBackward,
            Self::None => Self::None,
        }
    }
}
