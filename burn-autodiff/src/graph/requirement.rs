use super::ops::MetadataRef;

#[derive(Debug, Clone, Copy)]
pub enum Requirement {
    Grad,
    GradInBackward,
    None,
}

impl Requirement {
    pub fn infer(&self, other: &Self) -> Self {
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

    pub fn is_none(&self) -> bool {
        match self {
            Requirement::None => true,
            _ => false,
        }
    }

    pub fn from_metadata(metadata: &[MetadataRef]) -> Self {
        metadata
            .iter()
            .map(|metadata| metadata.requirement)
            .reduce(|acc, requirement| requirement.infer(&acc))
            .unwrap_or(Requirement::None)
    }
}
