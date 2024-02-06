use super::Operation;
use serde::{Deserialize, Serialize};

/// A body is composed of a list of [operators](Operator).
///
/// Note that the body assumes that the kernel will run on a 2D grid defined by the workgroup size
/// X and Y, but with Z=1.
#[derive(Debug, Clone, Serialize, Deserialize, new)]
pub struct Body {
    operators: Vec<Operation>,
}
