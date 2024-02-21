use super::{ConditionalAssign, Matmul, ReadGlobal, ReadGlobalWithLayout, WriteGlobal};
use crate::codegen::dialect::gpu::Vectorization;
use serde::{Deserialize, Serialize};

/// Tensor operations that can't be executed with a simple [operator](super::super::Operator) should use a
/// procedure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Procedure {
    /// Read a global array with the given layout.
    ///
    /// Crucial to read arrays that aren't contiguous and to perform correct broadcasting.
    ReadGlobalWithLayout(ReadGlobalWithLayout),
    /// Read a global array.
    ReadGlobal(ReadGlobal),
    /// Matrix Multiplication procedure.
    Matmul(Matmul),
    /// Write to a global array.
    WriteGlobal(WriteGlobal),
    /// Assign value to a variable based on a given condition.
    ConditionalAssign(ConditionalAssign),
}

impl Procedure {
    pub fn vectorize(&self, vectorization: Vectorization) -> Self {
        match self {
            Procedure::ReadGlobalWithLayout(op) => {
                Procedure::ReadGlobalWithLayout(op.vectorize(vectorization))
            }
            Procedure::ReadGlobal(op) => Procedure::ReadGlobal(op.vectorize(vectorization)),
            Procedure::Matmul(op) => Procedure::Matmul(op.vectorize(vectorization)),
            Procedure::WriteGlobal(op) => Procedure::WriteGlobal(op.vectorize(vectorization)),
            Procedure::ConditionalAssign(proc) => {
                Procedure::ConditionalAssign(proc.vectorize(vectorization))
            }
        }
    }
}
