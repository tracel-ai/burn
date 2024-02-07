use super::Variable;
use serde::{Deserialize, Serialize};

/// All operators that can be fused in a WGSL compute shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Operation {
    Add(BinaryOperation),
    Sub(BinaryOperation),
    Mul(BinaryOperation),
    Div(BinaryOperation),
    Abs(UnaryOperation),
    Exp(UnaryOperation),
    Log(UnaryOperation),
    Log1p(UnaryOperation),
    Cos(UnaryOperation),
    Sin(UnaryOperation),
    Tanh(UnaryOperation),
    Powf(BinaryOperation),
    Sqrt(UnaryOperation),
    Erf(UnaryOperation),
    Recip(UnaryOperation),
    Equal(BinaryOperation),
    Lower(BinaryOperation),
    Clamp(ClampOperation),
    Greater(BinaryOperation),
    LowerEqual(BinaryOperation),
    GreaterEqual(BinaryOperation),
    ConditionalAssign(ConditionalAssignOperation),
    AssignGlobal(UnaryOperation),
    AssignLocal(UnaryOperation),
    ReadGlobal(ReadGlobalOperation),
    ReadGlobalWithLayout(ReadGlobalWithLayoutOperation),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub struct BinaryOperation {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub struct UnaryOperation {
    pub input: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub struct ClampOperation {
    pub input: Variable,
    pub min_value: Variable,
    pub max_value: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub struct ConditionalAssignOperation {
    pub cond: Variable,
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub struct ReadGlobalOperation {
    pub variable: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub struct ReadGlobalWithLayoutOperation {
    pub variable: Variable,
    pub tensor_read_pos: usize,
    pub tensor_layout_pos: usize,
}
