use super::{Elem, Item, Scope, Variable};
use serde::{Deserialize, Serialize};

/// All operations that can be used in a GPU compute shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Operation {
    Operator(Operator),
    Metadata(Metadata),
    Algorithm(Algorithm),
    Loop(Loop),
}

/// All operations that can be used in a GPU compute shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Operator {
    Add(BinaryOperator),
    Sub(BinaryOperator),
    Mul(BinaryOperator),
    Div(BinaryOperator),
    Abs(UnaryOperator),
    Exp(UnaryOperator),
    Log(UnaryOperator),
    Log1p(UnaryOperator),
    Cos(UnaryOperator),
    Sin(UnaryOperator),
    Tanh(UnaryOperator),
    Powf(BinaryOperator),
    Sqrt(UnaryOperator),
    Erf(UnaryOperator),
    Recip(UnaryOperator),
    Equal(BinaryOperator),
    Lower(BinaryOperator),
    Clamp(ClampOperator),
    Greater(BinaryOperator),
    LowerEqual(BinaryOperator),
    GreaterEqual(BinaryOperator),
    ConditionalAssign(ConditionalAssignOperator),
    AssignGlobal(UnaryOperator),
    AssignLocal(UnaryOperator),
    ReadGlobal(ReadGlobalOperator),
    ReadGlobalWithLayout(ReadGlobalWithLayoutOperator),
    Modulo(BinaryOperator),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Algorithm {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Loop {
    Range(RangeLoop),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Metadata {
    Rank {
        out: Variable,
    },
    Stride {
        dim: Variable,
        var: Variable,
        out: Variable,
    },
    Shape {
        dim: Variable,
        var: Variable,
        out: Variable,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeLoop {
    pub i: Variable,
    pub start: Variable,
    pub end: Variable,
    pub scope: Scope,
}

impl RangeLoop {
    pub fn new<F: Fn(&Variable, &mut Scope)>(
        parent_scope: &mut Scope,
        start: Variable,
        end: Variable,
        func: F,
    ) -> Self {
        let mut scope = Scope::empty(parent_scope.prefix.clone() + "_loop");
        let index_ty = Item::Scalar(Elem::UInt);
        let i = scope.create_local(index_ty);

        func(&i, &mut scope);

        Self {
            i,
            start,
            end,
            scope,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryOperator {
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnaryOperator {
    pub input: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClampOperator {
    pub input: Variable,
    pub min_value: Variable,
    pub max_value: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalAssignOperator {
    pub cond: Variable,
    pub lhs: Variable,
    pub rhs: Variable,
    pub out: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadGlobalOperator {
    pub variable: Variable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadGlobalWithLayoutOperator {
    pub variable: Variable,
    pub tensor_read_pos: usize,
    pub tensor_layout_pos: usize,
}

impl Into<Operation> for Operator {
    fn into(self) -> Operation {
        Operation::Operator(self)
    }
}

impl Into<Operation> for Metadata {
    fn into(self) -> Operation {
        Operation::Metadata(self)
    }
}

impl Into<Operation> for Algorithm {
    fn into(self) -> Operation {
        Operation::Algorithm(self)
    }
}

impl Into<Operation> for Loop {
    fn into(self) -> Operation {
        Operation::Loop(self)
    }
}
