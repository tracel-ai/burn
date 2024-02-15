use super::{Elem, Item, Scope, Variable};
use serde::{Deserialize, Serialize};

/// All operations that can be used in a GPU compute shader.
///
/// Notes:
///
/// [Operator] and [Algorithm] can be vectorized, but [Metadata] and [Loop] can't.
/// Therefore, during tracing, only operators and algorithms can be registered, and during the
/// compilation phase, the algorithm will be expanded.
///
/// Algorithm expansions can safely use [Metadata] and [Loop] operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // Some variants might not be used with different flags
pub enum Operation {
    Operator(Operator),
    Metadata(Metadata),
    Algorithm(Algorithm),
    Loop(Loop),
}

/// All operator that can be used in a GPU compute shader.
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
    Modulo(BinaryOperator),
    Index(BinaryOperator),
}

/// Tensor operations that can't be executed with a simple [operator](Operator) should use an
/// algorithm.
///
/// Algorithms can be expanded to basic [operator](Operator) during compilation, but after
/// vectorization, since for loops and other construct can't simply be vectorized. This also gives
/// the vectorization state to the expansion function, which may create a different set of
/// [operator](Operator) depending on the vectorization state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Algorithm {
    /// Read an input array with the given layout.
    ///
    /// Crucial to read arrays that aren't contiguous and to perform correct broadcasting.
    ReadGlobalWithLayout(ReadGlobalWithLayoutAlgo),
    /// Read an input array.
    ReadGlobal(ReadGlobalAlgo),
}

/// Settings for the [Algorithm::ReadGlobalWithLayout] variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadGlobalWithLayoutAlgo {
    /// The array to be read.
    pub global: Variable,
    /// The layout to be used.
    pub layout: Variable,
    /// The output variable to write the result.
    pub out: Variable,
}

/// Settings for the [Algorithm::ReadGlobal] variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadGlobalAlgo {
    /// The array to be read.
    pub global: Variable,
    /// The output variable to write the result.
    pub out: Variable,
}

/// All loop variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Loop {
    /// A basic range loop.
    Range(RangeLoop),
}

/// All metadata that can be access in a shader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Metadata {
    /// The stride of an array at the given dimension.
    Stride {
        dim: Variable,
        var: Variable,
        out: Variable,
    },
    /// The shape of an array at the given dimension.
    Shape {
        dim: Variable,
        var: Variable,
        out: Variable,
    },
}

/// Settings for the [Loop::Range] variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeLoop {
    /// The loop index variable.
    pub i: Variable,
    /// The start value.
    pub start: Variable,
    /// The end value.
    pub end: Variable,
    /// The scope that contains all operations and variables declared in the loop body.
    pub scope: Scope,
}

impl RangeLoop {
    /// Registers a range loop to the given scope.
    pub fn register<F: Fn(&Variable, &mut Scope)>(
        parent_scope: &mut Scope,
        start: Variable,
        end: Variable,
        func: F,
    ) {
        let mut scope = parent_scope.child();
        let index_ty = Item::Scalar(Elem::UInt);
        let i = scope.create_local_undeclare(index_ty);

        func(&i, &mut scope);

        let op = Self {
            i,
            start,
            end,
            scope,
        };
        parent_scope.register(Loop::Range(op));
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

impl From<Operator> for Operation {
    fn from(val: Operator) -> Self {
        Operation::Operator(val)
    }
}

impl From<Metadata> for Operation {
    fn from(val: Metadata) -> Self {
        Operation::Metadata(val)
    }
}

impl From<Algorithm> for Operation {
    fn from(val: Algorithm) -> Self {
        Operation::Algorithm(val)
    }
}

impl From<Loop> for Operation {
    fn from(val: Loop) -> Self {
        Operation::Loop(val)
    }
}
