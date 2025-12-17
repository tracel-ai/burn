/// Implements NodeCodegen trait on onnx_ir::Node enum
/// Uses a simple macro to generate match arms for all supported nodes
use onnx_ir::{Argument, Node};
use proc_macro2::TokenStream;

use super::node_traits::NodeCodegen;
use crate::burn::{BurnImports, Field};
use burn_store::TensorSnapshot;

/// Macro to implement NodeCodegen on onnx_ir::Node by dispatching to individual node impls
macro_rules! impl_node_codegen_dispatch {
    ($($variant:ident),* $(,)?) => {
        impl NodeCodegen for Node {
            fn inputs(&self) -> &[Argument] {
                match self {
                    $(Node::$variant(n) => n.inputs(),)*
                    _ => panic!("Unsupported node type for inputs: {:?}", self),
                }
            }

            fn outputs(&self) -> &[Argument] {
                match self {
                    $(Node::$variant(n) => n.outputs(),)*
                    _ => panic!("Unsupported node type for outputs: {:?}", self),
                }
            }

            fn forward(&self, scope: &mut crate::burn::scope::ScopeAtPosition<'_>) -> TokenStream {
                match self {
                    $(Node::$variant(n) => n.forward(scope),)*
                    _ => panic!("Unsupported node type for forward: {:?}", self),
                }
            }

            fn field(&self) -> Option<Field> {
                match self {
                    $(Node::$variant(n) => n.field(),)*
                    _ => None,
                }
            }

            fn register_imports(&self, imports: &mut BurnImports) {
                match self {
                    $(Node::$variant(n) => n.register_imports(imports),)*
                    _ => {}
                }
            }

            fn collect_snapshots(&self, field_name: &str) -> Vec<TensorSnapshot> {
                match self {
                    $(Node::$variant(n) => n.collect_snapshots(field_name),)*
                    _ => vec![],
                }
            }
        }
    };
}

// List all supported node types here
// Just add/remove variant names as needed - one place to maintain!
impl_node_codegen_dispatch! {
    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    MatMul,

    // Comparison ops
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,

    // Boolean ops
    And,
    Or,
    Xor,

    // Unary ops
    Abs,
    Ceil,
    Cos,
    Cosh,
    Erf,
    Exp,
    Floor,
    Identity,
    Log,
    Neg,
    Not,
    Reciprocal,
    Round,
    Sigmoid,
    Sign,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,

    // Activation ops
    Relu,
    Gelu,
    LeakyRelu,
    HardSigmoid,
    Softmax,
    LogSoftmax,
    PRelu,

    // Shape ops
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,
    Transpose,
    Shape,
    Size,

    // Tensor ops
    Concat,
    Split,
    Slice,
    Gather,
    GatherElements,
    Tile,
    Expand,
    Pad,

    // Convolution ops
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,

    // Pooling ops
    AveragePool1d,
    AveragePool2d,
    MaxPool1d,
    MaxPool2d,
    GlobalAveragePool,

    // Normalization ops
    BatchNormalization,
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization,

    // Other ops
    Cast,
    Clip,
    CumSum,
    Dropout,
    Where,
    ArgMax,
    ArgMin,
    TopK,
    NonZero,
    OneHot,
    Pow,
    Mod,
    Trilu,

    // Bitwise ops
    BitShift,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,

    // Math ops
    Sum,
    Mean,
    Gemm,
    Linear,
    MatMulInteger,

    // Constant ops
    Constant,
    ConstantOfShape,
    EyeLike,
    Range,

    // Random ops
    RandomNormal,
    RandomUniform,
    RandomNormalLike,
    RandomUniformLike,
    Bernoulli,

    // Spatial ops
    DepthToSpace,
    SpaceToDepth,
    Resize,
    GridSample,

    // Test ops
    IsInf,
    IsNaN,

    // Special ops
    Attention,

    // Control flow ops
    If,
    Loop,
    Scan,

    // Recurrent neural network ops
    Lstm,

    // Reduce ops (handled by ReduceNode in onnx-ir)
    ReduceMax,
    ReduceMin,
    ReduceMean,
    ReduceProd,
    ReduceSum,
    ReduceSumSquare,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
}
