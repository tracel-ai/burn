//! ONNX node representation
//!
//! This module contains types for representing ONNX nodes, including their types,
//! configuration, inputs, outputs, and attributes.

use strum::{Display, EnumString};

use super::argument::Argument;
use super::attribute::Attributes;

// ============================================================================
// NodeBuilder - Intermediate representation from ONNX parsing
// ============================================================================

/// Reference to a runtime input by name and index.
/// Used in configs to point to node inputs instead of storing stale copies.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuntimeInputRef {
    /// Name of the input argument
    pub name: String,
    /// Index in the node's inputs array
    pub input_index: usize,
}

impl RuntimeInputRef {
    pub fn new(name: String, input_index: usize) -> Self {
        Self { name, input_index }
    }
}

/// Nodes produced by the ONNX parser
#[derive(Clone, Debug)]
pub(crate) struct NodeBuilder {
    /// The type of the node.
    /// This should be a valid ONNX operator.
    pub node_type: NodeType,

    /// The name of the node.
    pub name: String,

    /// The inputs of the node.
    pub inputs: Vec<Argument>,

    /// The outputs of the node.
    pub outputs: Vec<Argument>,

    /// ONNX attributes (opset-specific parameters)
    pub(crate) attrs: Attributes,
}

// ============================================================================
// Node enum - Type-safe representation with operation-specific config
// ============================================================================

/// Macro to define both NodeType and Node enums from a single source
///
/// This macro takes a list of variants and generates:
/// 1. NodeType enum - simple enum of all operation types (for parsing/registry)
/// 2. Node enum - full enum with name, inputs, outputs, and optional config
/// 3. Accessor methods (name(), inputs(), outputs()) that work for all Node variants
///
/// Usage:
/// ```ignore
/// define_node_enum! {
///     VariantName,              // No config
///     VariantName { config: ConfigType },  // With config
/// }
/// ```
macro_rules! define_node_enum {
    (
        $(
            $(#[$meta:meta])*
            $variant:ident $({ $($field:ident: $type:ty),* $(,)? })?
        ),* $(,)?
    ) => {
        /// Supported ONNX operators (plus Burn-specific extensions for dimensional mapping)
        ///
        /// See: <https://onnx.ai/onnx/operators/index.html>
        ///
        /// Note: Some operators have dimensional variants (e.g., Conv1d, Conv2d, Conv3d) that are
        /// Burn-specific extensions for better type safety and code generation.
        ///
        /// This enum is automatically generated from the Node enum definition to ensure
        /// they stay in sync.
        #[derive(Debug, Hash, Eq, PartialEq, EnumString, Clone, Display)]
        pub enum NodeType {
            $(
                $(#[$meta])*
                $variant,
            )*
        }

        /// Enum-based node representation
        ///
        /// Each ONNX operation is represented as a separate enum variant containing
        /// the operation-specific configuration.
        #[derive(Debug, Clone)]
        pub enum Node {
            $(
                $(#[$meta])*
                $variant {
                    name: String,
                    inputs: Vec<Argument>,
                    outputs: Vec<Argument>,
                    $($($field: $type),*)?
                },
            )*
        }

        impl Node {
            /// Get the node name
            pub fn name(&self) -> &str {
                match self {
                    $(Node::$variant { name, .. })|* => name,
                }
            }

            /// Get the node inputs
            pub fn inputs(&self) -> &[Argument] {
                match self {
                    $(Node::$variant { inputs, .. })|* => inputs,
                }
            }

            /// Get the node outputs
            pub fn outputs(&self) -> &[Argument] {
                match self {
                    $(Node::$variant { outputs, .. })|* => outputs,
                }
            }
        }
    };
}

use crate::node::*;

define_node_enum! {

    // ARITHMETIC & BASIC OPERATIONS (no config)
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Abs,
    Pow,
    Reciprocal,
    Sqrt,
    Exp,
    Log,
    Ceil,
    Floor,
    Round,
    Sign,
    Erf,

    // TRIGONOMETRIC OPERATIONS (no config)
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,

    // ACTIVATION FUNCTIONS
    Relu,
    Sigmoid,
    Softmax { config: softmax::SoftmaxConfig },
    LogSoftmax { config: log_softmax::LogSoftmaxConfig },
    LeakyRelu { config: leaky_relu::LeakyReluConfig },
    HardSigmoid { config: hard_sigmoid::HardSigmoidConfig },
    Elu,
    Selu,
    Celu,
    Gelu,
    Mish,
    Softplus,
    Softsign,
    ThresholdedRelu,
    HardSwish,
    PRelu,

    // COMPARISON & LOGICAL OPERATIONS (no config)
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    And,
    Or,
    Xor,
    Not,
    Where,

    // BITWISE OPERATIONS
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    BitShift { config: bitshift::BitShiftConfig },

    // REDUCTION OPERATIONS
    ArgMax { config: argmax::ArgMaxConfig },
    ArgMin { config: argmin::ArgMinConfig },
    ReduceMax { config: reduce::ReduceConfig },
    ReduceMin { config: reduce::ReduceConfig },
    ReduceMean { config: reduce::ReduceConfig },
    ReduceSum { config: reduce::ReduceConfig },
    ReduceProd { config: reduce::ReduceConfig },
    ReduceL1 { config: reduce::ReduceConfig },
    ReduceL2 { config: reduce::ReduceConfig },
    ReduceLogSum { config: reduce::ReduceConfig },
    ReduceLogSumExp { config: reduce::ReduceConfig },
    ReduceSumSquare { config: reduce::ReduceConfig },

    // AGGREGATION OPERATIONS (no config)
    Max,
    Min,
    Mean,
    Sum,

    // TENSOR MANIPULATION
    Cast { config: cast::CastConfig },
    Clip { config: clip::ClipConfig },
    Concat { config: concat::ConcatConfig },
    Expand { config: expand::ExpandConfig },
    Flatten { config: flatten::FlattenConfig },
    Gather { config: gather::GatherConfig },
    GatherElements { config: gather_elements::GatherElementsConfig },
    GatherND,
    Identity,
    Pad { config: pad::PadConfig },
    Reshape { config: reshape::ReshapeConfig },
    Resize { config: resize::ResizeConfig },
    Scatter,
    ScatterElements,
    ScatterND,
    Shape { config: shape::ShapeConfig },
    Size,
    Slice { config: slice::SliceConfig },
    Split { config: split::SplitConfig },
    Squeeze { config: squeeze::SqueezeConfig },
    Tile { config: tile::TileConfig },
    Transpose { config: transpose::TransposeConfig },
    Unsqueeze { config: unsqueeze::UnsqueezeConfig },
    DepthToSpace { config: depth_to_space::DepthToSpaceConfig },
    SpaceToDepth { config: space_to_depth::SpaceToDepthConfig },

    // MATRIX OPERATIONS
    MatMul,
    MatMulInteger,
    Gemm { config: gemm::GemmConfig },

    // CONVOLUTION & POOLING
    Conv1d { config: conv1d::Conv1dConfig },
    Conv2d { config: conv2d::Conv2dConfig },
    Conv3d { config: conv3d::Conv3dConfig },
    ConvTranspose1d { config: conv_transpose1d::ConvTranspose1dConfig },
    ConvTranspose2d { config: conv_transpose2d::ConvTranspose2dConfig },
    ConvTranspose3d { config: conv_transpose3d::ConvTranspose3dConfig },
    AveragePool1d { config: avg_pool1d::AvgPool1dConfig },
    AveragePool2d { config: avg_pool2d::AvgPool2dConfig },
    MaxPool1d { config: max_pool1d::MaxPool1dConfig },
    MaxPool2d { config: max_pool2d::MaxPool2dConfig },
    GlobalAveragePool,
    GlobalMaxPool,

    // NORMALIZATION
    BatchNormalization { config: batch_norm::BatchNormConfig },
    InstanceNormalization { config: instance_norm::InstanceNormConfig },
    LayerNormalization { config: layer_norm::LayerNormConfig },
    GroupNormalization { config: group_norm::GroupNormConfig },

    // DROPOUT & REGULARIZATION
    Dropout { config: dropout::DropoutConfig },

    // LINEAR & SPECIAL LAYERS
    Linear { config: linear::LinearConfig },
    Attention { config: attention::AttentionConfig },

    // CONSTANT GENERATION
    Constant,
    ConstantOfShape { config: constant_of_shape::ConstantOfShapeConfig },
    EyeLike { config: eye_like::EyeLikeConfig },

    // RANDOM OPERATIONS
    RandomNormal { config: random::RandomNormalConfig },
    RandomUniform { config: random::RandomUniformConfig },
    RandomNormalLike { config: random_like::RandomNormalLikeConfig },
    RandomUniformLike { config: random_like::RandomUniformLikeConfig },
    Bernoulli,

    // RANGE & SEQUENCE OPERATIONS
    Range { config: range::RangeConfig },
    OneHot { config: one_hot::OneHotConfig },

    // CONTROL FLOW
    If { config: if_node::IfConfig },
    Loop { config: loop_node::LoopConfig },
    Scan { config: scan_node::ScanConfig },

    // SPECIAL OPERATIONS
    IsInf { config: is_inf::IsInfConfig },
    IsNaN,
    NonZero,
    TopK { config: topk::TopKConfig },
    Unique,
    Trilu { config: trilu::TriluConfig },
    Mod { config: modulo::ModConfig },
    CumSum,

    // UNSUPPORTED / PLACEHOLDER OPERATIONS (not yet implemented in burn-import)
    // These are part of the ONNX spec but don't have full Node implementations yet
    AffineGrid,
    AveragePool,
    BlackmanWindow,
    CastLike,
    CenterCropPad,
    Col2Im,
    Compress,
    ConcatFromSequence,
    Conv,
    ConvInteger,
    ConvTranspose,
    Dft,
    DeformConv,
    DequantizeLinear,
    Det,
    DynamicQuantizeLinear,
    Einsum,
    GridSample,
    Gru,
    HammingWindow,
    HannWindow,
    Hardmax,
    Im,
    ImageDecoder,
    LpNormalization,
    LpPool,
    Lrn,
    Lstm,
    MaxPool,
    MaxRoiPool,
    MaxUnpool,
    MeanVarianceNormalization,
    MelWeightMatrix,
    Multinomial,
    NegativeLogLikelihoodLoss,
    NonMaxSuppression,
    Optional,
    OptionalGetElement,
    OptionalHasElement,
    QLinearConv,
    QLinearMatMul,
    QuantizeLinear,
    RMSNormalization,
    Rnn,
    RegexFullMatch,
    ReverseSequence,
    RoiAlign,
    RotaryEmbedding,
    SequenceAt,
    SequenceConstruct,
    SequenceEmpty,
    SequenceErase,
    SequenceInsert,
    SequenceLength,
    SequenceMap,
    Shrink,
    SoftmaxCrossEntropyLoss,
    SplitToSequence,
    Stft,
    StringConcat,
    StringNormalizer,
    StringSplit,
    Swish,
    TensorScatter,
    TfIdfVectorizer,
    Upsample,
}
