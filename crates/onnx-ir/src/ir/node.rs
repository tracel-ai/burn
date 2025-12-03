//! ONNX node representation
//!
//! This module contains types for representing ONNX nodes, including their types,
//! configuration, inputs, outputs, and attributes.

use strum::{Display, EnumString};

use super::argument::Argument;
use super::attribute::Attributes;

// ============================================================================
// RawNode - Intermediate representation from ONNX parsing
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
pub(crate) struct RawNode {
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

use crate::node::*;

/// Macro to define both NodeType and Node enums from a single source
macro_rules! define_node_enum {
    (
        $(
            $(#[$variant_meta:meta])*
            $variant:ident => $node_type:ty
        ),* $(,)?
    ) => {
        /// Supported ONNX operators (plus Burn-specific extensions for dimensional mapping)
        ///
        /// See: <https://onnx.ai/onnx/operators/index.html>
        ///
        /// Note: Some operators have dimensional variants (e.g., Conv1d, Conv2d, Conv3d) that are
        /// Burn-specific extensions for better type safety and code generation.
        #[derive(Debug, Hash, Eq, PartialEq, EnumString, Clone, Display)]
        #[strum(ascii_case_insensitive)]
        pub enum NodeType {
            $(
                $(#[$variant_meta])*
                $variant,
            )*
        }

        /// Enum-based node representation
        ///
        /// Each ONNX operation is represented as a separate enum variant containing
        /// the operation-specific node struct.
        #[derive(Debug, Clone)]
        pub enum Node {
            $(
                $(#[$variant_meta])*
                $variant($node_type),
            )*
        }

        impl Node {
            /// Get the node name
            pub fn name(&self) -> &str {
                match self {
                    $(
                        Node::$variant(inner) => &inner.name,
                    )*
                }
            }

            /// Get the node inputs
            pub fn inputs(&self) -> &[Argument] {
                match self {
                    $(
                        Node::$variant(inner) => &inner.inputs,
                    )*
                }
            }

            /// Get mutable node inputs (internal use only)
            pub(crate) fn inputs_mut(&mut self) -> &mut Vec<Argument> {
                match self {
                    $(
                        Node::$variant(inner) => &mut inner.inputs,
                    )*
                }
            }

            /// Get the node outputs
            pub fn outputs(&self) -> &[Argument] {
                match self {
                    $(
                        Node::$variant(inner) => &inner.outputs,
                    )*
                }
            }

            /// Get mutable node outputs (internal use only)
            pub(crate) fn outputs_mut(&mut self) -> &mut Vec<Argument> {
                match self {
                    $(
                        Node::$variant(inner) => &mut inner.outputs,
                    )*
                }
            }
        }
    };
}

define_node_enum! {
    // ARITHMETIC & BASIC OPERATIONS
    Add => arithmetic::AddNode,
    Sub => arithmetic::SubNode,
    Mul => arithmetic::MulNode,
    Div => arithmetic::DivNode,
    Neg => neg::NegNode,
    Abs => abs::AbsNode,
    Pow => pow::PowNode,
    Reciprocal => reciprocal::ReciprocalNode,
    Sqrt => sqrt::SqrtNode,
    Exp => exp::ExpNode,
    Log => log::LogNode,
    Ceil => ceil::CeilNode,
    Floor => floor::FloorNode,
    Round => round::RoundNode,
    Sign => sign::SignNode,
    Erf => erf::ErfNode,

    // TRIGONOMETRIC OPERATIONS
    Sin => sin::SinNode,
    Cos => cos::CosNode,
    Tan => tan::TanNode,
    Asin => elementwise::ElementwiseUnaryNode,
    Acos => elementwise::ElementwiseUnaryNode,
    Atan => elementwise::ElementwiseUnaryNode,
    Sinh => sinh::SinhNode,
    Cosh => cosh::CoshNode,
    Tanh => tanh::TanhNode,
    Asinh => elementwise::ElementwiseUnaryNode,
    Acosh => elementwise::ElementwiseUnaryNode,
    Atanh => elementwise::ElementwiseUnaryNode,

    // ACTIVATION FUNCTIONS
    Relu => relu::ReluNode,
    Sigmoid => sigmoid::SigmoidNode,
    Softmax => softmax::SoftmaxNode,
    LogSoftmax => log_softmax::LogSoftmaxNode,
    LeakyRelu => leaky_relu::LeakyReluNode,
    HardSigmoid => hard_sigmoid::HardSigmoidNode,
    Elu => elementwise::ElementwiseUnaryNode,
    Selu => elementwise::ElementwiseUnaryNode,
    Celu => elementwise::ElementwiseUnaryNode,
    Gelu => gelu::GeluNode,
    Mish => elementwise::ElementwiseUnaryNode,
    Softplus => elementwise::ElementwiseUnaryNode,
    Softsign => elementwise::ElementwiseUnaryNode,
    ThresholdedRelu => elementwise::ElementwiseUnaryNode,
    HardSwish => elementwise::ElementwiseUnaryNode,
    PRelu => prelu::PReluNode,

    // COMPARISON & LOGICAL OPERATIONS
    Equal => comparison::EqualNode,
    Greater => comparison::GreaterNode,
    GreaterOrEqual => comparison::GreaterOrEqualNode,
    Less => comparison::LessNode,
    LessOrEqual => comparison::LessOrEqualNode,
    And => and::AndNode,
    Or => or::OrNode,
    Xor => xor::XorNode,
    Not => not::NotNode,
    Where => where_op::WhereNode,

    // BITWISE OPERATIONS
    BitwiseAnd => bitwiseand::BitwiseAndNode,
    BitwiseOr => bitwiseor::BitwiseOrNode,
    BitwiseXor => bitwisexor::BitwiseXorNode,
    BitwiseNot => bitwisenot::BitwiseNotNode,
    BitShift => bitshift::BitShiftNode,

    // REDUCTION OPERATIONS
    ArgMax => argmax::ArgMaxNode,
    ArgMin => argmin::ArgMinNode,
    ReduceMax => reduce::ReduceMaxNode,
    ReduceMin => reduce::ReduceMinNode,
    ReduceMean => reduce::ReduceMeanNode,
    ReduceSum => reduce::ReduceSumNode,
    ReduceProd => reduce::ReduceProdNode,
    ReduceL1 => reduce::ReduceL1Node,
    ReduceL2 => reduce::ReduceL2Node,
    ReduceLogSum => reduce::ReduceLogSumNode,
    ReduceLogSumExp => reduce::ReduceLogSumExpNode,
    ReduceSumSquare => reduce::ReduceSumSquareNode,

    // AGGREGATION OPERATIONS
    Max => max::MaxNode,
    Min => min::MinNode,
    Mean => mean::MeanNode,
    Sum => sum::SumNode,

    // TENSOR MANIPULATION
    Cast => cast::CastNode,
    Clip => clip::ClipNode,
    Concat => concat::ConcatNode,
    Expand => expand::ExpandNode,
    Flatten => flatten::FlattenNode,
    Gather => gather::GatherNode,
    GatherElements => gather_elements::GatherElementsNode,
    GatherND => unsupported::GatherNDNode,
    Pad => pad::PadNode,
    Reshape => reshape::ReshapeNode,
    Resize => resize::ResizeNode,
    Scatter => unsupported::ScatterNode,
    ScatterElements => unsupported::ScatterElementsNode,
    ScatterND => unsupported::ScatterNDNode,
    Shape => shape::ShapeNode,
    Size => size::SizeNode,
    Slice => slice::SliceNode,
    Split => split::SplitNode,
    Squeeze => squeeze::SqueezeNode,
    Tile => tile::TileNode,
    Transpose => transpose::TransposeNode,
    Unsqueeze => unsqueeze::UnsqueezeNode,
    DepthToSpace => depth_to_space::DepthToSpaceNode,
    SpaceToDepth => space_to_depth::SpaceToDepthNode,

    // MATRIX OPERATIONS
    MatMul => matmul::MatMulNode,
    MatMulInteger => matmulinteger::MatMulIntegerNode,
    Gemm => gemm::GemmNode,

    // CONVOLUTION & POOLING
    Conv1d => conv1d::Conv1dNode,
    Conv2d => conv2d::Conv2dNode,
    Conv3d => conv3d::Conv3dNode,
    ConvTranspose1d => conv_transpose1d::ConvTranspose1dNode,
    ConvTranspose2d => conv_transpose2d::ConvTranspose2dNode,
    ConvTranspose3d => conv_transpose3d::ConvTranspose3dNode,
    AveragePool1d => avg_pool1d::AveragePool1dNode,
    AveragePool2d => avg_pool2d::AveragePool2dNode,
    MaxPool1d => max_pool1d::MaxPool1dNode,
    MaxPool2d => max_pool2d::MaxPool2dNode,
    GlobalAveragePool => global_avg_pool::GlobalAveragePoolNode,
    GlobalMaxPool => unsupported::GlobalMaxPoolNode,

    // NORMALIZATION
    BatchNormalization => batch_norm::BatchNormalizationNode,
    InstanceNormalization => instance_norm::InstanceNormalizationNode,
    LayerNormalization => layer_norm::LayerNormalizationNode,
    GroupNormalization => group_norm::GroupNormalizationNode,

    // DROPOUT & REGULARIZATION
    Dropout => dropout::DropoutNode,

    // LINEAR & SPECIAL LAYERS
    Linear => linear::LinearNode,
    Attention => attention::AttentionNode,

    // CONSTANT GENERATION
    Constant => constant::ConstantNode,
    ConstantOfShape => constant_of_shape::ConstantOfShapeNode,
    EyeLike => eye_like::EyeLikeNode,
    Identity => identity::IdentityNode,

    // RANDOM OPERATIONS
    RandomNormal => random::RandomNormalNode,
    RandomUniform => random::RandomUniformNode,
    RandomNormalLike => random_like::RandomNormalLikeNode,
    RandomUniformLike => random_like::RandomUniformLikeNode,
    Bernoulli => bernoulli::BernoulliNode,

    // RANGE & SEQUENCE OPERATIONS
    Range => range::RangeNode,
    OneHot => one_hot::OneHotNode,

    // CONTROL FLOW
    If => if_node::IfNode,
    Loop => loop_node::LoopNode,
    Scan => scan_node::ScanNode,

    // SPECIAL OPERATIONS
    IsInf => is_inf::IsInfNode,
    IsNaN => is_nan::IsNaNNode,
    NonZero => nonzero::NonZeroNode,
    TopK => topk::TopKNode,
    Unique => unsupported::UniqueNode,
    Trilu => trilu::TriluNode,
    Mod => modulo::ModNode,
    CumSum => unsupported::CumSumNode,

    // UNSUPPORTED / PLACEHOLDER OPERATIONS (not yet implemented in burn-import)
    AffineGrid => unsupported::AffineGridNode,
    AveragePool => unsupported::AveragePoolNode,
    BlackmanWindow => unsupported::BlackmanWindowNode,
    CastLike => unsupported::CastLikeNode,
    CenterCropPad => unsupported::CenterCropPadNode,
    Col2Im => unsupported::Col2ImNode,
    Compress => unsupported::CompressNode,
    ConcatFromSequence => unsupported::ConcatFromSequenceNode,
    Conv => unsupported::ConvNode,
    ConvInteger => unsupported::ConvIntegerNode,
    ConvTranspose => unsupported::ConvTransposeNode,
    Dft => unsupported::DftNode,
    DeformConv => unsupported::DeformConvNode,
    DequantizeLinear => unsupported::DequantizeLinearNode,
    Det => unsupported::DetNode,
    DynamicQuantizeLinear => unsupported::DynamicQuantizeLinearNode,
    Einsum => unsupported::EinsumNode,
    GridSample => grid_sample::GridSampleNode,
    Gru => unsupported::GruNode,
    HammingWindow => unsupported::HammingWindowNode,
    HannWindow => unsupported::HannWindowNode,
    Hardmax => unsupported::HardmaxNode,
    Im => unsupported::ImNode,
    ImageDecoder => unsupported::ImageDecoderNode,
    LpNormalization => unsupported::LpNormalizationNode,
    LpPool => unsupported::LpPoolNode,
    Lrn => unsupported::LrnNode,
    Lstm => lstm::LstmNode,
    MaxPool => unsupported::MaxPoolNode,
    MaxRoiPool => unsupported::MaxRoiPoolNode,
    MaxUnpool => unsupported::MaxUnpoolNode,
    MeanVarianceNormalization => unsupported::MeanVarianceNormalizationNode,
    MelWeightMatrix => unsupported::MelWeightMatrixNode,
    Multinomial => unsupported::MultinomialNode,
    NegativeLogLikelihoodLoss => unsupported::NegativeLogLikelihoodLossNode,
    NonMaxSuppression => unsupported::NonMaxSuppressionNode,
    Optional => unsupported::OptionalNode,
    OptionalGetElement => unsupported::OptionalGetElementNode,
    OptionalHasElement => unsupported::OptionalHasElementNode,
    QLinearConv => unsupported::QLinearConvNode,
    QLinearMatMul => unsupported::QLinearMatMulNode,
    QuantizeLinear => unsupported::QuantizeLinearNode,
    RMSNormalization => unsupported::RMSNormalizationNode,
    Rnn => unsupported::RnnNode,
    RegexFullMatch => unsupported::RegexFullMatchNode,
    ReverseSequence => unsupported::ReverseSequenceNode,
    RoiAlign => unsupported::RoiAlignNode,
    RotaryEmbedding => unsupported::RotaryEmbeddingNode,
    SequenceAt => unsupported::SequenceAtNode,
    SequenceConstruct => unsupported::SequenceConstructNode,
    SequenceEmpty => unsupported::SequenceEmptyNode,
    SequenceErase => unsupported::SequenceEraseNode,
    SequenceInsert => unsupported::SequenceInsertNode,
    SequenceLength => unsupported::SequenceLengthNode,
    SequenceMap => unsupported::SequenceMapNode,
    Shrink => unsupported::ShrinkNode,
    SoftmaxCrossEntropyLoss => unsupported::SoftmaxCrossEntropyLossNode,
    SplitToSequence => unsupported::SplitToSequenceNode,
    Stft => unsupported::StftNode,
    StringConcat => unsupported::StringConcatNode,
    StringNormalizer => unsupported::StringNormalizerNode,
    StringSplit => unsupported::StringSplitNode,
    Swish => unsupported::SwishNode,
    TensorScatter => unsupported::TensorScatterNode,
    TfIdfVectorizer => unsupported::TfIdfVectorizerNode,
    Upsample => unsupported::UpsampleNode,
}
