//! Enum-based node representation for ONNX operations
//!
//! This module contains the Node type which provides compile-time type safety
//! for ONNX operations by encoding the operation type and its configuration in enum variants.

use super::argument::Argument;

/// Macro to define the Node enum and generate accessor methods
///
/// This macro takes a list of variants and generates both:
/// 1. The Node enum with all variants having name, inputs, outputs (and optionally config)
/// 2. Accessor methods (name(), inputs(), outputs()) that work for all variants
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
use crate::node::argmax::ArgMaxConfig;
use crate::node::argmin::ArgMinConfig;
use crate::node::attention::AttentionConfig;
use crate::node::avg_pool1d::AvgPool1dConfig;
use crate::node::avg_pool2d::AvgPool2dConfig;
use crate::node::batch_norm::BatchNormConfig;
use crate::node::bitshift::BitShiftConfig;
use crate::node::cast::CastConfig;
use crate::node::clip::ClipConfig;
use crate::node::concat::ConcatConfig;
use crate::node::constant_of_shape::ConstantOfShapeConfig;
use crate::node::conv_transpose1d::ConvTranspose1dConfig;
use crate::node::conv_transpose2d::ConvTranspose2dConfig;
use crate::node::conv_transpose3d::ConvTranspose3dConfig;
use crate::node::conv1d::Conv1dConfig;
use crate::node::conv2d::Conv2dConfig;
use crate::node::conv3d::Conv3dConfig;
use crate::node::depth_to_space::DepthToSpaceConfig;
use crate::node::dropout::DropoutConfig;
use crate::node::expand::ExpandConfig;
use crate::node::eye_like::EyeLikeConfig;
use crate::node::flatten::FlattenConfig;
use crate::node::gather::GatherConfig;
use crate::node::gather_elements::GatherElementsConfig;
use crate::node::gemm::GemmConfig;
use crate::node::group_norm::GroupNormConfig;
use crate::node::hard_sigmoid::HardSigmoidConfig;
use crate::node::if_node::IfConfig;
use crate::node::instance_norm::InstanceNormConfig;
use crate::node::is_inf::IsInfConfig;
use crate::node::layer_norm::LayerNormConfig;
use crate::node::leaky_relu::LeakyReluConfig;
use crate::node::linear::LinearConfig;
use crate::node::log_softmax::LogSoftmaxConfig;
use crate::node::loop_node::LoopConfig;
use crate::node::max_pool1d::MaxPool1dConfig;
use crate::node::max_pool2d::MaxPool2dConfig;
use crate::node::modulo::ModConfig;
use crate::node::one_hot::OneHotConfig;
use crate::node::pad::PadConfig;
use crate::node::random::{RandomNormalConfig, RandomUniformConfig};
use crate::node::random_like::{RandomNormalLikeConfig, RandomUniformLikeConfig};
use crate::node::range::RangeConfig;
use crate::node::reduce::ReduceConfig;
use crate::node::reshape::ReshapeConfig;
use crate::node::resize::ResizeConfig;
use crate::node::scan_node::ScanConfig;
use crate::node::shape::ShapeConfig;
use crate::node::slice::SliceConfig;
use crate::node::softmax::SoftmaxConfig;
use crate::node::space_to_depth::SpaceToDepthConfig;
use crate::node::split::SplitConfig;
use crate::node::squeeze::SqueezeConfig;
use crate::node::tile::TileConfig;
use crate::node::topk::TopKConfig;
use crate::node::transpose::TransposeConfig;
use crate::node::trilu::TriluConfig;
use crate::node::unsqueeze::UnsqueezeConfig;

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
    Softmax { config: SoftmaxConfig },
    LogSoftmax { config: LogSoftmaxConfig },
    LeakyRelu { config: LeakyReluConfig },
    HardSigmoid { config: HardSigmoidConfig },
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
    BitShift { config: BitShiftConfig },

    // REDUCTION OPERATIONS
    ArgMax { config: ArgMaxConfig },
    ArgMin { config: ArgMinConfig },
    ReduceMax { config: ReduceConfig },
    ReduceMin { config: ReduceConfig },
    ReduceMean { config: ReduceConfig },
    ReduceSum { config: ReduceConfig },
    ReduceProd { config: ReduceConfig },
    ReduceL1 { config: ReduceConfig },
    ReduceL2 { config: ReduceConfig },
    ReduceLogSum { config: ReduceConfig },
    ReduceLogSumExp { config: ReduceConfig },
    ReduceSumSquare { config: ReduceConfig },

    // AGGREGATION OPERATIONS (no config)
    Max,
    Min,
    Mean,
    Sum,

    // TENSOR MANIPULATION
    Cast { config: CastConfig },
    Clip { config: ClipConfig },
    Concat { config: ConcatConfig },
    Expand { config: ExpandConfig },
    Flatten { config: FlattenConfig },
    Gather { config: GatherConfig },
    GatherElements { config: GatherElementsConfig },
    GatherND,
    Identity,
    Pad { config: PadConfig },
    Reshape { config: ReshapeConfig },
    Resize { config: ResizeConfig },
    Scatter,
    ScatterElements,
    ScatterND,
    Shape { config: ShapeConfig },
    Size,
    Slice { config: SliceConfig },
    Split { config: SplitConfig },
    Squeeze { config: SqueezeConfig },
    Tile { config: TileConfig },
    Transpose { config: TransposeConfig },
    Unsqueeze { config: UnsqueezeConfig },
    DepthToSpace { config: DepthToSpaceConfig },
    SpaceToDepth { config: SpaceToDepthConfig },

    // MATRIX OPERATIONS
    MatMul,
    MatMulInteger,
    Gemm { config: GemmConfig },

    // CONVOLUTION & POOLING
    Conv1d { config: Conv1dConfig },
    Conv2d { config: Conv2dConfig },
    Conv3d { config: Conv3dConfig },
    ConvTranspose1d { config: ConvTranspose1dConfig },
    ConvTranspose2d { config: ConvTranspose2dConfig },
    ConvTranspose3d { config: ConvTranspose3dConfig },
    AveragePool1d { config: AvgPool1dConfig },
    AveragePool2d { config: AvgPool2dConfig },
    MaxPool1d { config: MaxPool1dConfig },
    MaxPool2d { config: MaxPool2dConfig },
    GlobalAveragePool,
    GlobalMaxPool,

    // NORMALIZATION
    BatchNormalization { config: BatchNormConfig },
    InstanceNormalization { config: InstanceNormConfig },
    LayerNormalization { config: LayerNormConfig },
    GroupNormalization { config: GroupNormConfig },

    // DROPOUT & REGULARIZATION
    Dropout { config: DropoutConfig },

    // LINEAR & SPECIAL LAYERS
    Linear { config: LinearConfig },
    Attention { config: AttentionConfig },

    // CONSTANT GENERATION
    Constant,
    ConstantOfShape { config: ConstantOfShapeConfig },
    EyeLike { config: EyeLikeConfig },

    // RANDOM OPERATIONS
    RandomNormal { config: RandomNormalConfig },
    RandomUniform { config: RandomUniformConfig },
    RandomNormalLike { config: RandomNormalLikeConfig },
    RandomUniformLike { config: RandomUniformLikeConfig },
    Bernoulli,

    // RANGE & SEQUENCE OPERATIONS
    Range { config: RangeConfig },
    OneHot { config: OneHotConfig },

    // CONTROL FLOW
    If { config: IfConfig },
    Loop { config: LoopConfig },
    Scan { config: ScanConfig },

    // SPECIAL OPERATIONS
    IsInf { config: IsInfConfig },
    IsNaN,
    NonZero,
    TopK { config: TopKConfig },
    Unique,
    Trilu { config: TriluConfig },
    Mod { config: ModConfig },
    CumSum,
}
