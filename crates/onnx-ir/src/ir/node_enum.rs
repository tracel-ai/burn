//! Enum-based node representation for ONNX operations
//!
//! This module contains the Node type which provides compile-time type safety
//! for ONNX operations by encoding the operation type and its configuration in enum variants.

use super::argument::Argument;
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
use crate::node::random::RandomNormalConfig;
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

/// Enum-based node representation
///
/// Each ONNX operation is represented as a separate enum variant containing
/// the operation-specific configuration.
#[derive(Debug, Clone)]
pub enum Node {
    // =========================================================================
    // ARITHMETIC & BASIC OPERATIONS (no config)
    // =========================================================================
    Add {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Sub {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Mul {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Div {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Neg {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Abs {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Pow {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Reciprocal {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Sqrt {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Exp {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Log {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Ceil {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Floor {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Round {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Sign {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Erf {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // TRIGONOMETRIC OPERATIONS (no config)
    // =========================================================================
    Sin {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Cos {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Tan {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Asin {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Acos {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Atan {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Sinh {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Cosh {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Tanh {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Asinh {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Acosh {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Atanh {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // ACTIVATION FUNCTIONS
    // =========================================================================
    Relu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Sigmoid {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Softmax {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: SoftmaxConfig,
    },
    LogSoftmax {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: LogSoftmaxConfig,
    },
    LeakyRelu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: LeakyReluConfig,
    },
    HardSigmoid {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: HardSigmoidConfig,
    },
    Elu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Selu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Celu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Gelu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Mish {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Softplus {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Softsign {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    ThresholdedRelu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    HardSwish {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    PRelu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // COMPARISON & LOGICAL OPERATIONS (no config)
    // =========================================================================
    Equal {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Greater {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    GreaterOrEqual {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Less {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    LessOrEqual {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    And {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Or {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Xor {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Not {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Where {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // BITWISE OPERATIONS
    // =========================================================================
    BitwiseAnd {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    BitwiseOr {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    BitwiseXor {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    BitwiseNot {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    BitShift {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: BitShiftConfig,
    },

    // =========================================================================
    // REDUCTION OPERATIONS
    // =========================================================================
    ArgMax {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ArgMaxConfig,
    },
    ArgMin {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ArgMinConfig,
    },
    ReduceMax {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceMin {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceMean {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceSum {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceProd {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceL1 {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceL2 {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceLogSum {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceLogSumExp {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },
    ReduceSumSquare {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReduceConfig,
    },

    // =========================================================================
    // AGGREGATION OPERATIONS (no config)
    // =========================================================================
    Max {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Min {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Mean {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Sum {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // TENSOR MANIPULATION
    // =========================================================================
    Cast {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: CastConfig,
    },
    Clip {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ClipConfig,
    },
    Concat {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ConcatConfig,
    },
    Expand {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Flatten {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: FlattenConfig,
    },
    Gather {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: GatherConfig,
    },
    GatherElements {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: GatherElementsConfig,
    },
    GatherND {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Identity {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Pad {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: PadConfig,
    },
    Reshape {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ReshapeConfig,
    },
    Resize {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ResizeConfig,
    },
    Scatter {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    ScatterElements {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    ScatterND {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Shape {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ShapeConfig,
    },
    Size {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Slice {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: SliceConfig,
    },
    Split {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: SplitConfig,
    },
    Squeeze {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: SqueezeConfig,
    },
    Tile {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: TileConfig,
    },
    Transpose {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: TransposeConfig,
    },
    Unsqueeze {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: UnsqueezeConfig,
    },
    DepthToSpace {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: DepthToSpaceConfig,
    },
    SpaceToDepth {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: SpaceToDepthConfig,
    },

    // =========================================================================
    // MATRIX OPERATIONS
    // =========================================================================
    MatMul {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    MatMulInteger {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Gemm {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: GemmConfig,
    },

    // =========================================================================
    // CONVOLUTION & POOLING
    // =========================================================================
    Conv1d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: Conv1dConfig,
    },
    Conv2d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: Conv2dConfig,
    },
    Conv3d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: Conv3dConfig,
    },
    ConvTranspose1d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ConvTranspose1dConfig,
    },
    ConvTranspose2d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ConvTranspose2dConfig,
    },
    ConvTranspose3d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ConvTranspose3dConfig,
    },
    AveragePool1d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: AvgPool1dConfig,
    },
    AveragePool2d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: AvgPool2dConfig,
    },
    MaxPool1d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: MaxPool1dConfig,
    },
    MaxPool2d {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: MaxPool2dConfig,
    },
    GlobalAveragePool {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    GlobalMaxPool {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // NORMALIZATION
    // =========================================================================
    BatchNormalization {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: BatchNormConfig,
    },
    InstanceNormalization {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: InstanceNormConfig,
    },
    LayerNormalization {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: LayerNormConfig,
    },
    GroupNormalization {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: GroupNormConfig,
    },

    // =========================================================================
    // DROPOUT & REGULARIZATION
    // =========================================================================
    Dropout {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: DropoutConfig,
    },

    // =========================================================================
    // LINEAR & SPECIAL LAYERS
    // =========================================================================
    Linear {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: LinearConfig,
    },
    Attention {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: AttentionConfig,
    },

    // =========================================================================
    // CONSTANT GENERATION
    // =========================================================================
    Constant {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    ConstantOfShape {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ConstantOfShapeConfig,
    },
    EyeLike {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: EyeLikeConfig,
    },

    // =========================================================================
    // RANDOM OPERATIONS
    // =========================================================================
    RandomNormal {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: RandomNormalConfig,
    },
    RandomNormalLike {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: RandomNormalLikeConfig,
    },
    RandomUniformLike {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: RandomUniformLikeConfig,
    },
    Bernoulli {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    // =========================================================================
    // RANGE & SEQUENCE OPERATIONS
    // =========================================================================
    Range {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: RangeConfig,
    },
    OneHot {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: OneHotConfig,
    },

    // =========================================================================
    // CONTROL FLOW
    // =========================================================================
    If {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: IfConfig,
    },
    Loop {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: LoopConfig,
    },
    Scan {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ScanConfig,
    },

    // =========================================================================
    // SPECIAL OPERATIONS
    // =========================================================================
    IsInf {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: IsInfConfig,
    },
    IsNaN {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    NonZero {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    TopK {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: TopKConfig,
    },
    Unique {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
    Trilu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: TriluConfig,
    },
    Mod {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        config: ModConfig,
    },
    CumSum {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
}
