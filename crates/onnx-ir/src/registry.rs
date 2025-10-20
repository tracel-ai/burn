//! Processor registry for ONNX node types.
//!
//! This module provides a centralized registry that maps ONNX node types
//! to their corresponding processor implementations.

use crate::ir::NodeType;
use crate::processor::NodeProcessor;
use std::collections::HashMap;

/// Registry for node processors.
///
/// This registry maps node types to their corresponding processor implementations.
/// It provides a centralized way to look up and use processors for different node types.
pub struct ProcessorRegistry {
    processors: HashMap<NodeType, Box<dyn NodeProcessor>>,
}

impl ProcessorRegistry {
    /// Create a new empty processor registry
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
        }
    }

    /// Register a processor for a specific node type
    pub fn register(&mut self, node_type: NodeType, processor: Box<dyn NodeProcessor>) {
        self.processors.insert(node_type, processor);
    }

    /// Get the processor for a node type, or the default processor if not found
    pub fn get(&self, node_type: &NodeType) -> &dyn NodeProcessor {
        self.processors
            .get(node_type)
            .map(|b| b.as_ref())
            .unwrap_or(&crate::processor::DefaultProcessor)
    }
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessorRegistry {
    /// Create a registry with all standard ONNX processors registered
    ///
    /// This function registers all supported ONNX operators with their processors.
    /// When adding a new node type, add its registration here.
    pub fn with_standard_processors() -> Self {
        let mut registry = Self::new();

        // Basic arithmetic binary operations (with Shape/Scalar type propagation)
        registry.register(
            NodeType::Add,
            Box::new(crate::node::arithmetic::ArithmeticBinaryProcessor),
        );
        registry.register(
            NodeType::Sub,
            Box::new(crate::node::arithmetic::ArithmeticBinaryProcessor),
        );
        registry.register(
            NodeType::Mul,
            Box::new(crate::node::arithmetic::ArithmeticBinaryProcessor),
        );
        registry.register(
            NodeType::Div,
            Box::new(crate::node::arithmetic::ArithmeticBinaryProcessor),
        );

        // Other element-wise binary operations (simple broadcasting, no special type handling)
        registry.register(
            NodeType::Pow,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::Max,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::Min,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::And,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::Or,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::Xor,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(NodeType::Sum, Box::new(crate::node::sum::SumProcessor));

        // Unary operations
        registry.register(
            NodeType::Abs,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Neg,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Ceil,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Floor,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Sqrt,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Exp,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Log,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Sin,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Cos,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Tan,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Tanh,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Sinh,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Cosh,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Asin,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Acos,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Atan,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Asinh,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Acosh,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Atanh,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Erf,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(NodeType::Relu, Box::new(crate::node::relu::ReluProcessor));
        registry.register(
            NodeType::Sigmoid,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Sign,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Reciprocal,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Not,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Round,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Gelu,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor), // FIXME has config
        );

        // Pooling operations
        registry.register(
            NodeType::AveragePool1d,
            Box::new(crate::node::avg_pool1d::AvgPool1dProcessor),
        );
        registry.register(
            NodeType::AveragePool2d,
            Box::new(crate::node::avg_pool2d::AvgPool2dProcessor),
        );
        registry.register(
            NodeType::MaxPool1d,
            Box::new(crate::node::max_pool1d::MaxPool1dProcessor),
        );
        registry.register(
            NodeType::MaxPool2d,
            Box::new(crate::node::max_pool2d::MaxPool2dProcessor),
        );
        registry.register(
            NodeType::GlobalAveragePool,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );

        // Convolution operations
        registry.register(
            NodeType::Conv1d,
            Box::new(crate::node::conv1d::Conv1dProcessor),
        );
        registry.register(
            NodeType::Conv2d,
            Box::new(crate::node::conv2d::Conv2dProcessor),
        );
        registry.register(
            NodeType::Conv3d,
            Box::new(crate::node::conv3d::Conv3dProcessor),
        );
        registry.register(
            NodeType::ConvTranspose1d,
            Box::new(crate::node::conv_transpose1d::Convtranspose1dProcessor),
        );
        registry.register(
            NodeType::ConvTranspose2d,
            Box::new(crate::node::conv_transpose2d::Convtranspose2dProcessor),
        );
        registry.register(
            NodeType::ConvTranspose3d,
            Box::new(crate::node::conv_transpose3d::Convtranspose3dProcessor),
        );

        // Normalization operations
        registry.register(
            NodeType::BatchNormalization,
            Box::new(crate::node::batch_norm::BatchNormProcessor),
        );
        registry.register(
            NodeType::InstanceNormalization,
            Box::new(crate::node::instance_norm::InstanceNormProcessor),
        );
        registry.register(
            NodeType::LayerNormalization,
            Box::new(crate::node::layer_norm::LayerNormProcessor),
        );
        registry.register(
            NodeType::GroupNormalization,
            Box::new(crate::node::group_norm::GroupNormProcessor),
        );

        // Shape operations
        registry.register(
            NodeType::Reshape,
            Box::new(crate::node::reshape::ReshapeProcessor),
        );
        registry.register(
            NodeType::Transpose,
            Box::new(crate::node::transpose::TransposeProcessor),
        );
        registry.register(
            NodeType::Flatten,
            Box::new(crate::node::flatten::FlattenProcessor),
        );
        registry.register(
            NodeType::Squeeze,
            Box::new(crate::node::squeeze::SqueezeProcessor),
        );
        registry.register(
            NodeType::Unsqueeze,
            Box::new(crate::node::unsqueeze::UnsqueezeProcessor),
        );

        // Utility operations
        registry.register(NodeType::Clip, Box::new(crate::node::clip::ClipProcessor));
        registry.register(
            NodeType::Dropout,
            Box::new(crate::node::dropout::DropoutProcessor),
        );
        registry.register(NodeType::Pad, Box::new(crate::node::pad::PadProcessor));

        // Reduction operations
        registry.register(
            NodeType::ReduceMax,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceMin,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceMean,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceProd,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceSum,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceSumSquare,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceL1,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceL2,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceLogSum,
            Box::new(crate::node::reduce::ReduceProcessor),
        );
        registry.register(
            NodeType::ReduceLogSumExp,
            Box::new(crate::node::reduce::ReduceProcessor),
        );

        // Matrix operations
        registry.register(
            NodeType::MatMul,
            Box::new(crate::node::matmul::MatMulProcessor),
        );
        registry.register(
            NodeType::Linear,
            Box::new(crate::node::linear::LinearProcessor),
        );
        registry.register(NodeType::Gemm, Box::new(crate::node::gemm::GemmProcessor));
        registry.register(
            NodeType::MatMulInteger,
            Box::new(crate::node::matmulinteger::MatMulIntegerProcessor),
        );

        // Array operations
        registry.register(
            NodeType::Concat,
            Box::new(crate::node::concat::ConcatProcessor),
        );
        registry.register(
            NodeType::Split,
            Box::new(crate::node::split::SplitProcessor),
        );
        registry.register(
            NodeType::Gather,
            Box::new(crate::node::gather::GatherProcessor),
        );
        registry.register(
            NodeType::GatherElements,
            Box::new(crate::node::gather_elements::GatherElementsProcessor),
        );
        registry.register(
            NodeType::Slice,
            Box::new(crate::node::slice::SliceProcessor),
        );
        registry.register(NodeType::Tile, Box::new(crate::node::tile::TileProcessor));
        registry.register(
            NodeType::Expand,
            Box::new(crate::node::expand::ExpandProcessor),
        );

        // Comparison operations
        registry.register(
            NodeType::Equal,
            Box::new(crate::node::comparison::ComparisonProcessor),
        );
        registry.register(
            NodeType::Greater,
            Box::new(crate::node::comparison::ComparisonProcessor),
        );
        registry.register(
            NodeType::Less,
            Box::new(crate::node::comparison::ComparisonProcessor),
        );
        registry.register(
            NodeType::GreaterOrEqual,
            Box::new(crate::node::comparison::ComparisonProcessor),
        );
        registry.register(
            NodeType::LessOrEqual,
            Box::new(crate::node::comparison::ComparisonProcessor),
        );
        registry.register(
            NodeType::IsInf,
            Box::new(crate::node::is_inf::IsInfProcessor),
        );
        registry.register(
            NodeType::IsNaN,
            Box::new(crate::node::is_nan::IsNaNProcessor),
        );

        // Special operations
        registry.register(NodeType::Cast, Box::new(crate::node::cast::CastProcessor));
        registry.register(
            NodeType::Shape,
            Box::new(crate::node::shape::ShapeProcessor),
        );
        registry.register(NodeType::Size, Box::new(crate::node::size::SizeProcessor));
        registry.register(
            NodeType::Constant,
            Box::new(crate::node::constant::ConstantProcessor),
        );
        registry.register(
            NodeType::ConstantOfShape,
            Box::new(crate::node::constant_of_shape::ConstantOfShapeProcessor),
        );
        registry.register(
            NodeType::OneHot,
            Box::new(crate::node::one_hot::OneHotProcessor),
        );
        registry.register(
            NodeType::Where,
            Box::new(crate::node::where_op::WhereProcessor),
        );
        registry.register(
            NodeType::NonZero,
            Box::new(crate::node::nonzero::NonZeroProcessor),
        );

        // ArgMax/ArgMin/TopK
        registry.register(
            NodeType::ArgMax,
            Box::new(crate::node::argmax::ArgMaxProcessor),
        );
        registry.register(
            NodeType::ArgMin,
            Box::new(crate::node::argmin::ArgMinProcessor),
        );
        registry.register(NodeType::TopK, Box::new(crate::node::topk::TopKProcessor));

        // Spatial operations
        registry.register(
            NodeType::Resize,
            Box::new(crate::node::resize::ResizeProcessor),
        );
        registry.register(
            NodeType::DepthToSpace,
            Box::new(crate::node::depth_to_space::DepthToSpaceProcessor),
        );
        registry.register(
            NodeType::SpaceToDepth,
            Box::new(crate::node::space_to_depth::SpaceToDepthProcessor),
        );

        // Random operations
        registry.register(
            NodeType::RandomNormal,
            Box::new(crate::node::random::RandomProcessor),
        );
        registry.register(
            NodeType::RandomUniform,
            Box::new(crate::node::random::RandomProcessor),
        );
        registry.register(
            NodeType::RandomNormalLike,
            Box::new(crate::node::random_like::RandomLikeProcessor),
        );
        registry.register(
            NodeType::RandomUniformLike,
            Box::new(crate::node::random_like::RandomLikeProcessor),
        );
        registry.register(
            NodeType::Bernoulli,
            Box::new(crate::node::bernoulli::BernoulliProcessor),
        );

        // Misc operations
        registry.register(
            NodeType::EyeLike,
            Box::new(crate::node::eye_like::EyeLikeProcessor),
        );
        registry.register(
            NodeType::Range,
            Box::new(crate::node::range::RangeProcessor),
        );
        registry.register(
            NodeType::Attention,
            Box::new(crate::node::attention::AttentionProcessor),
        );
        registry.register(
            NodeType::BitShift,
            Box::new(crate::node::bitshift::BitShiftProcessor),
        );
        registry.register(
            NodeType::BitwiseAnd,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::BitwiseOr,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::BitwiseXor,
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );
        registry.register(
            NodeType::BitwiseNot,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
        registry.register(
            NodeType::Mod,
            Box::new(crate::node::modulo::ModuloProcessor),
        );
        registry.register(
            NodeType::Trilu,
            Box::new(crate::node::trilu::TriluProcessor),
        );
        registry.register(
            NodeType::LeakyRelu,
            Box::new(crate::node::leaky_relu::LeakyReluProcessor),
        );
        registry.register(
            NodeType::HardSigmoid,
            Box::new(crate::node::hard_sigmoid::HardSigmoidProcessor),
        );
        registry.register(
            NodeType::Softmax,
            Box::new(crate::node::softmax::SoftmaxProcessor),
        );
        registry.register(
            NodeType::LogSoftmax,
            Box::new(crate::node::log_softmax::LogSoftmaxProcessor),
        );
        registry.register(
            NodeType::PRelu,
            Box::new(crate::node::prelu::PReluProcessor),
        );

        registry
    }
}

// Processor registry singleton
use std::sync::OnceLock;

static PROCESSOR_REGISTRY: OnceLock<ProcessorRegistry> = OnceLock::new();

/// Get the processor registry singleton
pub fn get_processor_registry() -> &'static ProcessorRegistry {
    PROCESSOR_REGISTRY.get_or_init(ProcessorRegistry::with_standard_processors)
}
