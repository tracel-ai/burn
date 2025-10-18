//! Node processor trait and infrastructure for node-centric processing.
//!
//! This module defines the `NodeProcessor` trait with support for type preferences
//! and proper error handling.

use crate::ir::{Node, NodeConfig, NodeType};
use std::collections::HashMap;

/// Type preferences for node inputs
#[derive(Debug, Default, Clone)]
pub struct InputPreferences {
    preferences: HashMap<String, Vec<ArgPreference>>,
}

#[derive(Debug, Clone)]
pub enum ArgPreference {
    Scalar,
    Shape,
    Tensor,
}

impl InputPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(mut self, input_name: impl Into<String>, ty: ArgPreference) -> Self {
        self.preferences
            .entry(input_name.into())
            .or_default()
            .push(ty);
        self
    }

    pub fn get(&self, input_name: &str) -> &[ArgPreference] {
        self.preferences
            .get(input_name)
            .map_or(&[], |v| v.as_slice())
    }
}

/// Type preferences requested by consumers for a node's outputs
#[derive(Debug, Default, Clone)]
pub struct OutputPreferences {
    // output_name -> [(consumer_name, requested_type)]
    requests: HashMap<String, Vec<(String, ArgPreference)>>,
}

impl OutputPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        output_name: impl Into<String>,
        consumer: impl Into<String>,
        ty: ArgPreference,
    ) {
        self.requests
            .entry(output_name.into())
            .or_default()
            .push((consumer.into(), ty));
    }

    pub fn get(&self, output_name: &str) -> &[(String, ArgPreference)] {
        self.requests.get(output_name).map_or(&[], |v| v.as_slice())
    }
}

/// Errors that can occur during node processing
#[derive(Debug, Clone)]
pub enum ProcessError {
    UnsupportedOpset {
        required: usize,
        actual: usize,
    },
    MissingInput(String),
    MissingOutput(String),
    InvalidInputCount {
        expected: usize,
        actual: usize,
    },
    InvalidOutputCount {
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        expected: String,
        actual: String,
    },
    ConflictingPreferences {
        output: String,
        details: Vec<String>,
    },
    MissingAttribute(String),
    InvalidAttribute {
        name: String,
        reason: String,
    },
    Custom(String),
}

/// Node-specific processing logic trait.
///
/// Each node type implements this trait to declare how to process nodes
/// during type inference and configuration extraction.
pub trait NodeProcessor: Send + Sync {
    /// Declare what types this node prefers to receive on its inputs.
    ///
    /// This method allows a node to request specific types from its input producers.
    /// The system propagates these preferences back to producer nodes as `output_preferences`,
    /// which producers can optionally honor when setting their output types.
    ///
    /// # How it works:
    /// 1. Consumer node declares `input_preferences()` - what types it prefers for each input
    /// 2. System collects these and maps them to `OutputPreferences` for producer nodes
    /// 3. Producer nodes receive these in `infer_types(node, opset, output_preferences)`
    /// 4. Producers can optionally honor these preferences (e.g., Constant can convert to Shape/Scalar)
    ///
    /// # Example:
    /// ```rust,ignore
    /// // ArithmeticBinaryProcessor (Add, Sub, etc.) prefers Shape inputs when one operand is Shape
    /// fn input_preferences(&self, node: &Node, _opset: usize) -> Result<Option<InputPreferences>, ProcessError> {
    ///     if node.inputs[0].ty.is_shape() {
    ///         // Request the second input to also be Shape for better type consistency
    ///         Ok(Some(InputPreferences::new().add(&node.inputs[1].name, ArgPreference::Shape)))
    ///     } else {
    ///         Ok(None)
    ///     }
    /// }
    /// ```
    ///
    /// # Important:
    /// - Preferences are requests, not requirements - producers may ignore them
    /// - Not all producers honor preferences (e.g., Shape node always outputs Shape)
    /// - Preferences help optimize type representations (e.g., Constant as Shape vs Tensor)
    fn input_preferences(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        Ok(None)
    }

    /// Lift constant inputs, return names of lifted inputs
    fn lift_constants(&self, _node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        Ok(Vec::new())
    }

    /// Infer output types given preferences from consumers
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError>;

    /// Extract config for codegen
    fn extract_config(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        Ok(None)
    }
}

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
            .unwrap_or(&DefaultProcessor)
    }

    /// Check if a processor is registered for a node type
    pub fn has_processor(&self, node_type: &NodeType) -> bool {
        self.processors.contains_key(node_type)
    }
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessorRegistry {
    /// Create a registry with all standard ONNX processors registered
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
        registry.register(
            NodeType::Relu,
            Box::new(crate::node::elementwise::ElementwiseUnaryProcessor),
        );
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
            Box::new(crate::node::elementwise::ElementwiseBinaryProcessor),
        );

        registry
    }
}

/// Default processor for nodes without specific implementations.
///
/// This processor passes through the input type to output.
struct DefaultProcessor;

impl NodeProcessor for DefaultProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Default: preserve input type
        crate::util::same_as_input(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, Node, NodeType, TensorType};

    struct TestProcessor;

    impl NodeProcessor for TestProcessor {
        fn infer_types(
            &self,
            node: &mut Node,
            _opset: usize,
            _output_preferences: &OutputPreferences,
        ) -> Result<(), ProcessError> {
            // Simple test: copy input type to output
            if !node.inputs.is_empty() && !node.outputs.is_empty() {
                node.outputs[0].ty = node.inputs[0].ty.clone();
            }
            Ok(())
        }
    }

    #[test]
    fn test_infer_outputs() {
        let processor = TestProcessor;
        let prefs = OutputPreferences::new();
        let mut node = Node {
            node_type: NodeType::Add,
            name: "test_node".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match input type
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn test_processor_registry() {
        let mut registry = ProcessorRegistry::new();

        // Register a processor
        registry.register(NodeType::Add, Box::new(TestProcessor));

        // Check if processor is registered
        assert!(registry.has_processor(&NodeType::Add));
        assert!(!registry.has_processor(&NodeType::Sub));
    }

    #[test]
    fn test_default_processor() {
        let processor = DefaultProcessor;
        let prefs = OutputPreferences::new();
        let mut node = Node {
            node_type: NodeType::Relu,
            name: "test_relu".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                data_id: None,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Default processor should preserve input type
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.elem_type, ElementType::Float32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
