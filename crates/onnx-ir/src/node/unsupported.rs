//! Placeholder node types for unsupported/unimplemented ONNX operations
//!
//! These operations are part of the ONNX spec but don't have full implementations yet.
//! They are defined here as placeholders to maintain type safety and allow the graph
//! to be parsed even when it contains unsupported operations.

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Generic placeholder for unsupported operations
macro_rules! define_placeholder_node {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Debug, Clone)]
            pub struct $name {
                pub name: String,
                pub inputs: Vec<Argument>,
                pub outputs: Vec<Argument>,
            }
        )*
    };
}

define_placeholder_node! {
    AffineGridNode,
    AveragePoolNode,
    BlackmanWindowNode,
    CastLikeNode,
    CenterCropPadNode,
    Col2ImNode,
    CompressNode,
    ConcatFromSequenceNode,
    ConvNode,
    ConvIntegerNode,
    ConvTransposeNode,
    CumSumNode,
    DftNode,
    DeformConvNode,
    DequantizeLinearNode,
    DetNode,
    DynamicQuantizeLinearNode,
    EinsumNode,
    GatherNDNode,
    GlobalMaxPoolNode,
    GruNode,
    HammingWindowNode,
    HannWindowNode,
    HardmaxNode,
    ImNode,
    ImageDecoderNode,
    LpNormalizationNode,
    LpPoolNode,
    LrnNode,
    MaxPoolNode,
    MaxRoiPoolNode,
    MaxUnpoolNode,
    MeanVarianceNormalizationNode,
    MelWeightMatrixNode,
    MultinomialNode,
    NegativeLogLikelihoodLossNode,
    NonMaxSuppressionNode,
    OptionalNode,
    OptionalGetElementNode,
    OptionalHasElementNode,
    QLinearConvNode,
    QLinearMatMulNode,
    QuantizeLinearNode,
    RMSNormalizationNode,
    RnnNode,
    RegexFullMatchNode,
    ReverseSequenceNode,
    RoiAlignNode,
    RotaryEmbeddingNode,
    ScatterNode,
    ScatterElementsNode,
    ScatterNDNode,
    SequenceAtNode,
    SequenceConstructNode,
    SequenceEmptyNode,
    SequenceEraseNode,
    SequenceInsertNode,
    SequenceLengthNode,
    SequenceMapNode,
    ShrinkNode,
    SoftmaxCrossEntropyLossNode,
    SplitToSequenceNode,
    StftNode,
    StringConcatNode,
    StringNormalizerNode,
    StringSplitNode,
    SwishNode,
    TensorScatterNode,
    TfIdfVectorizerNode,
    UniqueNode,
    UpsampleNode,
}

/// Generic processor for unsupported operations
///
/// This processor creates placeholder nodes for operations that don't have full implementations.
/// It performs basic input/output validation but doesn't extract any configuration.
pub(crate) struct UnsupportedProcessor;

impl NodeProcessor for UnsupportedProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            // Accept any number of inputs/outputs
            inputs: InputSpec::AtLeast(0),
            outputs: OutputSpec::Range(0, usize::MAX),
        }
    }

    fn infer_types(
        &self,
        _node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // For unsupported nodes, we don't infer types
        // They will keep whatever types were provided in the ONNX file
        Ok(())
    }

    fn extract_config(&self, _node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        use crate::ir::NodeType;

        match builder.node_type {
            NodeType::GlobalMaxPool => Node::GlobalMaxPool(GlobalMaxPoolNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            NodeType::GatherND => Node::GatherND(GatherNDNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            NodeType::Scatter => Node::Scatter(ScatterNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            NodeType::ScatterElements => Node::ScatterElements(ScatterElementsNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            NodeType::ScatterND => Node::ScatterND(ScatterNDNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            NodeType::Unique => Node::Unique(UniqueNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            NodeType::CumSum => Node::CumSum(CumSumNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            _ => panic!("Unsupported node type: {:?}", builder.node_type),
        }
    }
}
