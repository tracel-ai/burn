//! Placeholder node types for unsupported/unimplemented ONNX operations
//!
//! These operations are part of the ONNX spec but don't have full implementations yet.
//! They are defined here as placeholders to maintain type safety and allow the graph
//! to be parsed even when it contains unsupported operations.

use crate::ir::Argument;

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
    GlobalAveragePoolNode,
    GlobalMaxPoolNode,
    GridSampleNode,
    GruNode,
    HammingWindowNode,
    HannWindowNode,
    HardmaxNode,
    IdentityNode,
    ImNode,
    ImageDecoderNode,
    LpNormalizationNode,
    LpPoolNode,
    LrnNode,
    LstmNode,
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
