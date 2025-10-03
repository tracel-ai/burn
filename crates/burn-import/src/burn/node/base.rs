use proc_macro2::TokenStream;
use serde::Serialize;
use std::marker::PhantomData;

use burn::record::PrecisionSettings;

use crate::burn::{BurnImports, Scope, Type, node::modulo::ModNode};

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

use super::abs::AbsNode;
use super::add::AddNode;
use super::argmax::ArgMaxNode;
use super::argmin::ArgMinNode;
use super::attention::AttentionNode;
use super::avg_pool1d::AvgPool1dNode;
use super::avg_pool2d::AvgPool2dNode;
use super::batch_norm::BatchNormNode;
use super::bernoulli::BernoulliNode;
use super::bitshift::BitShiftNode;
use super::bitwiseand::BitwiseAndNode;
use super::bitwisenot::BitwiseNotNode;
use super::bitwiseor::BitwiseOrNode;
use super::bitwisexor::BitwiseXorNode;
use super::bool_and::BoolAndNode;
use super::bool_or::BoolOrNode;
use super::bool_xor::BoolXorNode;
use super::cast::CastNode;
use super::ceil::CeilNode;
use super::clip::ClipNode;
use super::concat::ConcatNode;
use super::constant::ConstantNode;
use super::constant_of_shape::ConstantOfShapeNode;
use super::conv_transpose_1d::ConvTranspose1dNode;
use super::conv_transpose_2d::ConvTranspose2dNode;
use super::conv_transpose_3d::ConvTranspose3dNode;
use super::conv1d::Conv1dNode;
use super::conv2d::Conv2dNode;
use super::conv3d::Conv3dNode;
use super::cos::CosNode;
use super::cosh::CoshNode;
use super::depth_to_space::DepthToSpaceNode;
use super::div::DivNode;
use super::dropout::DropoutNode;
use super::equal::EqualNode;
use super::erf::ErfNode;
use super::exp::ExpNode;
use super::expand::ExpandNode;
use super::eye_like::EyeLikeNode;
use super::flatten::FlattenNode;
use super::floor::FloorNode;
use super::gather::GatherNode;
use super::gather_elements::GatherElementsNode;
use super::gelu::GeluNode;
use super::gemm::GemmNode;
use super::global_avg_pool::GlobalAvgPoolNode;
use super::greater::GreaterNode;
use super::greater_equal::GreaterEqualNode;
use super::group_norm::GroupNormNode;
use super::hard_sigmoid::HardSigmoidNode;
use super::identity::IdentityNode;
use super::instance_norm::InstanceNormNode;
use super::is_inf::IsInfNode;
use super::is_nan::IsNanNode;
use super::layer_norm::LayerNormNode;
use super::leaky_relu::LeakyReluNode;
use super::linear::LinearNode;
use super::log::LogNode;
use super::log_softmax::LogSoftmaxNode;
use super::lower::LowerNode;
use super::lower_equal::LowerEqualNode;
use super::matmul::MatmulNode;
use super::matmul_integer::MatMulIntegerNode;
use super::max_pair::MaxPairNode;
use super::max_pool1d::MaxPool1dNode;
use super::max_pool2d::MaxPool2dNode;
use super::mean::MeanNode;
use super::min_pair::MinPairNode;
use super::mul::MulNode;
use super::neg::NegNode;
use super::nonzero::NonZeroNode;
use super::not::NotNode;
use super::one_hot::OneHotNode;
use super::pad::PadNode;
use super::pow::PowNode;
use super::prelu::PReluNode;
use super::random_normal::RandomNormalNode;
use super::random_normal_like::RandomNormalLikeNode;
use super::random_uniform::RandomUniformNode;
use super::random_uniform_like::RandomUniformLikeNode;
use super::range::RangeNode;
use super::reciprocal::ReciprocalNode;
use super::reduce::ReduceNode;
use super::relu::ReluNode;
use super::reshape::ReshapeNode;
use super::resize::ResizeNode;
use super::round::RoundNode;
use super::shape::ShapeNode;
use super::sigmoid::SigmoidNode;
use super::sign::SignNode;
use super::sin::SinNode;
use super::sinh::SinhNode;
use super::size::SizeNode;
use super::slice::SliceNode;
use super::softmax::SoftmaxNode;
use super::space_to_depth::SpaceToDepthNode;
use super::split::SplitNode;
use super::sqrt::SqrtNode;
use super::squeeze::SqueezeNode;
use super::sub::SubNode;
use super::sum::SumNode;
use super::tan::TanNode;
use super::tanh::TanhNode;
use super::tile::TileNode;
use super::top_k::TopKNode;
use super::transpose::TransposeNode;
use super::trilu::TriluNode;
use super::unsqueeze::UnsqueezeNode;
use super::where_op::WhereNode;

/// Backend used for serialization.
pub type SerializationBackend = burn_ndarray::NdArray<f32>;

/// Codegen trait that should be implemented by all [node](Node) entries.
pub trait NodeCodegen<PS: PrecisionSettings>: std::fmt::Debug {
    /// All types that are used as inputs during the forward pass.
    ///
    /// # Notes
    /// The vec should not include types that are accessible with `self`.
    /// See [field type](NodeCodegen::field_type).
    fn input_types(&self) -> Vec<Type>;

    /// All types that are produced during the forward pass.
    fn output_types(&self) -> Vec<Type>;

    /// The forward pass implementation of the node.
    ///
    /// # Notes
    ///
    /// The [Scope](Scope) struct should be used for [input tensor type](Type::Tensor) access.
    /// The method [use_owned_tensor](Scope::use_owned_tensor) keeps track of tensor reference
    /// count and insert `clone` with necessary.
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream;

    /// Convert the node implementation into a [node entry](Node).
    fn into_node(self) -> Node<PS>;

    /// Register the necessary imports.
    fn register_imports(&self, _imports: &mut BurnImports) {}

    /// (Optional) Declare the type of the field
    ///
    /// # Notes
    ///
    /// This should be implemented when the node has some parameters.
    /// Just one field per type is possible, if the node has multiple types for its parameters, a
    /// tuple can be used.
    ///
    /// Other field functions should be implemented when this one returns something other than None.
    ///   * [field_init](NodeCodegen::field_init) to initialize parameters.
    ///   * [field_serialize](NodeCodegen::field_serialize) to create the model record.
    fn field_type(&self) -> Option<Type> {
        None
    }

    /// (Optional) Declare how the parameters are initialized.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_init(&self) -> Option<TokenStream> {
        None
    }

    /// (Optional) Declare how the parameters are serialized in a record.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        panic!("Serialization should be implemented when field_type is not None.");
    }
}

#[derive(Debug, Clone)]
pub enum Node<PS: PrecisionSettings> {
    Abs(AbsNode),
    Add(AddNode),
    ArgMax(ArgMaxNode),
    ArgMin(ArgMinNode),
    Attention(AttentionNode),
    AvgPool1d(AvgPool1dNode),
    AvgPool2d(AvgPool2dNode),
    BatchNorm(BatchNormNode),
    Bernoulli(BernoulliNode),
    BoolAnd(BoolAndNode),
    BoolOr(BoolOrNode),
    BoolXor(BoolXorNode),
    BitShift(BitShiftNode),
    BitwiseAnd(BitwiseAndNode),
    BitwiseOr(BitwiseOrNode),
    BitwiseNot(BitwiseNotNode),
    BitwiseXor(BitwiseXorNode),
    Cast(CastNode),
    Ceil(CeilNode),
    Clip(ClipNode),
    Concat(ConcatNode),
    Constant(ConstantNode),
    ConstantOfShape(ConstantOfShapeNode),
    Cos(CosNode),
    Cosh(CoshNode),
    Conv1d(Conv1dNode),
    Conv2d(Conv2dNode),
    Conv3d(Conv3dNode),
    ConvTranspose1d(ConvTranspose1dNode),
    ConvTranspose2d(ConvTranspose2dNode),
    ConvTranspose3d(ConvTranspose3dNode),
    DepthToSpace(DepthToSpaceNode),
    Div(DivNode),
    Dropout(DropoutNode),
    Equal(EqualNode),
    Erf(ErfNode),
    Exp(ExpNode),
    Expand(ExpandNode),
    EyeLike(EyeLikeNode),
    Flatten(FlattenNode),
    Floor(FloorNode),
    Gather(GatherNode),
    GatherElements(GatherElementsNode),
    Gelu(GeluNode),
    Gemm(GemmNode),
    Greater(GreaterNode),
    GreaterEqual(GreaterEqualNode),
    GlobalAvgPool(GlobalAvgPoolNode),
    GroupNorm(GroupNormNode),
    HardSigmoid(HardSigmoidNode),
    Identity(IdentityNode),
    InstanceNorm(InstanceNormNode),
    IsInf(IsInfNode),
    IsNan(IsNanNode),
    LayerNorm(LayerNormNode),
    LeakyRelu(LeakyReluNode),
    Linear(LinearNode),
    Log(LogNode),
    LogSoftmax(LogSoftmaxNode),
    Lower(LowerNode),
    LowerEqual(LowerEqualNode),
    Matmul(MatmulNode),
    MatmulInteger(MatMulIntegerNode),
    MaxPair(MaxPairNode),
    MaxPool1d(MaxPool1dNode),
    MaxPool2d(MaxPool2dNode),
    Mean(MeanNode),
    MinPair(MinPairNode),
    Mod(ModNode),
    Mul(MulNode),
    Neg(NegNode),
    NonZero(NonZeroNode),
    Not(NotNode),
    OneHot(OneHotNode),
    Pad(PadNode),
    Pow(PowNode),
    PRelu(PReluNode),
    RandomNormal(RandomNormalNode),
    RandomNormalLike(RandomNormalLikeNode),
    RandomUniform(RandomUniformNode),
    RandomUniformLike(RandomUniformLikeNode),
    Range(RangeNode),
    Reciprocal(ReciprocalNode),
    Reduce(ReduceNode),
    Relu(ReluNode),
    Reshape(ReshapeNode),
    Resize(ResizeNode),
    Round(RoundNode),
    Shape(ShapeNode),
    Sigmoid(SigmoidNode),
    Sign(SignNode),
    Sin(SinNode),
    Sinh(SinhNode),
    Size(SizeNode),
    Slice(SliceNode),
    Softmax(SoftmaxNode),
    SpaceToDepth(SpaceToDepthNode),
    Split(SplitNode),
    Sqrt(SqrtNode),
    Squeeze(SqueezeNode),
    Sub(SubNode),
    Sum(SumNode),
    Tan(TanNode),
    Tanh(TanhNode),
    Tile(TileNode),
    TopK(TopKNode),
    Transpose(TransposeNode),
    Trilu(TriluNode),
    Unsqueeze(UnsqueezeNode),
    Where(WhereNode),
    // For now, we have to keep the precision settings in order to correctly serialize the fields
    // into the right data types.
    _Unreachable(std::convert::Infallible, PhantomData<PS>),
}

macro_rules! match_all {
    ($self:expr, $func:expr) => {{
        #[allow(clippy::redundant_closure_call)]
        match $self {
            Node::Abs(node) => $func(node),
            Node::Add(node) => $func(node),
            Node::ArgMax(node) => $func(node),
            Node::ArgMin(node) => $func(node),
            Node::Attention(node) => $func(node),
            Node::AvgPool1d(node) => $func(node),
            Node::AvgPool2d(node) => $func(node),
            Node::BatchNorm(node) => $func(node),
            Node::Bernoulli(node) => $func(node),
            Node::BoolAnd(node) => $func(node),
            Node::BoolOr(node) => $func(node),
            Node::BoolXor(node) => $func(node),
            Node::BitShift(node) => $func(node),
            Node::BitwiseAnd(node) => $func(node),
            Node::BitwiseOr(node) => $func(node),
            Node::BitwiseNot(node) => $func(node),
            Node::BitwiseXor(node) => $func(node),
            Node::Cast(node) => $func(node),
            Node::Ceil(node) => $func(node),
            Node::Clip(node) => $func(node),
            Node::Concat(node) => $func(node),
            Node::Constant(node) => $func(node),
            Node::ConstantOfShape(node) => $func(node),
            Node::Cos(node) => $func(node),
            Node::Cosh(node) => $func(node),
            Node::Conv1d(node) => $func(node),
            Node::Conv2d(node) => $func(node),
            Node::Conv3d(node) => $func(node),
            Node::ConvTranspose1d(node) => $func(node),
            Node::ConvTranspose2d(node) => $func(node),
            Node::ConvTranspose3d(node) => $func(node),
            Node::DepthToSpace(node) => $func(node),
            Node::Div(node) => $func(node),
            Node::Dropout(node) => $func(node),
            Node::Equal(node) => $func(node),
            Node::Erf(node) => $func(node),
            Node::Exp(node) => $func(node),
            Node::Expand(node) => $func(node),
            Node::EyeLike(node) => $func(node),
            Node::Flatten(node) => $func(node),
            Node::Floor(node) => $func(node),
            Node::Gather(node) => $func(node),
            Node::GatherElements(node) => $func(node),
            Node::Gelu(node) => $func(node),
            Node::Gemm(node) => $func(node),
            Node::Greater(node) => $func(node),
            Node::GreaterEqual(node) => $func(node),
            Node::GlobalAvgPool(node) => $func(node),
            Node::GroupNorm(node) => $func(node),
            Node::HardSigmoid(node) => $func(node),
            Node::Identity(node) => $func(node),
            Node::InstanceNorm(node) => $func(node),
            Node::IsInf(node) => $func(node),
            Node::IsNan(node) => $func(node),
            Node::LayerNorm(node) => $func(node),
            Node::LeakyRelu(node) => $func(node),
            Node::Linear(node) => $func(node),
            Node::Log(node) => $func(node),
            Node::LogSoftmax(node) => $func(node),
            Node::Lower(node) => $func(node),
            Node::LowerEqual(node) => $func(node),
            Node::Matmul(node) => $func(node),
            Node::MatmulInteger(node) => $func(node),
            Node::MaxPair(node) => $func(node),
            Node::MaxPool1d(node) => $func(node),
            Node::MaxPool2d(node) => $func(node),
            Node::Mean(node) => $func(node),
            Node::MinPair(node) => $func(node),
            Node::Mod(node) => $func(node),
            Node::Mul(node) => $func(node),
            Node::Neg(node) => $func(node),
            Node::NonZero(node) => $func(node),
            Node::Not(node) => $func(node),
            Node::OneHot(node) => $func(node),
            Node::Pad(node) => $func(node),
            Node::Pow(node) => $func(node),
            Node::PRelu(node) => $func(node),
            Node::RandomNormal(node) => $func(node),
            Node::RandomNormalLike(node) => $func(node),
            Node::RandomUniform(node) => $func(node),
            Node::RandomUniformLike(node) => $func(node),
            Node::Range(node) => $func(node),
            Node::Reciprocal(node) => $func(node),
            Node::Reduce(node) => $func(node),
            Node::Relu(node) => $func(node),
            Node::Reshape(node) => $func(node),
            Node::Resize(node) => $func(node),
            Node::Round(node) => $func(node),
            Node::Shape(node) => $func(node),
            Node::Sigmoid(node) => $func(node),
            Node::Sign(node) => $func(node),
            Node::Sin(node) => $func(node),
            Node::Sinh(node) => $func(node),
            Node::Size(node) => $func(node),
            Node::Slice(node) => $func(node),
            Node::Softmax(node) => $func(node),
            Node::SpaceToDepth(node) => $func(node),
            Node::Split(node) => $func(node),
            Node::Sqrt(node) => $func(node),
            Node::Squeeze(node) => $func(node),
            Node::Sub(node) => $func(node),
            Node::Sum(node) => $func(node),
            Node::Tan(node) => $func(node),
            Node::Tanh(node) => $func(node),
            Node::Tile(node) => $func(node),
            Node::TopK(node) => $func(node),
            Node::Transpose(node) => $func(node),
            Node::Trilu(node) => $func(node),
            Node::Unsqueeze(node) => $func(node),
            Node::Where(node) => $func(node),
            _ => unimplemented!(),
        }
    }};
}

impl<PS: PrecisionSettings> Serialize for Node<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.field_serialize(serializer)
    }
}

impl<PS: PrecisionSettings> Node<PS> {
    pub fn name(&self) -> &str {
        match self {
            Node::Abs(_) => "abs",
            Node::Add(_) => "add",
            Node::ArgMax(_) => "argmax",
            Node::ArgMin(_) => "argmin",
            Node::Attention(_) => "attention",
            Node::AvgPool1d(_) => "avg_pool1d",
            Node::AvgPool2d(_) => "avg_pool2d",
            Node::BatchNorm(_) => "batch_norm",
            Node::Bernoulli(_) => "bernoulli",
            Node::BoolAnd(_) => "and",
            Node::BoolOr(_) => "or",
            Node::BoolXor(_) => "xor",
            Node::BitShift(_) => "bitshift",
            Node::BitwiseAnd(_) => "bitwiseand",
            Node::BitwiseOr(_) => "bitwiseor",
            Node::BitwiseNot(_) => "bitwisenot",
            Node::BitwiseXor(_) => "bitwisexor",
            Node::Cast(_) => "cast",
            Node::Ceil(_) => "ceil",
            Node::Clip(_) => "clip",
            Node::Concat(_) => "concat",
            Node::Constant(_) => "constant",
            Node::ConstantOfShape(_) => "constant_of_shape",
            Node::Cos(_) => "cos",
            Node::Cosh(_) => "cosh",
            Node::Conv1d(_) => "conv1d",
            Node::Conv2d(_) => "conv2d",
            Node::Conv3d(_) => "conv3d",
            Node::ConvTranspose1d(_) => "conv_transpose1d",
            Node::ConvTranspose2d(_) => "conv_transpose2d",
            Node::ConvTranspose3d(_) => "conv_transpose3d",
            Node::DepthToSpace(_) => "depth_to_space",
            Node::Div(_) => "div",
            Node::Dropout(_) => "dropout",
            Node::Equal(_) => "equal",
            Node::Erf(_) => "erf",
            Node::Exp(_) => "exp",
            Node::Expand(_) => "expand",
            Node::EyeLike(_) => "eye_like",
            Node::Flatten(_) => "flatten",
            Node::Floor(_) => "floor",
            Node::Gather(_) => "gather",
            Node::GatherElements(_) => "gather_elements",
            Node::Gelu(_) => "gelu",
            Node::Gemm(_) => "gemm",
            Node::Greater(_) => "greater",
            Node::GreaterEqual(_) => "greater_equal",
            Node::GlobalAvgPool(_) => "global_avg_pool",
            Node::GroupNorm(_) => "group_norm",
            Node::HardSigmoid(_) => "hard_sigmoid",
            Node::Identity(_) => "identity",
            Node::InstanceNorm(_) => "instance_norm",
            Node::IsInf(_) => "is_inf",
            Node::IsNan(_) => "is_nan",
            Node::LayerNorm(_) => "layer_norm",
            Node::LeakyRelu(_) => "leaky_relu",
            Node::Linear(_) => "linear",
            Node::Log(_) => "log",
            Node::LogSoftmax(_) => "log_softmax",
            Node::Lower(_) => "lower",
            Node::LowerEqual(_) => "lower_equal",
            Node::Matmul(_) => "matmul",
            Node::MatmulInteger(_) => "matmul_integer",
            Node::MaxPair(_) => "max_pair",
            Node::MaxPool1d(_) => "max_pool1d",
            Node::MaxPool2d(_) => "max_pool2d",
            Node::Mean(_) => "mean",
            Node::MinPair(_) => "min_pair",
            Node::Mod(_) => "mod",
            Node::Mul(_) => "mul",
            Node::Neg(_) => "neg",
            Node::NonZero(_) => "nonzero",
            Node::Not(_) => "not",
            Node::OneHot(_) => "one_hot",
            Node::Pad(_) => "pad",
            Node::Pow(_) => "pow",
            Node::PRelu(_) => "prelu",
            Node::RandomNormal(_) => "random_normal",
            Node::RandomNormalLike(_) => "random_normal_like",
            Node::RandomUniform(_) => "random_uniform",
            Node::RandomUniformLike(_) => "random_uniform_like",
            Node::Range(_) => "range",
            Node::Reciprocal(_) => "reciprocal",
            Node::Reduce(_) => "reduce",
            Node::Relu(_) => "relu",
            Node::Reshape(_) => "reshape",
            Node::Resize(_) => "resize",
            Node::Round(_) => "round",
            Node::Shape(_) => "shape",
            Node::Sigmoid(_) => "sigmoid",
            Node::Sign(_) => "sign",
            Node::Sin(_) => "sin",
            Node::Sinh(_) => "sinh",
            Node::Size(_) => "size",
            Node::Slice(_) => "slice",
            Node::Softmax(_) => "softmax",
            Node::SpaceToDepth(_) => "space_to_depth",
            Node::Split(_) => "split",
            Node::Sqrt(_) => "sqrt",
            Node::Squeeze(_) => "squeeze",
            Node::Sub(_) => "sub",
            Node::Sum(_) => "sum",
            Node::Tan(_) => "tan",
            Node::Tanh(_) => "tanh",
            Node::Tile(_) => "tile",
            Node::TopK(_) => "top_k",
            Node::Transpose(_) => "transpose",
            Node::Trilu(_) => "trilu",
            Node::Unsqueeze(_) => "unsqueeze",
            Node::Where(_) => "where",
            _ => unimplemented!(),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for Node<PS> {
    fn output_types(&self) -> Vec<Type> {
        match_all!(self, NodeCodegen::<PS>::output_types)
    }

    fn input_types(&self) -> Vec<Type> {
        match_all!(self, NodeCodegen::<PS>::input_types)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match_all!(self, |node| NodeCodegen::<PS>::forward(
            node,
            scope,
            node_position
        ))
    }

    fn field_type(&self) -> Option<Type> {
        match_all!(self, NodeCodegen::<PS>::field_type)
    }

    fn field_init(&self) -> Option<TokenStream> {
        match_all!(self, |node| NodeCodegen::<PS>::field_init(node,))
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match_all!(self, |node| NodeCodegen::<PS>::register_imports(
            node, imports
        ))
    }

    fn into_node(self) -> Node<PS> {
        self
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match_all!(self, |node| NodeCodegen::<PS>::field_serialize(
            node, serializer
        ))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::burn::{
        BurnImports, TensorType,
        graph::BurnGraph,
        node::{NodeCodegen, conv2d::Conv2dNode, matmul::MatmulNode, test::assert_tokens},
    };
    use burn::{record::FullPrecisionSettings, tensor::TensorData};
    use onnx_ir::node::{conv2d::Conv2dConfig, padding::PaddingConfig2d};
    use proc_macro2::TokenStream;
    use quote::quote;

    #[track_caller]
    pub(crate) fn one_node_graph<T: NodeCodegen<FullPrecisionSettings> + Clone + 'static>(
        node_gen: T,
        forward: TokenStream,
        input_names: Vec<String>,
        output_names: Vec<String>,
    ) {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(node_gen.clone());

        graph.register_input_output(input_names, output_names);

        let mut imports = BurnImports::default();
        node_gen.register_imports(&mut imports);
        let imports = imports.codegen();

        let expected = quote! {
            #imports

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                #forward
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_two_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            TensorType::new_float("tensor3", 4),
        ));
        graph.register(Conv2dNode::new(
            "conv2d",
            TensorType::new_float("tensor3", 4),
            TensorType::new_float("tensor4", 4),
            TensorData::from([2f32]),
            None,
            Conv2dConfig::new(
                [3, 3],
                [3, 3],
                [1, 1],
                PaddingConfig2d::Valid,
                [1, 1],
                1,
                true,
            ),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor4".to_string()],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::PaddingConfig2d;
            use burn::nn::conv::Conv2d;
            use burn::nn::conv::Conv2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv2d: Conv2d<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let conv2d = Conv2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init(device);

                    Self {
                        conv2d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4> {
                    let tensor3 = tensor1.matmul(tensor2);
                    let tensor4 = self.conv2d.forward(tensor3);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_clone_tensor() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorType::new_float("tensor1", 4),
            TensorType::new_float("tensor2", 4),
            TensorType::new_float("tensor3", 4),
        ));
        graph.register(Conv2dNode::new(
            "conv2d",
            TensorType::new_float("tensor2", 4),
            TensorType::new_float("tensor4", 4),
            TensorData::from([2f32]),
            None,
            Conv2dConfig::new(
                [3, 3],
                [3, 3],
                [1, 1],
                PaddingConfig2d::Valid,
                [1, 1],
                1,
                true,
            ),
        ));
        graph.register(MatmulNode::new(
            TensorType::new_float("tensor3", 4),
            TensorType::new_float("tensor4", 4),
            TensorType::new_float("output", 4),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["output".to_string()],
        );

        let expected = quote! {
            use burn::prelude::*;
            use burn::nn::PaddingConfig2d;
            use burn::nn::conv::Conv2d;
            use burn::nn::conv::Conv2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv2d: Conv2d<B>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model <B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let conv2d = Conv2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init(device);

                    Self {
                        conv2d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }
                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4> {
                    let tensor3 = tensor1.matmul(tensor2.clone());
                    let tensor4 = self.conv2d.forward(tensor2);
                    let output = tensor3.matmul(tensor4);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
