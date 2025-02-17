use std::marker::PhantomData;

use super::{
    argmax::ArgMaxNode, avg_pool1d::AvgPool1dNode, avg_pool2d::AvgPool2dNode,
    batch_norm::BatchNormNode, binary::BinaryNode, clip::ClipNode, concat::ConcatNode,
    constant::ConstantNode, constant_of_shape::ConstantOfShapeNode, conv1d::Conv1dNode,
    conv2d::Conv2dNode, conv3d::Conv3dNode, conv_transpose_1d::ConvTranspose1dNode,
    conv_transpose_2d::ConvTranspose2dNode, conv_transpose_3d::ConvTranspose3dNode,
    dropout::DropoutNode, expand::ExpandNode, floor::FloorNode, gather::GatherNode,
    gather_elements::GatherElementsNode, global_avg_pool::GlobalAvgPoolNode,
    layer_norm::LayerNormNode, linear::LinearNode, mask_where::WhereNode, matmul::MatmulNode,
    max_pool1d::MaxPool1dNode, max_pool2d::MaxPool2dNode, mean::MeanNode, one_hot::OneHotNode,
    pad::PadNode, prelu::PReluNode, random_normal::RandomNormalNode,
    random_normal_like::RandomNormalLikeNode, random_uniform::RandomUniformNode,
    random_uniform_like::RandomUniformLikeNode, range::RangeNode, reshape::ReshapeNode,
    resize::ResizeNode, slice::SliceNode, split::SplitNode, squeeze::SqueezeNode, sum::SumNode,
    tile::TileNode, top_k::TopKNode, trilu::TriluNode, unary::UnaryNode, unsqueeze::UnsqueezeNode,
};
use crate::burn::{BurnImports, Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use serde::Serialize;

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
    ArgMax(ArgMaxNode),
    AvgPool1d(AvgPool1dNode),
    AvgPool2d(AvgPool2dNode),
    BatchNorm(BatchNormNode),
    Binary(BinaryNode),
    Clip(ClipNode),
    Concat(ConcatNode),
    Constant(ConstantNode),
    Conv1d(Conv1dNode),
    Conv2d(Conv2dNode),
    Conv3d(Conv3dNode),
    ConvTranspose1d(ConvTranspose1dNode),
    ConvTranspose2d(ConvTranspose2dNode),
    ConvTranspose3d(ConvTranspose3dNode),
    PRelu(PReluNode),
    Dropout(DropoutNode),
    Expand(ExpandNode),
    Floor(FloorNode),
    Gather(GatherNode),
    GatherElements(GatherElementsNode),
    GlobalAvgPool(GlobalAvgPoolNode),
    LayerNorm(LayerNormNode),
    Linear(LinearNode),
    Matmul(MatmulNode),
    MaxPool1d(MaxPool1dNode),
    MaxPool2d(MaxPool2dNode),
    Mean(MeanNode),
    OneHot(OneHotNode),
    Pad(PadNode),
    Range(RangeNode),
    Reshape(ReshapeNode),
    Resize(ResizeNode),
    Slice(SliceNode),
    Squeeze(SqueezeNode),
    Split(SplitNode),
    Sum(SumNode),
    Tile(TileNode),
    TopK(TopKNode),
    Trilu(TriluNode),
    Unary(UnaryNode),
    Unsqueeze(UnsqueezeNode),
    Where(WhereNode),
    RandomNormal(RandomNormalNode),
    RandomNormalLike(RandomNormalLikeNode),
    RandomUniform(RandomUniformNode),
    RandomUniformLike(RandomUniformLikeNode),
    ConstantOfShape(ConstantOfShapeNode),
    // For now, we have to keep the precision settings in order to correctly serialize the fields
    // into the right data types.
    _Unreachable(std::convert::Infallible, PhantomData<PS>),
}

macro_rules! match_all {
    ($self:expr, $func:expr) => {{
        #[allow(clippy::redundant_closure_call)]
        match $self {
            Node::ArgMax(node) => $func(node),
            Node::AvgPool1d(node) => $func(node),
            Node::AvgPool2d(node) => $func(node),
            Node::BatchNorm(node) => $func(node),
            Node::Binary(node) => $func(node),
            Node::Clip(node) => $func(node),
            Node::Concat(node) => $func(node),
            Node::Constant(node) => $func(node),
            Node::Conv1d(node) => $func(node),
            Node::Conv2d(node) => $func(node),
            Node::Conv3d(node) => $func(node),
            Node::ConvTranspose1d(node) => $func(node),
            Node::ConvTranspose2d(node) => $func(node),
            Node::ConvTranspose3d(node) => $func(node),
            Node::PRelu(node) => $func(node),
            Node::Dropout(node) => $func(node),
            Node::Expand(node) => $func(node),
            Node::Floor(node) => $func(node),
            Node::Gather(node) => $func(node),
            Node::GatherElements(node) => $func(node),
            Node::GlobalAvgPool(node) => $func(node),
            Node::LayerNorm(node) => $func(node),
            Node::Linear(node) => $func(node),
            Node::Matmul(node) => $func(node),
            Node::MaxPool1d(node) => $func(node),
            Node::MaxPool2d(node) => $func(node),
            Node::Mean(node) => $func(node),
            Node::OneHot(node) => $func(node),
            Node::Pad(node) => $func(node),
            Node::Range(node) => $func(node),
            Node::Reshape(node) => $func(node),
            Node::Resize(node) => $func(node),
            Node::Slice(node) => $func(node),
            Node::Squeeze(node) => $func(node),
            Node::Sum(node) => $func(node),
            Node::Tile(node) => $func(node),
            Node::TopK(node) => $func(node),
            Node::Trilu(node) => $func(node),
            Node::Unary(node) => $func(node),
            Node::Unsqueeze(node) => $func(node),
            Node::Where(node) => $func(node),
            Node::RandomNormal(node) => $func(node),
            Node::RandomNormalLike(node) => $func(node),
            Node::RandomUniform(node) => $func(node),
            Node::RandomUniformLike(node) => $func(node),
            Node::ConstantOfShape(node) => $func(node),
            Node::Split(node) => $func(node),
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
            Node::ArgMax(_) => "argmax",
            Node::AvgPool1d(_) => "avg_pool1d",
            Node::AvgPool2d(_) => "avg_pool2d",
            Node::BatchNorm(_) => "batch_norm",
            Node::Binary(binary) => binary.binary_type.as_str(),
            Node::Concat(_) => "concat",
            Node::Clip(_) => "clip",
            Node::Constant(_) => "constant",
            Node::Conv1d(_) => "conv1d",
            Node::Conv2d(_) => "conv2d",
            Node::Conv3d(_) => "conv3d",
            Node::ConvTranspose1d(_) => "conv_transpose1d",
            Node::ConvTranspose2d(_) => "conv_transpose2d",
            Node::ConvTranspose3d(_) => "conv_transpose3d",
            Node::PRelu(_) => "prelu",
            Node::Dropout(_) => "dropout",
            Node::Expand(_) => "expand",
            Node::Floor(_) => "floor",
            Node::Gather(_) => "gather",
            Node::GatherElements(_) => "gather_elements",
            Node::GlobalAvgPool(_) => "global_avg_pool",
            Node::LayerNorm(_) => "layer_norm",
            Node::Linear(_) => "linear",
            Node::Matmul(_) => "matmul",
            Node::MaxPool1d(_) => "max_pool1d",
            Node::MaxPool2d(_) => "max_pool2d",
            Node::Mean(_) => "mean",
            Node::OneHot(_) => "one_hot",
            Node::Pad(_) => "pad",
            Node::Range(_) => "range",
            Node::Reshape(_) => "reshape",
            Node::Resize(_) => "resize",
            Node::Slice(_) => "slice",
            Node::Squeeze(_) => "squeeze",
            Node::Sum(_) => "add",
            Node::Tile(_) => "tile",
            Node::TopK(_) => "top_k",
            Node::Trilu(_) => "trilu",
            Node::Unary(unary) => unary.kind.as_str(),
            Node::Unsqueeze(_) => "unsqueeze",
            Node::Where(_) => "where",
            Node::RandomNormal(_) => "random_normal",
            Node::RandomNormalLike(_) => "random_normal_like",
            Node::RandomUniform(_) => "random_uniform",
            Node::RandomUniformLike(_) => "random_uniform_like",
            Node::ConstantOfShape(_) => "constant_of_shape",
            Node::Split(_) => "split",
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
        graph::BurnGraph,
        node::{conv2d::Conv2dNode, matmul::MatmulNode, test::assert_tokens, NodeCodegen},
        BurnImports, TensorType,
    };
    use burn::{
        nn::conv::Conv2dConfig, nn::PaddingConfig2d, record::FullPrecisionSettings,
        tensor::TensorData,
    };
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
            Conv2dConfig::new([3, 3], [3, 3]).with_padding(PaddingConfig2d::Valid),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor4".to_string()],
        );

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::conv::Conv2dConfig;
            use burn::nn::conv::Conv2d;
            use burn::nn::PaddingConfig2d;

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
            Conv2dConfig::new([3, 3], [3, 3]).with_padding(PaddingConfig2d::Valid),
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
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
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
