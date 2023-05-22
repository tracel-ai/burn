use super::{
    batch_norm::BatchNormNode, conv2d::Conv2dNode, flatten::FlattenNode, linear::LinearNode,
    log_softmax::LogSoftmaxNode, matmul::MatmulNode, relu::ReLUNode,
};
use crate::burn::{BurnImports, Scope, Type};
use burn::record::PrecisionSettings;
use burn_ndarray::NdArrayBackend;
use proc_macro2::TokenStream;
use serde::Serialize;

/// Backend used for serialization.
pub type SerializationBackend = NdArrayBackend<f32>;

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

    /// (Optional) Declare how the parameters are initialized with and without a record.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_init(&self, _with_record: bool) -> Option<TokenStream> {
        None
    }

    /// (Optional) Declare how the parameters are serialized in a record.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        panic!("Serialization should be implemented when field_type is not None.");
    }
}

#[derive(Debug)]
pub enum Node<PS: PrecisionSettings> {
    Matmul(MatmulNode),
    Conv2d(Conv2dNode<PS>),
    Linear(LinearNode<PS>),
    BatchNorm(BatchNormNode<PS>),
    ReLU(ReLUNode),
    Flatten(FlattenNode),
    LogSoftmax(LogSoftmaxNode),
}

macro_rules! match_all {
    ($self:expr, $func:expr) => {{
        match $self {
            Node::Matmul(node) => $func(node),
            Node::Conv2d(node) => $func(node),
            Node::Linear(node) => $func(node),
            Node::BatchNorm(node) => $func(node),
            Node::ReLU(node) => $func(node),
            Node::Flatten(node) => $func(node),
            Node::LogSoftmax(node) => $func(node),
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

    fn field_init(&self, with_record: bool) -> Option<TokenStream> {
        match_all!(self, |node| NodeCodegen::<PS>::field_init(
            node,
            with_record
        ))
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
mod tests {
    use crate::burn::{
        graph::BurnGraph,
        node::{conv2d::Conv2dNode, matmul::MatmulNode, test::assert_tokens},
        TensorType,
    };
    use burn::{nn::conv::Conv2dConfig, record::FullPrecisionSettings, tensor::Data};
    use quote::quote;

    #[test]
    fn test_codegen_two_nodes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(MatmulNode::new(
            TensorType::new("tensor1", 4),
            TensorType::new("tensor2", 4),
            TensorType::new("tensor3", 4),
        ));
        graph.register(Conv2dNode::new(
            "conv2d",
            TensorType::new("tensor3", 4),
            TensorType::new("tensor4", 4),
            Data::from([2.]).serialize(),
            None,
            Conv2dConfig::new([3, 3], [3, 3]),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::conv::Conv2d;
            use burn::nn::conv::Conv2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv2d: Conv2d<B>,
            }

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let conv2d = Conv2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init_with(record.conv2d);

                    Self {
                        conv2d,
                    }
                }

                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
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
            TensorType::new("tensor1", 4),
            TensorType::new("tensor2", 4),
            TensorType::new("tensor3", 4),
        ));
        graph.register(Conv2dNode::new(
            "conv2d",
            TensorType::new("tensor2", 4),
            TensorType::new("tensor4", 4),
            Data::from([2.]).serialize(),
            None,
            Conv2dConfig::new([3, 3], [3, 3]),
        ));
        graph.register(MatmulNode::new(
            TensorType::new("tensor3", 4),
            TensorType::new("tensor4", 4),
            TensorType::new("output", 4),
        ));

        let expected = quote! {
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };
            use burn::nn::conv::Conv2d;
            use burn::nn::conv::Conv2dConfig;

            #[derive(Module, Debug)]
            pub struct Model <B: Backend> {
                conv2d: Conv2d<B>,
            }

            impl<B: Backend> Model <B> {
                pub fn new_with(record: ModelRecord<B>) -> Self {
                    let conv2d = Conv2dConfig::new([3, 3], [3, 3])
                        .with_stride([1, 1])
                        .with_dilation([1, 1])
                        .with_groups(1)
                        .with_bias(true)
                        .init_with(record.conv2d);

                    Self {
                        conv2d,
                    }
                }

                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
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
