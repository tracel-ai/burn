use proc_macro2::TokenStream;
use serde::Serialize;

use burn::record::PrecisionSettings;

use crate::burn::{BurnImports, Scope, Type};

pub type SerializationBackend = burn_ndarray::NdArray<f32>;

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

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

// Note: Node enum and imports are now generated in registry.rs
// All node-specific imports are handled by the registry

// Import the generated Node enum and supporting code from registry
use super::registry::Node;

// Helper macro for dispatching on Node enum (more flexible than dispatch_node!)
macro_rules! match_all {
    ($self:expr, $func:expr) => {{
        #[allow(clippy::redundant_closure_call)]
        match $self {
            Node::Abs(node) => $func(node),
            Node::Add(node) => $func(node),
            Node::ArgMax(node) => $func(node),
            Node::ArgMin(node) => $func(node),
            Node::Attention(node) => $func(node),
            Node::AveragePool1d(node) => $func(node),
            Node::AveragePool2d(node) => $func(node),
            Node::BatchNormalization(node) => $func(node),
            Node::Bernoulli(node) => $func(node),
            Node::And(node) => $func(node),
            Node::Or(node) => $func(node),
            Node::Xor(node) => $func(node),
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
            Node::GreaterOrEqual(node) => $func(node),
            Node::GlobalAveragePool(node) => $func(node),
            Node::GroupNormalization(node) => $func(node),
            Node::HardSigmoid(node) => $func(node),
            Node::Identity(node) => $func(node),
            Node::InstanceNormalization(node) => $func(node),
            Node::IsInf(node) => $func(node),
            Node::IsNaN(node) => $func(node),
            Node::LayerNormalization(node) => $func(node),
            Node::LeakyRelu(node) => $func(node),
            Node::Linear(node) => $func(node),
            Node::Log(node) => $func(node),
            Node::LogSoftmax(node) => $func(node),
            Node::Less(node) => $func(node),
            Node::LessOrEqual(node) => $func(node),
            Node::MatMul(node) => $func(node),
            Node::MatMulInteger(node) => $func(node),
            Node::Max(node) => $func(node),
            Node::MaxPool1d(node) => $func(node),
            Node::MaxPool2d(node) => $func(node),
            Node::Mean(node) => $func(node),
            Node::Min(node) => $func(node),
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
            Node::ReduceMax(node) => $func(node),
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
            Node::_Unreachable(_, _) => unimplemented!(),
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
