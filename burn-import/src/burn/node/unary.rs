use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use std::rc::Rc;

// Simple fn pointer that receive input as a token stream and return function call.
type FnPointer = Rc<dyn Fn(TokenStream) -> TokenStream>;

/// Node for all unary operators.
#[derive(Clone, new)]
pub struct UnaryNode {
    pub input: Type,
    pub output: Type,
    pub kind: UnaryNodeKind,
    function: FnPointer,
}

/// Type of unary node.
#[derive(Clone)]
pub enum UnaryNodeKind {
    Cast,
    Cos,
    Erf,
    Exp,
    Flatten,
    Gelu,
    Log,
    LogSoftmax,
    Neg,
    Reciprocal,
    Relu,
    Sigmoid,
    Softmax,
    Sqrt,
    Tanh,
    Transpose,
}

impl UnaryNodeKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cast => "cast",
            Self::Cos => "cos",
            Self::Erf => "erf",
            Self::Exp => "exp",
            Self::Flatten => "flatten",
            Self::Gelu => "gelu",
            Self::Log => "log",
            Self::LogSoftmax => "log_softmax",
            Self::Neg => "neg",
            Self::Reciprocal => "reciprocal",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Softmax => "softmax",
            Self::Sqrt => "sqrt",
            Self::Tanh => "tanh",
            Self::Transpose => "transpose",
        }
    }
}

impl std::fmt::Debug for UnaryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "UnaryNode {{ input: {:?}, output: {:?}, name: {} }}",
                self.input,
                self.output,
                self.kind.as_str()
            )
            .as_str(),
        )
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for UnaryNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.input.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        // Get the lhs name in the form of token stream.
        let input = match &self.input {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor or scalar"),
        };

        // let input = scope.tensor_use_owned(&self.input, node_position);
        let output = &self.output.name();
        let function = (self.function)(input);

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Unary(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register the imports depending on the kind of the node.
        match self.kind {
            UnaryNodeKind::Neg => {
                imports.register("core::ops::Neg");
            }
            _ => {}
        }
    }
}

impl UnaryNode {
    pub(crate) fn erf(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.erf() };
        Self::new(input, output, UnaryNodeKind::Erf, Rc::new(function))
    }

    pub(crate) fn flatten(input: Type, output: Type, start_dim: usize, end_dim: usize) -> Self {
        let start_dim = start_dim.to_tokens();
        let end_dim = end_dim.to_tokens();
        let function = move |input| quote! { #input.flatten(#start_dim, #end_dim) };

        Self::new(input, output, UnaryNodeKind::Flatten, Rc::new(function))
    }

    pub(crate) fn relu(input: Type, output: Type) -> Self {
        let function = move |input| quote! { burn::tensor::activation::relu(#input) };
        Self::new(input, output, UnaryNodeKind::Relu, Rc::new(function))
    }

    pub(crate) fn sigmoid(input: Type, output: Type) -> Self {
        let function = move |input| quote! { burn::tensor::activation::sigmoid(#input) };
        Self::new(input, output, UnaryNodeKind::Sigmoid, Rc::new(function))
    }

    pub(crate) fn log_softmax(input: Type, output: Type, dim: usize) -> Self {
        let dim = dim.to_tokens();
        let function = move |input| quote! { burn::tensor::activation::log_softmax(#input, #dim) };
        Self::new(input, output, UnaryNodeKind::LogSoftmax, Rc::new(function))
    }

    pub(crate) fn softmax(input: Type, output: Type, dim: usize) -> Self {
        let dim = dim.to_tokens();
        let function = move |input| quote! { burn::tensor::activation::softmax(#input, #dim) };
        Self::new(input, output, UnaryNodeKind::Softmax, Rc::new(function))
    }

    pub(crate) fn sqrt(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.sqrt()};
        Self::new(input, output, UnaryNodeKind::Sqrt, Rc::new(function))
    }

    pub(crate) fn tanh(input: Type, output: Type) -> Self {
        let function = move |input| quote! { burn::tensor::activation::tanh(#input)};
        Self::new(input, output, UnaryNodeKind::Tanh, Rc::new(function))
    }

    pub(crate) fn transpose(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.transpose() };
        Self::new(input, output, UnaryNodeKind::Transpose, Rc::new(function))
    }

    pub(crate) fn reciprocal(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.recip() };
        Self::new(input, output, UnaryNodeKind::Reciprocal, Rc::new(function))
    }

    pub(crate) fn cos(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.cos()};
        Self::new(input, output, UnaryNodeKind::Cos, Rc::new(function))
    }

    pub(crate) fn exp(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.exp()};
        Self::new(input, output, UnaryNodeKind::Exp, Rc::new(function))
    }

    pub(crate) fn gelu(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.gelu()};
        Self::new(input, output, UnaryNodeKind::Gelu, Rc::new(function))
    }

    pub(crate) fn log(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.log()};
        Self::new(input, output, UnaryNodeKind::Log, Rc::new(function))
    }

    pub(crate) fn neg(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.neg()};
        Self::new(input, output, UnaryNodeKind::Neg, Rc::new(function))
    }

    /// Casts the input to the output type.
    ///
    /// Currently this function only supports the following conversions:
    /// 1) scalar -> scalar
    ///
    /// TODO: Implement the following conversions:
    /// 2) tensor int -> tensor float
    /// 3) tensor float -> tensor int
    /// 4) tensor -> scalar
    /// 5) scalar -> tensor
    pub(crate) fn cast(input: Type, output: Type) -> Self {
        match (input.clone(), output.clone()) {
            (Type::Scalar(input_scalar), Type::Scalar(output_scalar)) => {
                if input_scalar.kind == output_scalar.kind {
                    // If the input and output types are the same, we don't need to cast.
                    Self::new(input, output, UnaryNodeKind::Cast, Rc::new(|input| input))
                } else {
                    // If the input and output types are different, we need to cast.
                    let ty = output_scalar.ty();
                    Self::new(
                        input,
                        output,
                        UnaryNodeKind::Cast,
                        Rc::new(move |input| quote! { #input as #ty }),
                    )
                }
            }
            (Type::Tensor(_input_tensor), Type::Tensor(_output_tensor)) => {
                // TODO: Implement this after tensor Int is implemented (@antimora 8/2/2023)
                // TODO: If the input is scalar and the output type is a tensor,
                // we should generate another code block. (@antimora 8/4/2023)
                // Tensor::from_data(Data::from([#input]).convert()).unsqueeze();
                todo!()
            }

            _ => panic!("output must be a tensor"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::node::tests::one_node_graph;
    use crate::burn::{ScalarKind, ScalarType, TensorType};

    #[test]
    fn test_unary_codegen_flatten() {
        one_node_graph(
            UnaryNode::flatten(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                1,
                2,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.flatten(1, 2);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_erf() {
        one_node_graph(
            UnaryNode::erf(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.erf();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_relu() {
        one_node_graph(
            UnaryNode::relu(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::relu(tensor1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_sigmoid() {
        one_node_graph(
            UnaryNode::sigmoid(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::sigmoid(tensor1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_log_softmax() {
        one_node_graph(
            UnaryNode::log_softmax(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                1,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::log_softmax(tensor1, 1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_softmax() {
        one_node_graph(
            UnaryNode::softmax(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                1,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::softmax(tensor1, 1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_tanh() {
        one_node_graph(
            UnaryNode::tanh(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::tanh(tensor1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_transpose() {
        one_node_graph(
            UnaryNode::transpose(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.transpose();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_reciprocal() {
        one_node_graph(
            UnaryNode::reciprocal(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.recip();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_cast() {
        one_node_graph(
            UnaryNode::cast(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float64)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float32)),
            ),
            quote! {
                pub fn forward(&self, scalar1: f64) -> f32 {
                    let scalar2 = scalar1 as f32;

                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );
        one_node_graph(
            UnaryNode::cast(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float64)),
            ),
            quote! {
                pub fn forward(&self, scalar1: f32) -> f64 {
                    let scalar2 = scalar1 as f64;

                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_cos() {
        one_node_graph(
            UnaryNode::cos(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.cos();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_exp() {
        one_node_graph(
            UnaryNode::exp(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.exp();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_gelu() {
        one_node_graph(
            UnaryNode::gelu(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.gelu();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_log() {
        one_node_graph(
            UnaryNode::log(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.log();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_neg_scalar() {
        one_node_graph(
            UnaryNode::neg(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float64)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float64)),
            ),
            quote! {
                pub fn forward(&self, scalar1: f64) -> f64 {
                    let scalar2 = scalar1.neg();

                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );
    }

    #[test]
    fn test_unary_neg_tensor() {
        one_node_graph(
            UnaryNode::neg(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.neg();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }
}
