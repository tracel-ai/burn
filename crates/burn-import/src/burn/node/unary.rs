use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, ToTokens, Type};
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
    // Input and output tensor types (required for codegen imports)
    Cast(Option<TensorKind>, Option<TensorKind>),
    Cos,
    Erf,
    Exp,
    Flatten,
    Gelu,
    LeakyRelu,
    Log,
    LogSoftmax,
    Neg,
    Not,
    ReduceMax,
    ReduceMin,
    ReduceMean,
    ReduceSum,
    Reciprocal,
    Relu,
    Shape,
    Sigmoid,
    Sin,
    Softmax,
    Sqrt,
    Tanh,
    Transpose,
    Sign,
}

impl UnaryNodeKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cast(..) => "cast",
            Self::Cos => "cos",
            Self::Erf => "erf",
            Self::Exp => "exp",
            Self::Flatten => "flatten",
            Self::Gelu => "gelu",
            Self::LeakyRelu => "leaky_relu",
            Self::Log => "log",
            Self::LogSoftmax => "log_softmax",
            Self::Neg => "neg",
            Self::Not => "not",
            Self::ReduceMax => "reduce_max",
            Self::ReduceMin => "reduce_min",
            Self::ReduceMean => "reduce_mean",
            Self::ReduceSum => "reduce_sum",
            Self::Reciprocal => "reciprocal",
            Self::Relu => "relu",
            Self::Shape => "shape",
            Self::Sigmoid => "sigmoid",
            Self::Sin => "sin",
            Self::Softmax => "softmax",
            Self::Sqrt => "sqrt",
            Self::Tanh => "tanh",
            Self::Transpose => "transpose",
            Self::Sign => "sign",
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
            UnaryNodeKind::Shape => {
                imports.register("burn::tensor::Int");
            }
            UnaryNodeKind::Not => {
                imports.register("burn::tensor::Bool");
            }
            UnaryNodeKind::Cast(Some(input_kind), Some(output_kind)) => {
                if input_kind == TensorKind::Bool || output_kind == TensorKind::Bool {
                    imports.register("burn::tensor::Bool");
                }
                if input_kind == TensorKind::Int || output_kind == TensorKind::Int {
                    imports.register("burn::tensor::Int");
                }
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

    pub(crate) fn leaky_relu(input: Type, output: Type, alpha: f64) -> Self {
        let alpha = alpha.to_tokens();
        let function = move |input| quote! { burn::tensor::activation::leaky_relu(#input, #alpha) };
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

    pub(crate) fn transpose(input: Type, output: Type, perm: Vec<i64>) -> Self {
        let perm = perm.to_tokens();
        let function = move |input| quote! { #input.permute(#perm) };
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

    pub(crate) fn sin(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.sin()};
        Self::new(input, output, UnaryNodeKind::Sin, Rc::new(function))
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

    pub(crate) fn not(input: Type, output: Type) -> Self {
        // Not ONNX operator is constrained to bool tensors, so no need to check the type.
        let function = move |input| quote! { #input.bool_not()};
        Self::new(input, output, UnaryNodeKind::Not, Rc::new(function))
    }

    /// Casts the input to the output type.
    pub(crate) fn cast(input: Type, output: Type) -> Self {
        match (input.clone(), output.clone()) {
            (Type::Scalar(input_scalar), Type::Scalar(output_scalar)) => {
                if input_scalar.kind == output_scalar.kind {
                    // If the input and output types are the same, we don't need to cast.
                    Self::new(
                        input,
                        output,
                        UnaryNodeKind::Cast(None, None),
                        Rc::new(|input| input),
                    )
                } else {
                    // If the input and output types are different, we need to cast.
                    let ty = output_scalar.ty();
                    Self::new(
                        input,
                        output,
                        UnaryNodeKind::Cast(None, None),
                        Rc::new(move |input| quote! { #input as #ty }),
                    )
                }
            }
            (Type::Tensor(input_tensor), Type::Tensor(output_tensor)) => {
                if input_tensor.kind == output_tensor.kind {
                    // If the input and output types are the same, we don't need to cast.
                    Self::new(
                        input,
                        output,
                        UnaryNodeKind::Cast(Some(input_tensor.kind), Some(output_tensor.kind)),
                        Rc::new(|input| input),
                    )
                } else {
                    // If the input and output types are different, we need to cast.
                    let function = match output_tensor.kind {
                        TensorKind::Bool => move |input| quote! { #input.bool()},
                        TensorKind::Int => move |input| quote! { #input.int()},
                        TensorKind::Float => move |input| quote! { #input.float()},
                    };

                    Self::new(
                        input,
                        output,
                        UnaryNodeKind::Cast(Some(input_tensor.kind), Some(output_tensor.kind)),
                        Rc::new(function),
                    )
                }
            }
            _ => panic!("output must be a tensor or scalar"),
        }
    }

    pub(crate) fn reduce_max(input: Type, output: Type, dim: Option<usize>) -> Self {
        if let Type::Tensor(ref tensor) = output {
            if let Some(dim) = dim {
                if tensor.kind == TensorKind::Bool {
                    // Max is only implemented on numeric tensors
                    panic!("ReduceMax is not supported for boolean");
                }

                // ReduceMax, keepdims=1, axes=[dim]
                let dim = dim.to_tokens();
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceMax,
                    Rc::new(move |input| quote! { #input.max_dim(#dim) }),
                )
            } else {
                // ReduceMax, keepdims=0, axes=None
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceMax,
                    Rc::new(move |input| quote! { #input.max() }),
                )
            }
        } else {
            panic!("ReduceMax only supports tensor output");
        }
    }

    pub(crate) fn reduce_min(input: Type, output: Type, dim: Option<usize>) -> Self {
        if let Type::Tensor(ref tensor) = output {
            if let Some(dim) = dim {
                if tensor.kind == TensorKind::Bool {
                    // Min is only implemented on numeric tensors
                    panic!("ReduceMin is not supported for boolean");
                }
                // ReduceMin, keepdims=1, axes=[dim]
                let dim = dim.to_tokens();
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceMin,
                    Rc::new(move |input| quote! { #input.min_dim(#dim) }),
                )
            } else {
                // ReduceMin, keepdims=0, axes=None
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceMin,
                    Rc::new(move |input| quote! { #input.min() }),
                )
            }
        } else {
            panic!("ReduceMin only supports tensor output");
        }
    }

    pub(crate) fn reduce_mean(input: Type, output: Type, dim: Option<usize>) -> Self {
        // ReduceMean is constrained to numeric tensors, so no need to check for bool.
        if let Type::Tensor(_) = output {
            if let Some(dim) = dim {
                // ReduceMean, keepdims=1, axes=[dim]
                let dim = dim.to_tokens();
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceMean,
                    Rc::new(move |input| quote! { #input.mean_dim(#dim) }),
                )
            } else {
                // ReduceMean, keepdims=0, axes=None
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceMean,
                    Rc::new(move |input| quote! { #input.mean() }),
                )
            }
        } else {
            panic!("ReduceMean only supports tensor output");
        }
    }

    pub(crate) fn reduce_sum(input: Type, output: Type, dim: Option<usize>) -> Self {
        if let Type::Tensor(ref tensor) = output {
            if let Some(dim) = dim {
                if tensor.kind == TensorKind::Bool {
                    // Sum is only implemented on numeric tensors
                    panic!("ReduceSum is not supported for boolean");
                }

                // ReduceSum, keepdims=1, axes=[dim]
                let dim = dim.to_tokens();
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceSum,
                    Rc::new(move |input| quote! { #input.sum_dim(#dim) }),
                )
            } else {
                // ReduceSum, keepdims=0, axes=None
                Self::new(
                    input,
                    output,
                    UnaryNodeKind::ReduceSum,
                    Rc::new(move |input| quote! { #input.sum() }),
                )
            }
        } else {
            panic!("ReduceSum only supports tensor output");
        }
    }

    pub(crate) fn shape(input: Type, output: Type, start_dim: usize, end_dim: usize) -> Self {
        // Shape as defined by the ONNX op should return a tensor because other ops
        // (e.g., Gather) will be used on a tensor
        let function = move |input| {
            quote! {
                Tensor::<B, 1, Int>::from_data(
                    burn::tensor::Data::from(&#input.dims()[#start_dim..#end_dim])
                        .from_usize::<i64>()
                        .convert::<burn::tensor::ops::IntElem<B>>(),
                    &#input.device(),
                )
            }
        };
        Self::new(input, output, UnaryNodeKind::Shape, Rc::new(function))
    }

    pub(crate) fn sign(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.sign()};
        Self::new(input, output, UnaryNodeKind::Sign, Rc::new(function))
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
    fn test_unary_codegen_leaky_relu() {
        one_node_graph(
            UnaryNode::leaky_relu(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                0.1,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::leaky_relu(tensor1, 0.1);

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
                vec![0, 3, 1, 2],
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.permute([0, 3, 1, 2]);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_reduce_max() {
        one_node_graph(
            UnaryNode::reduce_max(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                Some(1),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.max_dim(1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::reduce_max(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 1)),
                None,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                    let tensor2 = tensor1.max();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_reduce_min() {
        one_node_graph(
            UnaryNode::reduce_min(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                Some(1),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.min_dim(1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::reduce_min(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 1)),
                None,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                    let tensor2 = tensor1.min();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_reduce_mean() {
        one_node_graph(
            UnaryNode::reduce_mean(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                Some(1),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.mean_dim(1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::reduce_mean(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 1)),
                None,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                    let tensor2 = tensor1.mean();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_reduce_sum() {
        one_node_graph(
            UnaryNode::reduce_sum(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                Some(1),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.sum_dim(1);

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::reduce_sum(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 1)),
                None,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1> {
                    let tensor2 = tensor1.sum();

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
        one_node_graph(
            UnaryNode::cast(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_int("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Int> {
                    let tensor2 = tensor1.int();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
        one_node_graph(
            UnaryNode::cast(
                Type::Tensor(TensorType::new_int("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4, Int>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.float();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
        one_node_graph(
            UnaryNode::cast(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.bool();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
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
    fn test_unary_codegen_sin() {
        one_node_graph(
            UnaryNode::sin(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.sin();

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

    #[test]
    fn test_unary_codegen_not() {
        one_node_graph(
            UnaryNode::not(
                Type::Tensor(TensorType::new_bool("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4, Bool>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.bool_not();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_shape() {
        one_node_graph(
            UnaryNode::shape(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_int("tensor2", 1)),
                1,
                3,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 1, Int> {
                    let tensor2 = Tensor::<B, 1, Int>::from_data(
                        burn::tensor::Data::from(&tensor1.dims()[1usize..3usize])
                            .from_usize::<i64>()
                            .convert::<burn::tensor::ops::IntElem<B>>(),
                        &tensor1.device(),
                    );

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }

    #[test]
    fn test_unary_sign_tensor() {
        one_node_graph(
            UnaryNode::sign(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.sign();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }
}
