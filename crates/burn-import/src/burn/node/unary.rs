use super::{Node, NodeCodegen};
use crate::burn::{BurnImports, Scope, TensorKind, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::is_inf::IsInfConfig;
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
    Abs,
    Cos,
    Cosh,
    Erf,
    Exp,
    Flatten,
    Gelu,
    LeakyRelu,
    HardSigmoid,
    IsInf,
    IsNaN,
    Log,
    LogSoftmax,
    Neg,
    Not,
    Reciprocal,
    Relu,
    Shape,
    Sigmoid,
    Sin,
    Sinh,
    Softmax,
    Sqrt,
    Tan,
    Tanh,
    Transpose,
    Sign,
    Size,
}

impl UnaryNodeKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cast(..) => "cast",
            Self::Abs => "abs",
            Self::Cos => "cos",
            Self::Cosh => "cosh",
            Self::Erf => "erf",
            Self::Exp => "exp",
            Self::Flatten => "flatten",
            Self::Gelu => "gelu",
            Self::LeakyRelu => "leaky_relu",
            Self::HardSigmoid => "hard_sigmoid",
            Self::IsInf => "is_inf",
            Self::IsNaN => "is_nan",
            Self::Log => "log",
            Self::LogSoftmax => "log_softmax",
            Self::Neg => "neg",
            Self::Not => "not",
            Self::Reciprocal => "reciprocal",
            Self::Relu => "relu",
            Self::Shape => "shape",
            Self::Sigmoid => "sigmoid",
            Self::Sin => "sin",
            Self::Sinh => "sinh",
            Self::Softmax => "softmax",
            Self::Sqrt => "sqrt",
            Self::Tan => "tan",
            Self::Tanh => "tanh",
            Self::Transpose => "transpose",
            Self::Sign => "sign",
            Self::Size => "size",
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
            Type::Shape(shape) => {
                let name = &shape.name;
                quote! { #name }
            }
            _ => panic!("Input must be a tensor, scalar, or shape"),
        };

        let output = &self.output.name();
        let function = (self.function)(input);

        match &self.output {
            Type::Shape(shape_type) => {
                let dim = shape_type.rank.to_tokens();

                // Shape operations now return i64 directly from the shape function
                quote! {
                    let #output: [i64;#dim] = #function;
                }
            }
            _ => {
                quote! {
                    let #output = #function;
                }
            }
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
                imports.register("alloc::vec::Vec");
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

    pub(crate) fn flatten(input: Type, output: Type, axis: usize) -> Self {
        if axis == 0 {
            let function = move |input| quote! {#input.reshape::<2>([1, -1])};
            Self::new(input, output, UnaryNodeKind::Flatten, Rc::new(function))
        } else {
            let axis = axis.to_tokens();
            let function = move |input| {
                quote! {
                    {
                        let leading_dim = #input.shape().dims[..#axis].iter().product::<usize>() as i32;
                        #input.reshape::<2, _>([leading_dim, -1])
                    };
                }
            };
            Self::new(input, output, UnaryNodeKind::Flatten, Rc::new(function))
        }
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

    pub(crate) fn hard_sigmoid(input: Type, output: Type, alpha: f64, beta: f64) -> Self {
        let alpha = alpha.to_tokens();
        let beta = beta.to_tokens();
        let function =
            move |input| quote! { burn::tensor::activation::hard_sigmoid(#input, #alpha, #beta) };
        Self::new(input, output, UnaryNodeKind::HardSigmoid, Rc::new(function))
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

    pub(crate) fn abs(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.abs()};
        Self::new(input, output, UnaryNodeKind::Abs, Rc::new(function))
    }

    pub(crate) fn tan(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.tan()};
        Self::new(input, output, UnaryNodeKind::Tan, Rc::new(function))
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

    pub(crate) fn cosh(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.cosh()};
        Self::new(input, output, UnaryNodeKind::Cosh, Rc::new(function))
    }

    pub(crate) fn sin(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.sin()};
        Self::new(input, output, UnaryNodeKind::Sin, Rc::new(function))
    }

    pub(crate) fn sinh(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.sinh()};
        Self::new(input, output, UnaryNodeKind::Sinh, Rc::new(function))
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

    pub(crate) fn shape(input: Type, output: Type, start_dim: usize, end_dim: usize) -> Self {
        let start_dim_tok = start_dim.to_tokens();
        let end_dim_tok = end_dim.to_tokens();

        let function: Rc<dyn Fn(TokenStream) -> TokenStream> = match &input {
            Type::Tensor(_) => Rc::new(move |input| {
                quote! {
                    #input.dims()[#start_dim_tok..#end_dim_tok]
                        .iter()
                        .map(|&x| x as i64)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                }
            }),
            Type::Shape(shape_type) => {
                // If input is already a shape array [i64; N], the Shape operation
                // returns the dimensionality of the shape (which is N) as a Shape(1) array
                // This matches the ONNX semantics where Shape of a shape gives you the rank
                let rank_value = shape_type.rank as i64;
                Rc::new(move |_input| {
                    quote! { [#rank_value] }
                })
            }
            _ => panic!("Shape operation only supports Tensor or Shape inputs"),
        };
        Self::new(input, output, UnaryNodeKind::Shape, function)
    }

    pub(crate) fn sign(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.sign()};
        Self::new(input, output, UnaryNodeKind::Sign, Rc::new(function))
    }

    pub(crate) fn size(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.shape.num_elements()};
        Self::new(input, output, UnaryNodeKind::Size, Rc::new(function))
    }

    pub(crate) fn is_inf(input: Type, output: Type, config: IsInfConfig) -> Self {
        let function = match &output {
            Type::Scalar(_) => match (config.detect_negative, config.detect_positive) {
                (true, true) => move |input| quote! { #input.is_infinite() },
                (false, true) => {
                    move |input| quote! { #input.is_infinite() && #input.is_sign_positive() }
                }
                (true, false) => {
                    move |input| quote! { #input.is_infinite() && #input.is_sign_negative() }
                }
                (false, false) => move |_| quote! { false },
            },
            Type::Tensor(_) => match (config.detect_negative, config.detect_positive) {
                (true, true) => move |input| quote! { #input.is_inf() },
                (false, true) => {
                    move |input| quote! { #input.clone().is_inf().bool_and(#input.greater_elem(0.0)) }
                }
                (true, false) => {
                    move |input| quote! { #input.clone().is_inf().bool_and(#input.lower_elem(0.0)) }
                }
                (false, false) => move |input| quote! { #input.zeros_like().bool() },
            },
            v => panic!("IsInf only supports scalar or tensor outputs, but got: {v:?}"),
        };
        Self::new(input, output, UnaryNodeKind::IsInf, Rc::new(function))
    }

    pub(crate) fn is_nan(input: Type, output: Type) -> Self {
        let function = move |input| quote! { #input.is_nan() };
        Self::new(input, output, UnaryNodeKind::IsNaN, Rc::new(function))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::node::tests::one_node_graph;
    use crate::burn::{ScalarKind, ScalarType, ShapeType, TensorType};

    #[test]
    fn test_unary_codegen_flatten() {
        one_node_graph(
            UnaryNode::flatten(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 2)),
                1,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 2> {
                    let tensor2 = {
                        let leading_dim = tensor1.shape().dims[..1].iter().product::<usize>() as i32;
                        tensor1.reshape::<2, _>([leading_dim, -1])
                    };

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
    fn test_unary_codegen_hard_sigmoid() {
        one_node_graph(
            UnaryNode::hard_sigmoid(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                0.2,
                0.5,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = burn::tensor::activation::hard_sigmoid(tensor1, 0.2, 0.5);

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
    fn test_unary_codegen_tan() {
        one_node_graph(
            UnaryNode::tan(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.tan();

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
                Type::Shape(ShapeType::new("shape1", 2)),
                1,
                3,
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> [i64; 2] {
                    let shape1: [i64; 2] = tensor1.dims()[1..3]
                        .iter()
                        .map(|&x| x as i64)
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap();

                    shape1
                }
            },
            vec!["tensor1".to_string()],
            vec!["shape1".to_string()],
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

    #[test]
    fn test_unary_codegen_size() {
        one_node_graph(
            UnaryNode::size(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Int64)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> i64 {
                    let scalar1 = tensor1.shape.num_elements();

                    scalar1
                }
            },
            vec!["tensor1".to_string()],
            vec!["scalar1".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_is_inf() {
        one_node_graph(
            UnaryNode::is_inf(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                IsInfConfig::new(true, true),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.is_inf();
                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                IsInfConfig::new(false, true),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.clone().is_inf().bool_and(tensor1.greater_elem(0.0));
                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                IsInfConfig::new(true, false),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.clone().is_inf().bool_and(tensor1.lower_elem(0.0));
                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                IsInfConfig::new(false, false),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.zeros_like().bool();
                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                IsInfConfig::new(true, true),
            ),
            quote! {
                pub fn forward(&self, scalar1: f32) -> bool {
                    let scalar2 = scalar1.is_infinite();
                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                IsInfConfig::new(false, true),
            ),
            quote! {
                pub fn forward(&self, scalar1: f32) -> bool {
                    let scalar2 = scalar1.is_infinite() && scalar1.is_sign_positive();
                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                IsInfConfig::new(true, false),
            ),
            quote! {
                pub fn forward(&self, scalar1: f32) -> bool {
                    let scalar2 = scalar1.is_infinite() && scalar1.is_sign_negative();
                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_inf(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                IsInfConfig::new(false, false),
            ),
            quote! {
                pub fn forward(&self, scalar1: f32) -> bool {
                    let scalar2 = false;
                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_is_nan() {
        one_node_graph(
            UnaryNode::is_nan(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4, Bool> {
                    let tensor2 = tensor1.is_nan();
                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );

        one_node_graph(
            UnaryNode::is_nan(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
            ),
            quote! {
                pub fn forward(&self, scalar1: f32) -> bool {
                    let scalar2 = scalar1.is_nan();
                    scalar2
                }
            },
            vec!["scalar1".to_string()],
            vec!["scalar2".to_string()],
        );
    }

    #[test]
    fn test_unary_codegen_abs() {
        one_node_graph(
            UnaryNode::abs(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.abs();

                    tensor2
                }
            },
            vec!["tensor1".to_string()],
            vec!["tensor2".to_string()],
        );
    }
}
