use super::{Node, NodeCodegen};
use crate::burn::{Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use std::sync::Arc;

#[derive(Clone)]
pub enum BinaryType {
    Add,
    Sub,
    Mul,
    Div,
    Equal,
    Powf,
    Powi,
    Min,
    Max,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
}

impl BinaryType {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            BinaryType::Add => "add",
            BinaryType::Sub => "sub",
            BinaryType::Mul => "mul",
            BinaryType::Div => "div",
            BinaryType::Equal => "equal",
            BinaryType::Powi => "powi",
            BinaryType::Powf => "powf",
            BinaryType::Min => "min_pair",
            BinaryType::Max => "max_pair",
            BinaryType::Greater => "greater",
            BinaryType::GreaterOrEqual => "greater_equal",
            BinaryType::Less => "lower",
            BinaryType::LessOrEqual => "lower_equal",
        }
    }
}

// Simple fn pointer that receive input as a token stream and return function call.
type FnPointer = Arc<dyn Fn(TokenStream, TokenStream) -> TokenStream>;

/// Node for all binary operators.
#[derive(Clone, new)]
pub struct BinaryNode {
    pub lhs: Type,
    pub rhs: Type,
    pub output: Type,
    pub binary_type: BinaryType,
    function: FnPointer,
}

impl std::fmt::Debug for BinaryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "BinaryNode {{ lhs: {:?}, rhs: {:?}, output: {:?}, name: {:?} }}",
                self.lhs,
                self.rhs,
                self.output,
                self.binary_type.as_str()
            )
            .as_str(),
        )
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for BinaryNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![self.lhs.clone(), self.rhs.clone()]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        // Get the lhs name in the form of token stream.
        let lhs = match &self.lhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor or scalar"),
        };

        // Get the rhs name in the form of token stream
        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor or scalar"),
        };

        let output = &self.output.name();
        let function = (self.function)(lhs, rhs);

        quote! {
            let #output = #function;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Binary(self)
    }
}

impl BinaryNode {
    pub(crate) fn add(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.add(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.add_scalar(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #rhs.add_scalar(#lhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs + #rhs },
            _ => panic!("Addition is supported for tensor and scalar only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Add, Arc::new(function))
    }

    pub(crate) fn sub(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.sub(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.sub_scalar(#rhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs - #rhs },
            (Type::Scalar(_), Type::Tensor(_)) => move |lhs, rhs| quote! { -#rhs.sub_scalar(#lhs) },
            _ => panic!("Subtraction is supported for tensor and scalar only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Sub, Arc::new(function))
    }

    pub(crate) fn mul(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.mul(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.mul_scalar(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #rhs.mul_scalar(#lhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs * #rhs },
            _ => panic!("Multiplication is supported for tensor and scalar only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Mul, Arc::new(function))
    }

    pub(crate) fn div(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.div(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.div_scalar(#rhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs / #rhs },
            _ => panic!("Division is supported for tensor and scalar only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Div, Arc::new(function))
    }

    pub(crate) fn equal(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.equal(#rhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs == #rhs },
            _ => panic!("Comparison is supported for tensor to tensor and scalar to scalar only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Equal, Arc::new(function))
    }
    pub(crate) fn powf(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.powf(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.powf_scalar(#rhs) },
            _ => panic!("pow is supported for tensor only"),
        };
        Self::new(lhs, rhs, output, BinaryType::Powf, Arc::new(function))
    }
    pub(crate) fn powi(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.powi(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.powi_scalar(#rhs) },
            _ => panic!("pow is supported for tensor only"),
        };
        Self::new(lhs, rhs, output, BinaryType::Powi, Arc::new(function))
    }

    pub(crate) fn min_pair(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.min_pair(#rhs) },
            _ => panic!("min_pair is supported for tensor only"),
        };
        Self::new(lhs, rhs, output, BinaryType::Min, Arc::new(function))
    }

    pub(crate) fn max_pair(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.max_pair(#rhs) },
            _ => panic!("max is supported for tensor only"),
        };
        Self::new(lhs, rhs, output, BinaryType::Max, Arc::new(function))
    }

    pub(crate) fn greater(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.greater(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => {
                move |lhs, rhs| quote! { #lhs.greater_elem(#rhs) }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L > R == R < L
                move |lhs, rhs| quote! { #rhs.lower_elem(#lhs) }
            }
            (lhs, rhs) => panic!("greater is not supported for {lhs:?} > {rhs:?}"),
        };
        Self::new(lhs, rhs, output, BinaryType::Greater, Arc::new(function))
    }

    pub(crate) fn greater_equal(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => {
                move |lhs, rhs| quote! { #lhs.greater_equal(#rhs) }
            }
            (Type::Tensor(_), Type::Scalar(_)) => {
                move |lhs, rhs| quote! { #lhs.greater_equal_elem(#rhs) }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L >= R == R <= L
                move |lhs, rhs| quote! { #rhs.lower_equal_elem(#lhs) }
            }
            (lhs, rhs) => panic!("greater_equal is not supported for {lhs:?} > {rhs:?}"),
        };
        Self::new(
            lhs,
            rhs,
            output,
            BinaryType::GreaterOrEqual,
            Arc::new(function),
        )
    }

    pub(crate) fn lower(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.lower(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.lower_elem(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L < R == R > L
                move |lhs, rhs| quote! { #rhs.greater_elem(#lhs) }
            }
            (lhs, rhs) => panic!("lower is not supported for {lhs:?} > {rhs:?}"),
        };
        Self::new(lhs, rhs, output, BinaryType::Less, Arc::new(function))
    }

    pub(crate) fn lower_equal(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #lhs.lower_equal(#rhs) },
            (Type::Tensor(_), Type::Scalar(_)) => {
                move |lhs, rhs| quote! { #lhs.lower_equal_elem(#rhs) }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L <= R == R >= L
                move |lhs, rhs| quote! { #rhs.greater_equal_elem(#lhs) }
            }
            (lhs, rhs) => panic!("lower_equal is not supported for {lhs:?} > {rhs:?}"),
        };
        Self::new(
            lhs,
            rhs,
            output,
            BinaryType::LessOrEqual,
            Arc::new(function),
        )
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::graph::BurnGraph;
    use crate::burn::node::test::assert_tokens;
    use crate::burn::node::tests::one_node_graph;
    use crate::burn::{ScalarKind, ScalarType, TensorType};

    macro_rules! test_binary_operator_on_tensors {
    ($operator:ident) => {{
      one_node_graph(
        BinaryNode::$operator(
          Type::Tensor(TensorType::new_float("tensor1", 4)),
          Type::Tensor(TensorType::new_float("tensor2", 4)),
          Type::Tensor(TensorType::new_float("tensor3", 4)),
        ),
        quote! {
            pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                let tensor3 = tensor1.$operator(tensor2);

                tensor3
            }
        },
        vec!["tensor1".to_string(), "tensor2".to_string()],
        vec!["tensor3".to_string()],
      );
    }};
  }

    macro_rules! test_binary_operator_on_tensor_and_scalar {
        ($operator:ident, $burn_operator:ident) => {{
            one_node_graph(
                BinaryNode::$operator(
                    Type::Tensor(TensorType::new_float("tensor1", 4)),
                    Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                    Type::Tensor(TensorType::new_float("tensor3", 4)),
                ),
                quote! {
                    pub fn forward(&self, scalar1: f32, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                        let tensor3 = tensor1.$burn_operator(scalar1);

                        tensor3
                    }
                },
                vec!["scalar1".to_string(), "tensor1".to_string()],
                vec!["tensor3".to_string()],
            );
        }};
    }

    macro_rules! test_binary_operator_on_scalar_and_scalar {
        ($operator:ident, $scalar_operator:tt) => {{
            one_node_graph(
                BinaryNode::$operator(
                    Type::Scalar(ScalarType::new("scalar1", ScalarKind::Float32)),
                    Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float32)),
                    Type::Scalar(ScalarType::new("scalar3", ScalarKind::Float32)),
                ),
                quote! {
                    pub fn forward(&self, scalar1: f32, scalar2: f32) -> f32 {
                        let scalar3 = scalar1 $scalar_operator scalar2;

                        scalar3
                    }
                },
                vec!["scalar1".to_string(), "scalar2".to_string()],
                vec!["scalar3".to_string()],
            );
        }};
    }

    #[test]
    fn test_binary_codegen_add() {
        test_binary_operator_on_tensors!(add);
    }

    #[test]
    fn test_binary_codegen_add_scalar() {
        test_binary_operator_on_tensor_and_scalar!(add, add_scalar);
    }

    #[test]
    fn test_binary_codegen_add_scalars() {
        test_binary_operator_on_scalar_and_scalar!(add, +);
    }

    #[test]
    fn test_binary_codegen_sub() {
        test_binary_operator_on_tensors!(sub);
    }

    #[test]
    fn test_binary_codegen_sub_scalar() {
        test_binary_operator_on_tensor_and_scalar!(sub, sub_scalar);
    }

    #[test]
    fn test_binary_codegen_sub_scalars() {
        test_binary_operator_on_scalar_and_scalar!(sub, -);
    }

    #[test]
    fn test_binary_codegen_mul() {
        test_binary_operator_on_tensors!(mul);
    }

    #[test]
    fn test_binary_codegen_mul_scalar() {
        test_binary_operator_on_tensor_and_scalar!(mul, mul_scalar);
    }

    #[test]
    fn test_binary_codegen_mul_scalars() {
        test_binary_operator_on_scalar_and_scalar!(mul, *);
    }
    #[test]
    fn test_binary_codegen_powi() {
        test_binary_operator_on_tensors!(powi);
    }

    #[test]
    fn test_binary_codegen_powf() {
        test_binary_operator_on_tensors!(powf);
    }

    #[test]
    fn test_binary_codegen_powi_scalar() {
        test_binary_operator_on_tensor_and_scalar!(powi, powi_scalar);
    }

    #[test]
    fn test_binary_codegen_powf_scalar() {
        test_binary_operator_on_tensor_and_scalar!(powf, powf_scalar);
    }

    #[test]
    fn test_binary_codegen_div() {
        test_binary_operator_on_tensors!(div);
    }

    #[test]
    fn test_binary_codegen_div_scalar() {
        test_binary_operator_on_tensor_and_scalar!(div, div_scalar);
    }

    #[test]
    fn test_binary_codegen_div_scalars() {
        test_binary_operator_on_scalar_and_scalar!(div, /);
    }

    #[test]
    fn test_binary_codegen_min() {
        test_binary_operator_on_tensors!(min_pair);
    }

    #[test]
    fn test_binary_codegen_max() {
        test_binary_operator_on_tensors!(max_pair);
    }

    #[test]
    fn test_binary_codegen_greater() {
        test_binary_operator_on_tensors!(greater);
    }

    #[test]
    fn test_binary_codegen_greater_scalar() {
        test_binary_operator_on_tensor_and_scalar!(greater, greater_elem);
    }

    #[test]
    fn test_binary_codegen_greater_or_equal() {
        test_binary_operator_on_tensors!(greater_equal);
    }

    #[test]
    fn test_binary_codegen_greater_or_equal_scalar() {
        test_binary_operator_on_tensor_and_scalar!(greater_equal, greater_equal_elem);
    }

    #[test]
    fn test_binary_codegen_less() {
        test_binary_operator_on_tensors!(lower);
    }

    #[test]
    fn test_binary_codegen_less_scalar() {
        test_binary_operator_on_tensor_and_scalar!(lower, lower_elem);
    }

    #[test]
    fn test_binary_codegen_less_or_equal() {
        test_binary_operator_on_tensors!(lower_equal);
    }

    #[test]
    fn test_binary_codegen_less_or_equal_scalar() {
        test_binary_operator_on_tensor_and_scalar!(lower_equal, lower_equal_elem);
    }

    #[test]
    fn test_binary_codegen_equal_tensors() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let node_gen = BinaryNode::equal(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            Type::Tensor(TensorType::new_bool("tensor3", 4)),
        );

        graph.register(node_gen);

        graph.register_input_output(
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        let expected = quote! {
            use burn::tensor::Bool;
            use burn::{
                module::Module,
                tensor::{backend::Backend, Tensor},
            };

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
                pub fn forward(
                    &self,
                    tensor1: Tensor<B, 4>,
                    tensor2: Tensor<B, 4>
                ) -> Tensor<B, 4, Bool> {
                    let tensor3 = tensor1.equal(tensor2);

                    tensor3
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_binary_codegen_equal_scalars() {
        test_binary_operator_on_scalar_and_scalar!(equal, ==);
    }
}
