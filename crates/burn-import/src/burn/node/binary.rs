use super::{Node, NodeCodegen};
use crate::burn::{ScalarKind, Scope, TensorKind, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;
use std::sync::Arc;

#[derive(Clone, PartialEq)]
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
    And,
    Or,
    Xor,
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
            BinaryType::And => "and",
            BinaryType::Or => "or",
            BinaryType::Xor => "xor",
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
            Type::Shape(shape) => {
                let name = shape.name.clone();
                quote! { #name }
            }
            _ => panic!("lhs must be a tensor, scalar, or shape"),
        };

        // Get the rhs name in the form of token stream
        let rhs = match &self.rhs {
            Type::Tensor(tensor) => scope.tensor_use_owned(tensor, node_position),
            Type::Scalar(scalar) => {
                let name = scalar.name.clone();
                quote! { #name }
            }
            Type::Shape(shape) => {
                let name = shape.name.clone();
                quote! { #name }
            }
            _ => panic!("rhs must be a tensor, scalar, or shape"),
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
    fn create_broadcast_function(
        op_name: &'static str,
        lhs_rank: usize,
        rhs_rank: usize,
    ) -> FnPointer {
        use quote::format_ident;

        if lhs_rank == rhs_rank {
            Arc::new(move |lhs, rhs| {
                let op = format_ident!("{}", op_name);
                quote! { #lhs.#op(#rhs) }
            })
        } else if lhs_rank > rhs_rank {
            // Broadcast rhs to match lhs rank by adding leading dimensions
            Arc::new(move |lhs, rhs| {
                let op = format_ident!("{}", op_name);
                // Need to add (lhs_rank - rhs_rank) dimensions at the beginning
                let num_dims = lhs_rank - rhs_rank;
                let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                let dims_tokens = quote! { &[#(#dims),*] };
                quote! { #lhs.#op(#rhs.unsqueeze_dims(#dims_tokens)) }
            })
        } else {
            // Broadcast lhs to match rhs rank by adding leading dimensions
            Arc::new(move |lhs, rhs| {
                let op = format_ident!("{}", op_name);
                // Need to add (rhs_rank - lhs_rank) dimensions at the beginning
                let num_dims = rhs_rank - lhs_rank;
                let dims: Vec<isize> = (0..num_dims).map(|i| i as isize).collect();
                let dims_tokens = quote! { &[#(#dims),*] };
                quote! { #lhs.unsqueeze_dims(#dims_tokens).#op(#rhs) }
            })
        }
    }

    pub(crate) fn add(lhs: Type, rhs: Type, output: Type) -> Self {
        log::debug!("BinaryNode::add called with lhs: {lhs:?}, rhs: {rhs:?}, output: {output:?}");
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Add,
                    Self::create_broadcast_function("add", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.add_scalar(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #rhs.add_scalar(#lhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs + #rhs },
            (Type::Shape(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                            *result_item = result_item.saturating_add(*rhs_item);
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Scalar(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for result_item in result.iter_mut() {
                            *result_item = result_item.saturating_add(#rhs as i64);
                        }
                        result
                    }
                }
            },
            (Type::Scalar(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #rhs;
                        for result_item in result.iter_mut() {
                            *result_item = result_item.saturating_add(#lhs as i64);
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).add(#rhs)
                    }
                }
            }
            (Type::Tensor(_), Type::Shape(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        #lhs.add(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                    }
                }
            }
            _ => panic!("Addition is supported for tensor, scalar, and shape types only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Add, Arc::new(function))
    }

    pub(crate) fn sub(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Sub,
                    Self::create_broadcast_function("sub", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.sub_scalar(#rhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs - #rhs },
            (Type::Scalar(_), Type::Tensor(_)) => move |lhs, rhs| quote! { -#rhs.sub_scalar(#lhs) },
            (Type::Shape(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                            *result_item = result_item.saturating_sub(*rhs_item);
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Scalar(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for result_item in result.iter_mut() {
                            *result_item = result_item.saturating_sub(#rhs as i64);
                        }
                        result
                    }
                }
            },
            (Type::Scalar(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #rhs;
                        for result_item in result.iter_mut() {
                            *result_item = (#lhs as i64).saturating_sub(*result_item);
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).sub(#rhs)
                    }
                }
            }
            (Type::Tensor(_), Type::Shape(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        #lhs.sub(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                    }
                }
            }
            _ => panic!("Subtraction is supported for tensor, scalar, and shape types only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Sub, Arc::new(function))
    }

    pub(crate) fn mul(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Mul,
                    Self::create_broadcast_function("mul", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.mul_scalar(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => move |lhs, rhs| quote! { #rhs.mul_scalar(#lhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs * #rhs },
            (Type::Shape(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                            *result_item = result_item.saturating_mul(*rhs_item);
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Scalar(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for result_item in result.iter_mut() {
                            *result_item = result_item.saturating_mul(#rhs as i64);
                        }
                        result
                    }
                }
            },
            (Type::Scalar(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #rhs;
                        for result_item in result.iter_mut() {
                            *result_item = result_item.saturating_mul(#lhs as i64);
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).mul(#rhs)
                    }
                }
            }
            (Type::Tensor(_), Type::Shape(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        #lhs.mul(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                    }
                }
            }
            _ => panic!("Multiplication is supported for tensor, scalar, and shape types only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Mul, Arc::new(function))
    }

    pub(crate) fn div(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Div,
                    Self::create_broadcast_function("div", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.div_scalar(#rhs) },
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs / #rhs },
            (Type::Shape(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                            *result_item = if *rhs_item != 0 { *result_item / *rhs_item } else { *result_item };
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Scalar(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for result_item in result.iter_mut() {
                            *result_item = if #rhs as i64 != 0 { *result_item / (#rhs as i64) } else { *result_item };
                        }
                        result
                    }
                }
            },
            (Type::Scalar(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #rhs;
                        for result_item in result.iter_mut() {
                            *result_item = if *result_item != 0 { (#lhs as i64) / *result_item } else { (#lhs as i64) };
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).div(#rhs)
                    }
                }
            }
            (Type::Tensor(_), Type::Shape(_)) => {
                // Convert shape to tensor for the operation
                move |lhs, rhs| {
                    quote! {
                        #lhs.div(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                    }
                }
            }
            _ => panic!("Division is supported for tensor, scalar, and shape types only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Div, Arc::new(function))
    }

    pub(crate) fn equal(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Equal,
                    Self::create_broadcast_function("equal", lhs_rank, rhs_rank),
                );
            }
            (Type::Scalar(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs == #rhs },
            (Type::Shape(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let mut result = #lhs;
                        for (result_item, rhs_item) in result.iter_mut().zip(#rhs.iter()) {
                            *result_item = if result_item == rhs_item { 1i64 } else { 0i64 };
                        }
                        result
                    }
                }
            },
            (Type::Shape(_), Type::Tensor(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let shape_tensor = Tensor::<B, 1, Int>::from_data(#lhs.as_slice(), &*self.device);
                        shape_tensor.equal(#rhs)
                    }
                }
            },
            (Type::Tensor(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    {
                        let shape_tensor = Tensor::<B, 1, Int>::from_data(#rhs.as_slice(), &*self.device);
                        #lhs.equal(shape_tensor)
                    }
                }
            },
            _ => panic!(
                "Comparison is supported for tensor to tensor, scalar to scalar, shape to shape, and shape to tensor only"
            ),
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
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Greater,
                    Self::create_broadcast_function("greater", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => {
                move |lhs, rhs| quote! { #lhs.greater_elem(#rhs) }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L > R == R < L
                move |lhs, rhs| quote! { #rhs.lower_elem(#lhs) }
            }
            (Type::Shape(_), Type::Tensor(_)) => move |lhs, rhs| {
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).greater(#rhs)
                }
            },
            (Type::Tensor(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    #lhs.greater(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                }
            },
            (lhs, rhs) => panic!("greater is not supported for {lhs:?} > {rhs:?}"),
        };
        Self::new(lhs, rhs, output, BinaryType::Greater, Arc::new(function))
    }

    pub(crate) fn greater_equal(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::GreaterOrEqual,
                    Self::create_broadcast_function("greater_equal", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => {
                move |lhs, rhs| quote! { #lhs.greater_equal_elem(#rhs) }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L >= R == R <= L
                move |lhs, rhs| quote! { #rhs.lower_equal_elem(#lhs) }
            }
            (Type::Shape(_), Type::Tensor(_)) => move |lhs, rhs| {
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).greater_equal(#rhs)
                }
            },
            (Type::Tensor(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    #lhs.greater_equal(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                }
            },
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
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::Less,
                    Self::create_broadcast_function("lower", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => move |lhs, rhs| quote! { #lhs.lower_elem(#rhs) },
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L < R == R > L
                move |lhs, rhs| quote! { #rhs.greater_elem(#lhs) }
            }
            (Type::Shape(_), Type::Tensor(_)) => move |lhs, rhs| {
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).lower(#rhs)
                }
            },
            (Type::Tensor(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    #lhs.lower(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                }
            },
            (lhs, rhs) => panic!("lower is not supported for {lhs:?} > {rhs:?}"),
        };
        Self::new(lhs, rhs, output, BinaryType::Less, Arc::new(function))
    }

    pub(crate) fn lower_equal(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                let lhs_rank = lhs_tensor.rank;
                let rhs_rank = rhs_tensor.rank;
                return Self::new(
                    lhs,
                    rhs,
                    output,
                    BinaryType::LessOrEqual,
                    Self::create_broadcast_function("lower_equal", lhs_rank, rhs_rank),
                );
            }
            (Type::Tensor(_), Type::Scalar(_)) => {
                move |lhs, rhs| quote! { #lhs.lower_equal_elem(#rhs) }
            }
            (Type::Scalar(_), Type::Tensor(_)) => {
                // L <= R == R >= L
                move |lhs, rhs| quote! { #rhs.greater_equal_elem(#lhs) }
            }
            (Type::Shape(_), Type::Tensor(_)) => move |lhs, rhs| {
                quote! {
                    Tensor::<B, 1, burn::tensor::Int>::from_data(&#lhs as &[_], &*self.device).lower_equal(#rhs)
                }
            },
            (Type::Tensor(_), Type::Shape(_)) => move |lhs, rhs| {
                quote! {
                    #lhs.lower_equal(Tensor::<B, 1, burn::tensor::Int>::from_data(&#rhs as &[_], &*self.device))
                }
            },
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

    pub(crate) fn bool_and(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                if lhs_tensor.kind != TensorKind::Bool || rhs_tensor.kind != TensorKind::Bool {
                    panic!("and operation requires boolean tensors");
                }
                move |lhs, rhs| quote! { #lhs.bool_and(#rhs) }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                if lhs_scalar.kind != ScalarKind::Bool || rhs_scalar.kind != ScalarKind::Bool {
                    panic!("and operation requires boolean scalars");
                }
                move |lhs, rhs| quote! { #lhs && #rhs }
            }
            _ => panic!("and is supported for tensor and scalar bool only"),
        };

        Self::new(lhs, rhs, output, BinaryType::And, Arc::new(function))
    }

    pub(crate) fn bool_or(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                if lhs_tensor.kind != TensorKind::Bool || rhs_tensor.kind != TensorKind::Bool {
                    panic!("or operation requires boolean tensors");
                }
                move |lhs, rhs| quote! { #lhs.bool_or(#rhs) }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                if lhs_scalar.kind != ScalarKind::Bool || rhs_scalar.kind != ScalarKind::Bool {
                    panic!("or operation requires boolean scalars");
                }
                move |lhs, rhs| quote! { #lhs || #rhs }
            }
            _ => panic!("or is supported for tensor and scalar bool only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Or, Arc::new(function))
    }

    pub(crate) fn bool_xor(lhs: Type, rhs: Type, output: Type) -> Self {
        let function = match (&lhs, &rhs) {
            (Type::Tensor(lhs_tensor), Type::Tensor(rhs_tensor)) => {
                if lhs_tensor.kind != TensorKind::Bool || rhs_tensor.kind != TensorKind::Bool {
                    panic!("xor operation requires boolean tensors");
                }
                move |lhs, rhs| quote! { #lhs.not_equal(#rhs) }
            }
            (Type::Scalar(lhs_scalar), Type::Scalar(rhs_scalar)) => {
                if lhs_scalar.kind != ScalarKind::Bool || rhs_scalar.kind != ScalarKind::Bool {
                    panic!("xor operation requires boolean scalars");
                }
                move |lhs, rhs| quote! { #lhs ^ #rhs }
            }
            _ => panic!("xor is supported for tensor and scalar bool only"),
        };

        Self::new(lhs, rhs, output, BinaryType::Xor, Arc::new(function))
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
    ($operator:ident, $burn_operator:ident) => {{
      one_node_graph(
        BinaryNode::$operator(
          Type::Tensor(TensorType::new_float("tensor1", 4)),
          Type::Tensor(TensorType::new_float("tensor2", 4)),
          Type::Tensor(TensorType::new_float("tensor3", 4)),
        ),
        quote! {
            pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                let tensor3 = tensor1.$burn_operator(tensor2);

                tensor3
            }
        },
        vec!["tensor1".to_string(), "tensor2".to_string()],
        vec!["tensor3".to_string()],
      );
    }};
    ($operator:ident) => {
        test_binary_operator_on_tensors!($operator, $operator);
    };
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
            use burn::prelude::*;

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

    #[test]
    fn test_binary_codegen_bool_and() {
        // Test tensor boolean AND
        one_node_graph(
            BinaryNode::bool_and(
                Type::Tensor(TensorType::new_bool("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                Type::Tensor(TensorType::new_bool("tensor3", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4, Bool>, tensor2: Tensor<B, 4, Bool>) -> Tensor<B, 4, Bool> {
                    let tensor3 = tensor1.bool_and(tensor2);

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        // Test scalar boolean AND
        one_node_graph(
            BinaryNode::bool_and(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Bool)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                Type::Scalar(ScalarType::new("scalar3", ScalarKind::Bool)),
            ),
            quote! {
                pub fn forward(&self, scalar1: bool, scalar2: bool) -> bool {
                    let scalar3 = scalar1 && scalar2;

                    scalar3
                }
            },
            vec!["scalar1".to_string(), "scalar2".to_string()],
            vec!["scalar3".to_string()],
        );
    }

    #[test]
    fn test_binary_codegen_bool_or() {
        // Test tensor boolean OR
        one_node_graph(
            BinaryNode::bool_or(
                Type::Tensor(TensorType::new_bool("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                Type::Tensor(TensorType::new_bool("tensor3", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4, Bool>, tensor2: Tensor<B, 4, Bool>) -> Tensor<B, 4, Bool> {
                    let tensor3 = tensor1.bool_or(tensor2);

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        // Test scalar boolean OR
        one_node_graph(
            BinaryNode::bool_or(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Bool)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                Type::Scalar(ScalarType::new("scalar3", ScalarKind::Bool)),
            ),
            quote! {
                pub fn forward(&self, scalar1: bool, scalar2: bool) -> bool {
                    let scalar3 = scalar1 || scalar2;

                    scalar3
                }
            },
            vec!["scalar1".to_string(), "scalar2".to_string()],
            vec!["scalar3".to_string()],
        );
    }

    #[test]
    fn test_binary_codegen_bool_xor() {
        // Test tensor boolean XOR
        one_node_graph(
            BinaryNode::bool_xor(
                Type::Tensor(TensorType::new_bool("tensor1", 4)),
                Type::Tensor(TensorType::new_bool("tensor2", 4)),
                Type::Tensor(TensorType::new_bool("tensor3", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4, Bool>, tensor2: Tensor<B, 4, Bool>) -> Tensor<B, 4, Bool> {
                    let tensor3 = tensor1.not_equal(tensor2);

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );

        // Test scalar boolean XOR
        one_node_graph(
            BinaryNode::bool_xor(
                Type::Scalar(ScalarType::new("scalar1", ScalarKind::Bool)),
                Type::Scalar(ScalarType::new("scalar2", ScalarKind::Bool)),
                Type::Scalar(ScalarType::new("scalar3", ScalarKind::Bool)),
            ),
            quote! {
                pub fn forward(&self, scalar1: bool, scalar2: bool) -> bool {
                    let scalar3 = scalar1 ^ scalar2;

                    scalar3
                }
            },
            vec!["scalar1".to_string(), "scalar2".to_string()],
            vec!["scalar3".to_string()],
        );
    }

    #[test]
    fn test_broadcast_add_different_ranks() {
        // Test 3D + 2D tensors
        one_node_graph(
            BinaryNode::add(
                Type::Tensor(TensorType::new_float("tensor1", 3)),
                Type::Tensor(TensorType::new_float("tensor2", 2)),
                Type::Tensor(TensorType::new_float("tensor3", 3)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 3>, tensor2: Tensor<B, 2>) -> Tensor<B, 3> {
                    let tensor3 = tensor1.add(tensor2.unsqueeze_dims(&[0isize]));

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );
    }

    #[test]
    fn test_broadcast_sub_different_ranks() {
        // Test 2D - 3D tensors
        one_node_graph(
            BinaryNode::sub(
                Type::Tensor(TensorType::new_float("tensor1", 2)),
                Type::Tensor(TensorType::new_float("tensor2", 3)),
                Type::Tensor(TensorType::new_float("tensor3", 3)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 2>, tensor2: Tensor<B, 3>) -> Tensor<B, 3> {
                    let tensor3 = tensor1.unsqueeze_dims(&[0isize]).sub(tensor2);

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );
    }

    #[test]
    fn test_broadcast_mul_different_ranks() {
        // Test 4D * 2D tensors
        one_node_graph(
            BinaryNode::mul(
                Type::Tensor(TensorType::new_float("tensor1", 4)),
                Type::Tensor(TensorType::new_float("tensor2", 2)),
                Type::Tensor(TensorType::new_float("tensor3", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 4>, tensor2: Tensor<B, 2>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.mul(tensor2.unsqueeze_dims(&[0isize, 1isize]));

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );
    }

    #[test]
    fn test_broadcast_div_different_ranks() {
        // Test 1D / 4D tensors
        one_node_graph(
            BinaryNode::div(
                Type::Tensor(TensorType::new_float("tensor1", 1)),
                Type::Tensor(TensorType::new_float("tensor2", 4)),
                Type::Tensor(TensorType::new_float("tensor3", 4)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 1>, tensor2: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor3 = tensor1.unsqueeze_dims(&[0isize, 1isize, 2isize]).div(tensor2);

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );
    }

    #[test]
    fn test_broadcast_same_ranks() {
        // Test that same rank tensors don't get unsqueeze
        one_node_graph(
            BinaryNode::add(
                Type::Tensor(TensorType::new_float("tensor1", 3)),
                Type::Tensor(TensorType::new_float("tensor2", 3)),
                Type::Tensor(TensorType::new_float("tensor3", 3)),
            ),
            quote! {
                pub fn forward(&self, tensor1: Tensor<B, 3>, tensor2: Tensor<B, 3>) -> Tensor<B, 3> {
                    let tensor3 = tensor1.add(tensor2);

                    tensor3
                }
            },
            vec!["tensor1".to_string(), "tensor2".to_string()],
            vec!["tensor3".to_string()],
        );
    }

    #[test]
    fn test_create_broadcast_function_same_rank() {
        let func = BinaryNode::create_broadcast_function("add", 3, 3);
        let lhs = quote! { tensor1 };
        let rhs = quote! { tensor2 };
        let result = func(lhs, rhs);

        let expected = quote! { tensor1.add(tensor2) };
        assert_eq!(result.to_string(), expected.to_string());
    }

    #[test]
    fn test_create_broadcast_function_lhs_higher_rank() {
        let func = BinaryNode::create_broadcast_function("mul", 4, 2);
        let lhs = quote! { tensor1 };
        let rhs = quote! { tensor2 };
        let result = func(lhs, rhs);

        let expected = quote! { tensor1.mul(tensor2.unsqueeze_dims(&[0isize, 1isize])) };
        assert_eq!(result.to_string(), expected.to_string());
    }

    #[test]
    fn test_create_broadcast_function_rhs_higher_rank() {
        let func = BinaryNode::create_broadcast_function("sub", 2, 5);
        let lhs = quote! { tensor1 };
        let rhs = quote! { tensor2 };
        let result = func(lhs, rhs);

        let expected = quote! { tensor1.unsqueeze_dims(&[0isize, 1isize, 2isize]).sub(tensor2) };
        assert_eq!(result.to_string(), expected.to_string());
    }

    #[test]
    fn test_broadcast_all_operations() {
        // Test that all four operations support broadcasting
        type OpFn = fn(Type, Type, Type) -> BinaryNode;
        let ops: Vec<(&str, OpFn)> = vec![
            ("add", BinaryNode::add as OpFn),
            ("sub", BinaryNode::sub as OpFn),
            ("mul", BinaryNode::mul as OpFn),
            ("div", BinaryNode::div as OpFn),
        ];

        for (op_name, op_fn) in ops {
            // Each operation should handle different rank tensors
            let node = op_fn(
                Type::Tensor(TensorType::new_float("x", 3)),
                Type::Tensor(TensorType::new_float("y", 2)),
                Type::Tensor(TensorType::new_float("z", 3)),
            );

            // Should not panic - just verify it creates a valid node
            match node.binary_type {
                BinaryType::Add if op_name == "add" => {}
                BinaryType::Sub if op_name == "sub" => {}
                BinaryType::Mul if op_name == "mul" => {}
                BinaryType::Div if op_name == "div" => {}
                _ => panic!("Unexpected binary type for {}", op_name),
            }
        }
    }
}
