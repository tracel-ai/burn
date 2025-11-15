use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{ScalarType, ShapeType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct WhereNode {
    /// Bool tensor. When True (nonzero), yield X, otherwise yield Y.
    pub condition: Type,
    /// Values selected at indices where condition is True.
    pub x: Type,
    /// Values selected at indices where condition is False.
    pub y: Type,
    pub output: Type,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for WhereNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<crate::burn::Type> {
        vec![self.condition.clone(), self.x.clone(), self.y.clone()]
    }

    fn forward(&self, scope: &mut crate::burn::Scope, node_position: usize) -> TokenStream {
        match &self.output {
            Type::Tensor(out) => {
                let cond = Self::input_as_tensor(&self.condition, out.rank, scope, node_position);
                let y = Self::input_as_tensor(&self.y, out.rank, scope, node_position);
                let out_id = &out.name;

                if let Type::Scalar(x) = &self.x {
                    let x = &x.name;
                    quote! {
                        let #out_id = #y.mask_fill(#cond, #x);
                    }
                } else {
                    let x = Self::input_as_tensor(&self.x, out.rank, scope, node_position);
                    quote! {
                        let #out_id = #y.mask_where(#cond, #x);
                    }
                }
            }
            Type::Scalar(out) => {
                // Scalar out means all inputs are scalars as well:
                let cond = self.condition.as_scalar();
                let x = self.x.as_scalar();
                let y = self.y.as_scalar();
                Self::forward_scalar(out, cond, x, y)
            }
            Type::Shape(out) => {
                // Shape output - all inputs should be shapes or compatible types
                Self::forward_shape(out, &self.condition, &self.x, &self.y)
            }
            other => panic!("Where cannot handle {other:?}"),
        }
    }

    fn into_node(self) -> super::Node<PS> {
        Node::Where(self)
    }
}

impl OnnxIntoNode for WhereNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let (inputs, outputs) = match node {
            onnx_ir::Node::Where {
                inputs, outputs, ..
            } => (inputs, outputs),
            _ => panic!("Expected Where node"),
        };
        let condition = Type::from(inputs.first().unwrap());
        let x = Type::from(inputs.get(1).unwrap());
        let y = Type::from(inputs.get(2).unwrap());
        let output = Type::from(outputs.first().unwrap());
        Self::new(condition, x, y, output)
    }
}

impl WhereNode {
    fn forward_scalar(
        out: &ScalarType,
        cond: &ScalarType,
        x: &ScalarType,
        y: &ScalarType,
    ) -> TokenStream {
        let out_name = &out.name;
        let out_type = out.ty();
        let cond_name = &cond.name;
        let x_name = &x.name;
        let y_name = &y.name;

        quote! {
            let #out_name : #out_type = if #cond_name {
                #x_name
            }
            else {
                #y_name
            };
        }
    }

    fn forward_shape(out: &ShapeType, condition: &Type, x: &Type, y: &Type) -> TokenStream {
        let out_name = &out.name;

        // Generate code based on input types - only semantically valid combinations
        match (condition, x, y) {
            (Type::Shape(cond), Type::Shape(x_shape), Type::Shape(y_shape)) => {
                // All shapes: element-wise selection between shape dimensions
                // Each element of condition determines whether to take from x or y
                let cond_name = &cond.name;
                let x_name = &x_shape.name;
                let y_name = &y_shape.name;

                quote! {
                    let #out_name = {
                        let mut result = #y_name;
                        for (i, (cond_item, x_item)) in #cond_name.iter().zip(#x_name.iter()).enumerate() {
                            if *cond_item != 0 {
                                result[i] = *x_item;
                            }
                        }
                        result
                    };
                }
            }
            (Type::Scalar(cond), Type::Shape(x_shape), Type::Shape(y_shape)) => {
                // Scalar condition: select entire shape x or y
                let cond_name = &cond.name;
                let x_name = &x_shape.name;
                let y_name = &y_shape.name;

                quote! {
                    let #out_name = if #cond_name { #x_name } else { #y_name };
                }
            }
            _ => panic!(
                "Where with Shape output only supports: \
                 (Shape, Shape, Shape) for element-wise selection or \
                 (Scalar, Shape, Shape) for whole shape selection"
            ),
        }
    }

    fn input_as_tensor(
        input: &Type,
        broadcast_rank: usize,
        scope: &mut crate::burn::Scope,
        node_position: usize,
    ) -> TokenStream {
        let (tensor, rank) = match input {
            Type::Tensor(t) => (scope.tensor_use_owned(t, node_position), t.rank),
            Type::Scalar(s) => (s.to_full_tensor(&vec![1; broadcast_rank]), broadcast_rank),
            Type::Shape(s) => (s.to_tensor(), 1),
            _ => panic!("Where op: {input:?} input not implemented"),
        };
        if rank < broadcast_rank {
            // Generate unsqueeze_dims to add trailing dimensions for broadcasting
            // Create a vector of dimension indices to unsqueeze at the end
            let dims_to_unsqueeze: Vec<isize> =
                (rank..broadcast_rank).map(|d| d as isize).collect();
            let dims = quote! { &[#(#dims_to_unsqueeze),*] };
            quote! { #tensor.unsqueeze_dims(#dims) }
        } else {
            tensor
        }
    }
}

#[cfg(test)]
mod tests {

    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        ScalarKind, ShapeType, TensorType,
        graph::BurnGraph,
        node::{test::assert_tokens, where_op::WhereNode},
    };

    #[test]
    fn test_codegen_where() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Tensor(TensorType::new_float("tensor3", 2)),
            Type::Tensor(TensorType::new_float("tensor4", 2)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "tensor3".to_string(),
            ],
            vec!["tensor4".to_string()],
            &[],
            &[],
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
                    tensor1: Tensor<B, 2, Bool>,
                    tensor2: Tensor<B, 2>,
                    tensor3: Tensor<B, 2>
                ) -> Tensor<B, 2> {
                    let tensor4 = tensor3.mask_where(tensor1, tensor2);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_broadcasted() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Tensor(TensorType::new_float("tensor3", 3)),
            Type::Tensor(TensorType::new_float("tensor4", 4)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "tensor3".to_string(),
            ],
            vec!["tensor4".to_string()],
            &[],
            &[],
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
                    tensor1: Tensor<B, 4, Bool>,
                    tensor2: Tensor<B, 2>,
                    tensor3: Tensor<B, 3>
                ) -> Tensor<B, 4> {
                    let tensor4 = tensor3
                        .unsqueeze_dims(&[3isize])
                        .mask_where(tensor1, tensor2.unsqueeze_dims(&[2isize, 3isize]));

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_scalar_x() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 2)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float64)),
            Type::Tensor(TensorType::new_float("tensor3", 2)),
            Type::Tensor(TensorType::new_float("tensor4", 2)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "scalar2".to_string(),
                "tensor3".to_string(),
            ],
            vec!["tensor4".to_string()],
            &[],
            &[],
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
                    tensor1: Tensor<B, 2, Bool>,
                    scalar2: f64,
                    tensor3: Tensor<B, 2>
                ) -> Tensor<B, 2> {
                    let tensor4 = tensor3.mask_fill(tensor1, scalar2);

                    tensor4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_scalar_y() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Tensor(TensorType::new_bool("tensor1", 2)),
            Type::Tensor(TensorType::new_float("tensor2", 2)),
            Type::Scalar(ScalarType::new("scalar3", ScalarKind::Float64)),
            Type::Tensor(TensorType::new_float("tensor4", 2)),
        ));

        graph.register_input_output(
            vec![
                "tensor1".to_string(),
                "tensor2".to_string(),
                "scalar3".to_string(),
            ],
            vec!["tensor4".to_string()],
            &[],
            &[],
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
                    tensor1: Tensor<B, 2, Bool>,
                    tensor2: Tensor<B, 2>,
                    scalar3: f64
                ) -> Tensor<B, 2> {
                    let tensor4 = Tensor::<B, 2, burn::tensor::Float>::full([1, 1], scalar3, &*self.device)
                        .mask_where(tensor1, tensor2);

                    tensor4
                }
            }

        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_all_scalar() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Bool)),
            Type::Scalar(ScalarType::new("scalar2", ScalarKind::Float64)),
            Type::Scalar(ScalarType::new("scalar3", ScalarKind::Float64)),
            Type::Scalar(ScalarType::new("scalar4", ScalarKind::Float64)),
        ));

        graph.register_input_output(
            vec![
                "scalar1".to_string(),
                "scalar2".to_string(),
                "scalar3".to_string(),
            ],
            vec!["scalar4".to_string()],
            &[],
            &[],
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
                    scalar1: bool,
                    scalar2: f64,
                    scalar3: f64
                ) -> f64 {
                    let scalar4: f64 = if scalar1 { scalar2 } else { scalar3 };

                    scalar4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_all_shapes() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Shape(ShapeType::new("shape1", 3)),
            Type::Shape(ShapeType::new("shape2", 3)),
            Type::Shape(ShapeType::new("shape3", 3)),
            Type::Shape(ShapeType::new("shape4", 3)),
        ));

        graph.register_input_output(
            vec![
                "shape1".to_string(),
                "shape2".to_string(),
                "shape3".to_string(),
            ],
            vec!["shape4".to_string()],
            &[],
            &[],
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
                    shape1: [i64; 3],
                    shape2: [i64; 3],
                    shape3: [i64; 3]
                ) -> [i64; 3] {
                    let shape4 = {
                        let mut result = shape3;
                        for (i, (cond_item, x_item)) in shape1.iter().zip(shape2.iter()).enumerate() {
                            if *cond_item != 0 {
                                result[i] = *x_item;
                            }
                        }
                        result
                    };

                    shape4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_where_scalar_cond_shape_values() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(WhereNode::new(
            Type::Scalar(ScalarType::new("scalar1", ScalarKind::Bool)),
            Type::Shape(ShapeType::new("shape2", 3)),
            Type::Shape(ShapeType::new("shape3", 3)),
            Type::Shape(ShapeType::new("shape4", 3)),
        ));

        graph.register_input_output(
            vec![
                "scalar1".to_string(),
                "shape2".to_string(),
                "shape3".to_string(),
            ],
            vec!["shape4".to_string()],
            &[],
            &[],
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
                    scalar1: bool,
                    shape2: [i64; 3],
                    shape3: [i64; 3]
                ) -> [i64; 3] {
                    let shape4 = if scalar1 { shape2 } else { shape3 };

                    shape4
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
