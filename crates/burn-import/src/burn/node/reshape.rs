use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub enum ReshapeShape {
    Static(Vec<i64>),
    Runtime(Type),
}

#[derive(Debug, Clone)]
pub struct ReshapeNode {
    pub input: Type,  // Changed to Type to support both Tensor and Shape inputs
    pub output: Type, // Changed from TensorType to Type to support scalar outputs
    pub shape: ReshapeShape,
}

impl ReshapeNode {
    pub fn new<S: Into<ReshapeShape>>(input: Type, output: Type, shape: S) -> Self {
        Self {
            input,
            output,
            shape: shape.into(),
        }
    }
}

impl From<Vec<i64>> for ReshapeShape {
    fn from(shape: Vec<i64>) -> Self {
        ReshapeShape::Static(shape)
    }
}

impl From<Type> for ReshapeShape {
    fn from(shape: Type) -> Self {
        ReshapeShape::Runtime(shape)
    }
}

impl ReshapeNode {
    // Helper for runtime shape reshaping
    fn reshape_with_runtime_shape(
        &self,
        input: TokenStream,
        output: &proc_macro2::Ident,
        shape_type: &Type,
        output_rank: usize,
    ) -> TokenStream {
        match shape_type {
            Type::Shape(shape) => {
                let shape_name = &shape.name;
                quote! {
                    let #output = #input.reshape(#shape_name);
                }
            }
            Type::Tensor(tensor) => {
                let shape_name = &tensor.name;
                let array_init = (0..output_rank)
                    .map(|i| {
                        let idx = proc_macro2::Literal::usize_unsuffixed(i);
                        quote! { shape_array[#idx] as usize }
                    })
                    .collect::<Vec<_>>();

                quote! {
                    let shape_data = #shape_name.to_data();
                    let shape_array = shape_data.as_slice::<i64>().unwrap();
                    let #output = #input.reshape([#(#array_init),*]);
                }
            }
            _ => panic!("Shape parameter must be a tensor or shape type"),
        }
    }

    // Handle reshaping when input is a Tensor
    fn reshape_tensor_input(&self, input: TokenStream) -> TokenStream {
        match &self.output {
            Type::Scalar(scalar) => {
                let output_name = &scalar.name;
                let elem_type = scalar.ty();
                quote! {
                    let #output_name = #input.into_scalar().elem::<#elem_type>();
                }
            }
            Type::Shape(_) => unimplemented!("Tensor to Shape is not supported"),
            Type::Tensor(output_tensor) => {
                let output = &output_tensor.name;
                match &self.shape {
                    ReshapeShape::Static(shape_values) => {
                        let shape_values = shape_values.to_tokens();
                        quote! {
                            let #output = #input.reshape(#shape_values);
                        }
                    }
                    ReshapeShape::Runtime(shape_type) => self.reshape_with_runtime_shape(
                        input,
                        output,
                        shape_type,
                        output_tensor.rank,
                    ),
                }
            }
            _ => panic!("Unexpected output type"),
        }
    }

    // Handle reshaping when input is a Shape
    fn reshape_shape_input(&self, input_shape: &crate::burn::ShapeType) -> TokenStream {
        let shape_name = &input_shape.name;
        let input_rank = input_shape.rank;

        match &self.output {
            Type::Scalar(scalar) => {
                if input_rank != 1 {
                    panic!(
                        "Shape to scalar requires Shape(1), got Shape({})",
                        input_rank
                    );
                }
                let output_name = &scalar.name;
                let elem_type = scalar.ty();
                quote! {
                    let #output_name = #shape_name[0] as #elem_type;
                }
            }
            Type::Shape(output_shape) => {
                let output_name = &output_shape.name;
                let output_rank = output_shape.rank;

                if input_rank == output_rank {
                    quote! {
                        let #output_name = #shape_name;
                    }
                } else {
                    quote! {
                        let #output_name: [i64; #output_rank] = {
                            let mut result = [0i64; #output_rank];
                            let copy_len = #input_rank.min(#output_rank);
                            result[..copy_len].copy_from_slice(&#shape_name[..copy_len]);
                            result
                        };
                    }
                }
            }
            Type::Tensor(output_tensor) => {
                // Convert Shape to Tensor first, then reshape
                let shape_to_tensor = quote! {
                    {
                        let shape_array = #shape_name as [i64; #input_rank];
                        Tensor::<B, 1, Int>::from_data(
                            TensorData::from(shape_array),
                            &self.device
                        )
                    }
                };

                let output = &output_tensor.name;
                match &self.shape {
                    ReshapeShape::Static(shape_values) => {
                        let shape_values = shape_values.to_tokens();
                        quote! {
                            let #output = #shape_to_tensor.reshape(#shape_values);
                        }
                    }
                    ReshapeShape::Runtime(shape_type) => self.reshape_with_runtime_shape(
                        shape_to_tensor,
                        output,
                        shape_type,
                        output_tensor.rank,
                    ),
                }
            }
            _ => panic!("Unexpected output type"),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ReshapeNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        match &self.shape {
            ReshapeShape::Static(_) => vec![self.input.clone()],
            ReshapeShape::Runtime(shape_type) => {
                vec![self.input.clone(), shape_type.clone()]
            }
        }
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        // Simplified logic driven by input type
        match &self.input {
            Type::Tensor(input_tensor) => {
                // Tensor input path
                let input = scope.tensor_use_owned(input_tensor, node_position);
                self.reshape_tensor_input(input)
            }
            Type::Shape(input_shape) => {
                // Shape input path
                self.reshape_shape_input(input_shape)
            }
            _ => panic!("Reshape: unexpected input type"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Reshape(self)
    }
}

impl OnnxIntoNode for ReshapeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let input_arg = node.inputs.first().unwrap();
        let output_arg = node.outputs.first().unwrap();
        let output = Type::from(output_arg);
        let config = node.config::<onnx_ir::node::reshape::ReshapeConfig>();
        let input = Type::from(input_arg);
        match &config.shape {
            onnx_ir::node::reshape::ReshapeInput::Static(shape) => {
                Self::new(input, output, shape.clone())
            }
            onnx_ir::node::reshape::ReshapeInput::Runtime(shape_ref) => {
                // Get the actual argument using the RuntimeInputRef
                let shape_arg = &node.inputs[shape_ref.input_index];
                let shape_input = Type::from(shape_arg);
                Self::new(input, output, shape_input)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{reshape::ReshapeNode, test::assert_tokens},
    };

    #[test]
    fn test_codegen_reshape_with_static_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            vec![4, 4, 4, 4],
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["tensor2".to_string()]);

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
                pub fn forward(&self, tensor1: Tensor<B, 4>) -> Tensor<B, 4> {
                    let tensor2 = tensor1.reshape([4, 4, 4, 4]);

                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reshape_with_tensor_as_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 1)),
            Type::Tensor(TensorType::new_float("output", 2)),
            Type::Tensor(TensorType::new_int("shape", 1)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "shape".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 1>, shape: Tensor<B, 1, Int>) -> Tensor<B, 2> {
                    let shape_data = shape.to_data();
                    let shape_array = shape_data.as_slice::<i64>().unwrap();
                    let output = tensor1.reshape([shape_array[0] as usize, shape_array[1] as usize]);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reshape_with_shape_type() {
        use crate::burn::ShapeType;

        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 4)),
            Type::Tensor(TensorType::new_float("output", 2)),
            Type::Shape(ShapeType::new("shape", 2)),
        ));

        graph.register_input_output(
            vec!["tensor1".to_string(), "shape".to_string()],
            vec!["output".to_string()],
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
                pub fn forward(&self, tensor1: Tensor<B, 4>, shape: [i64; 2]) -> Tensor<B, 2> {
                    let output = tensor1.reshape(shape);

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_reshape_to_scalar() {
        use crate::burn::ScalarType;

        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ReshapeNode::new(
            Type::Tensor(TensorType::new_float("tensor1", 2)),
            Type::Scalar(ScalarType::new("output", crate::burn::ScalarKind::Float32)),
            vec![], // Empty shape for scalar
        ));

        graph.register_input_output(vec!["tensor1".to_string()], vec!["output".to_string()]);

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
                pub fn forward(&self, tensor1: Tensor<B, 2>) -> f32 {
                    let output = tensor1.into_scalar().elem::<f32>();

                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
