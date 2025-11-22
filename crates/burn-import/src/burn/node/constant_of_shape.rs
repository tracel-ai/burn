use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{Scope, ToTokens, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// Shape parameter for ConstantOfShape (codegen side)
#[derive(Debug, Clone)]
pub enum ConstantOfShapeShapeParam {
    Static(Vec<i64>),
    Runtime(Type),
}

/// Node for ConstantOfShape operation.
#[derive(Debug, Clone)]
pub struct ConstantOfShapeNode {
    pub shape: ConstantOfShapeShapeParam,
    pub output: Type,
    pub value: ConstantValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    /// Float constant.
    Float32(f32),
    Float64(f64),

    /// Integer constant.
    Int32(i32),
    Int64(i64),

    // Boolean constant.
    Bool(bool),
}

impl ConstantOfShapeNode {
    pub fn new(shape: ConstantOfShapeShapeParam, output: Type, value: ConstantValue) -> Self {
        // Verify output type
        assert!(
            matches!(output, Type::Tensor(_) | Type::Scalar(_) | Type::Shape(_)),
            "ConstantOfShape output needs to be a Tensor, Scalar, or Shape!"
        );

        // Note: Runtime shape validation is done in onnx-ir's constant_of_shape_config

        Self {
            shape,
            output,
            value,
        }
    }
}

impl ConstantValue {
    pub fn val_tokens(&self) -> TokenStream {
        match self {
            Self::Float32(val) => quote! { #val },
            Self::Float64(val) => quote! { #val },
            Self::Int32(val) => quote! { #val },
            Self::Int64(val) => quote! { #val },
            Self::Bool(val) => quote! { #val },
        }
    }

    pub fn from_vec<T: Into<Self> + Copy>(mut source: Vec<T>) -> Self {
        assert_eq!(
            source.len(),
            1,
            "ConstantOfShape value from a vec needs to have exactly 1 element!"
        );
        source.drain(..).next().unwrap().into()
    }
}

impl From<f32> for ConstantValue {
    fn from(value: f32) -> Self {
        Self::Float32(value)
    }
}
impl From<f64> for ConstantValue {
    fn from(value: f64) -> Self {
        Self::Float64(value)
    }
}
impl From<i32> for ConstantValue {
    fn from(value: i32) -> Self {
        Self::Int32(value)
    }
}
impl From<i64> for ConstantValue {
    fn from(value: i64) -> Self {
        Self::Int64(value)
    }
}
impl From<bool> for ConstantValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantOfShapeNode {
    fn input_types(&self) -> Vec<Type> {
        match &self.shape {
            ConstantOfShapeShapeParam::Static(_) => vec![],
            ConstantOfShapeShapeParam::Runtime(t) => vec![t.clone()],
        }
    }

    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = self.output.name();
        let value = self.value.val_tokens();

        // Generate shape expression based on Static or Runtime
        let shape_expr = match &self.shape {
            ConstantOfShapeShapeParam::Static(static_shape) => {
                // We have static shape values - embed them directly in the code
                let shape_values = static_shape.iter().map(|v| {
                    let val = *v as usize;
                    quote! { #val }
                });
                quote! { [#(#shape_values),*] }
            }
            ConstantOfShapeShapeParam::Runtime(t) => {
                // Runtime shape input
                let input_name = t.name();
                quote! { #input_name }
            }
        };

        match &self.output {
            Type::Scalar(scalar) => {
                // For scalar output, the input shape should be empty (rank 0)
                // Just return the constant value directly
                let ty = scalar.ty();
                quote! {
                    let #output: #ty = #value;
                }
            }
            Type::Tensor(tensor) => {
                let output_rank = tensor.rank.to_tokens();

                // Note: in the generated code, self.device is a &module::Ignored<Device>,
                // so to get a &Device, &* is needed

                match &self.value {
                    ConstantValue::Bool(bool) => {
                        // Currently there is no full bool tensor support in the backend
                        // So we use 0 or 1 with bool type casting
                        // See: https://github.com/tracel-ai/burn/issues/1535
                        if *bool {
                            quote! {
                                let #output = Tensor::<B, #output_rank, Int>::ones(#shape_expr, &*self.device).bool();
                            }
                        } else {
                            quote! {
                                let #output = Tensor::<B, #output_rank, Int>::zeros(#shape_expr, &*self.device).bool();
                            }
                        }
                    }
                    _ => quote! {
                        let #output = Tensor::full(#shape_expr, #value, &*self.device);
                    },
                }
            }
            Type::Shape(shape) => {
                // Optimization: When ConstantOfShape outputs Shape(1) with Int64,
                // we directly create a shape array instead of a tensor.
                // This is a common pattern for shape manipulation operations.
                assert_eq!(shape.rank, 1, "Shape optimization only supports Shape(1)");

                // The input is Shape(1) which means [N] where N is the dimension
                // We need to create a shape array with that single value
                quote! {
                    // Input shape tells us the size, value tells us what to fill
                    let #output: [i64; 1] = [#value];
                }
            }
            _ => unreachable!("ConstantOfShape output must be Tensor, Scalar, or Shape"),
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::ConstantOfShape(self)
    }
}

impl OnnxIntoNode for ConstantOfShapeNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        use onnx_ir::node::constant_of_shape::ConstantOfShapeShape;

        let onnx_ir::Node::ConstantOfShape(n) = node else {
            panic!("Expected ConstantOfShape node");
        };

        // Convert from onnx-ir enum to codegen enum
        let shape = match &n.config.shape {
            ConstantOfShapeShape::Static(values) => {
                ConstantOfShapeShapeParam::Static(values.clone())
            }
            ConstantOfShapeShape::Runtime(runtime_ref) => {
                let arg = &n.inputs[runtime_ref.input_index];
                ConstantOfShapeShapeParam::Runtime(Type::from(arg))
            }
        };

        let output = Type::from(n.outputs.first().unwrap());

        // The value of the output elements. Should be a one-element tensor.
        // If not specified, it defaults to a tensor of value 0 and datatype float32
        let value = n
            .config
            .value
            .as_ref()
            .map(|tensor_data| match tensor_data.dtype {
                onnx_ir::ir::DType::F32 => {
                    ConstantValue::from_vec(tensor_data.to_vec::<f32>().unwrap())
                }
                onnx_ir::ir::DType::F64 => {
                    ConstantValue::from_vec(tensor_data.to_vec::<f64>().unwrap())
                }
                onnx_ir::ir::DType::I32 => {
                    ConstantValue::from_vec(tensor_data.to_vec::<i32>().unwrap())
                }
                onnx_ir::ir::DType::I64 => {
                    ConstantValue::from_vec(tensor_data.to_vec::<i64>().unwrap())
                }
                onnx_ir::ir::DType::Bool => {
                    ConstantValue::from_vec(tensor_data.to_vec::<bool>().unwrap())
                }
                ty => panic!("Unsupported value type {ty:?} for ConstantOfShape!"),
            })
            .unwrap_or(ConstantValue::Float32(0.0f32));

        ConstantOfShapeNode::new(shape, output, value)
    }
}

#[cfg(test)]
mod tests {
    use burn::record::FullPrecisionSettings;

    use super::*;
    use crate::burn::{
        TensorType,
        graph::BurnGraph,
        node::{constant_of_shape::ConstantOfShapeNode, test::assert_tokens},
    };

    #[test]
    fn test_constant_val() {
        assert_eq!(ConstantValue::from(1i32), ConstantValue::Int32(1i32));
        assert_eq!(ConstantValue::from(-1i64), ConstantValue::Int64(-1i64));
        assert_eq!(ConstantValue::from(0f32), ConstantValue::Float32(0f32));
        assert_eq!(ConstantValue::from(0f64), ConstantValue::Float64(0f64));
        assert_eq!(ConstantValue::from(true), ConstantValue::Bool(true));
        assert_eq!(
            ConstantValue::from_vec(vec![2i32]),
            ConstantValue::Int32(2i32)
        );
    }

    #[test]
    fn test_codegen_nodes() {
        use crate::burn::ShapeType;

        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConstantOfShapeNode::new(
            ConstantOfShapeShapeParam::Runtime(Type::Shape(ShapeType::new("shape1", 4))),
            Type::Tensor(TensorType::new_float("tensor2", 4)),
            ConstantValue::Float32(1.25f32),
        ));

        graph.register_input_output(
            vec!["shape1".to_string()],
            vec!["tensor2".to_string()],
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
                pub fn forward(&self, shape1: [i64;4]) -> Tensor<B, 4> {
                    let tensor2 = Tensor::full(shape1, 1.25f32, &*self.device);
                    tensor2
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_scalar_output() {
        use crate::burn::{ScalarType, ShapeType};

        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConstantOfShapeNode::new(
            ConstantOfShapeShapeParam::Runtime(Type::Shape(ShapeType::new("shape1", 0))),
            Type::Scalar(ScalarType::new("scalar1", crate::burn::ScalarKind::Float32)),
            ConstantValue::Float32(42.0f32),
        ));

        graph.register_input_output(
            vec!["shape1".to_string()],
            vec!["scalar1".to_string()],
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
                pub fn forward(&self, shape1: [i64;0]) -> f32 {
                    let scalar1: f32 = 42f32;
                    scalar1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_shape_output() {
        use crate::burn::ShapeType;

        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        graph.register(ConstantOfShapeNode::new(
            ConstantOfShapeShapeParam::Runtime(Type::Shape(ShapeType::new("shape_input", 1))),
            Type::Shape(ShapeType::new("shape_output", 1)),
            ConstantValue::Int64(10i64),
        ));

        graph.register_input_output(
            vec!["shape_input".to_string()],
            vec!["shape_output".to_string()],
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
                pub fn forward(&self, shape_input: [i64;1]) -> [i64;1] {
                    // Input shape tells us the size, value tells us what to fill
                    let shape_output: [i64; 1] = [10i64];
                    shape_output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_static_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        // Test with static shape values
        graph.register(ConstantOfShapeNode::new(
            ConstantOfShapeShapeParam::Static(vec![2, 3, 4]),
            Type::Tensor(TensorType::new_float("tensor1", 3)),
            ConstantValue::Float32(0.5f32),
        ));

        graph.register_input_output(vec![], vec!["tensor1".to_string()], &[], &[]);

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
                pub fn forward(&self) -> Tensor<B, 3> {
                    let tensor1 = Tensor::full([2usize, 3usize, 4usize], 0.5f32, &*self.device);
                    tensor1
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
