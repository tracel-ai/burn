use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{
    ScalarKind, ScalarType, Scope, ShapeType, TensorKind, TensorType, ToTokens, Type,
};
use burn::{
    module::ParamId,
    record::{ParamSerde, PrecisionSettings},
    tensor::TensorData,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ConstantNode {
    pub name: String,
    pub value: ConstantValue,
    pub output: Type,
}

#[derive(Debug, Clone, new)]
pub enum ConstantValue {
    /// Float constant.
    Float32(f32),
    Float64(f64),

    /// Integer constant.
    Int32(i32),
    Int64(i64),

    // Boolean constant.
    Bool(bool),

    /// Tensor constant.
    Tensor(TensorType, TensorData),

    /// Shape constant.
    Shape(Vec<usize>),
}

impl ConstantValue {
    pub fn ty_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Float32(_) => quote! { f32 },
            ConstantValue::Float64(_) => quote! { f64 },
            ConstantValue::Int32(_) => quote! { i32 },
            ConstantValue::Int64(_) => quote! { i64 },
            ConstantValue::Bool(_) => quote! { bool },
            ConstantValue::Tensor(tensor_type, _) => {
                let ty = tensor_type.ty();
                quote! { burn::module::Param<#ty>}
            }
            ConstantValue::Shape(shape_vec) => {
                let rank = proc_macro2::Literal::usize_unsuffixed(shape_vec.len());
                quote! { [i64; #rank] }
            }
        }
    }

    pub fn val_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Float32(val) => quote! { #val },
            ConstantValue::Float64(val) => quote! { #val },
            ConstantValue::Int32(val) => quote! { #val },
            ConstantValue::Int64(val) => quote! { #val },
            ConstantValue::Bool(val) => quote! { #val },
            ConstantValue::Tensor(_, _) => {
                panic!("Tensor constant is not assignable.")
            }
            ConstantValue::Shape(shape_vec) => {
                let values: Vec<_> = shape_vec
                    .iter()
                    .map(|&v| {
                        let v_lit = proc_macro2::Literal::i64_suffixed(v as i64);
                        quote! { #v_lit }
                    })
                    .collect();
                quote! { [#(#values),*] }
            }
        }
    }
}

impl ConstantNode {
    pub fn new(name: String, value: ConstantValue, output: Type) -> Self {
        Self {
            name,
            value,
            output,
        }
    }
    pub fn constant_value_into_type(&self) -> Type {
        let name = Ident::new(self.name.as_str(), Span::call_site());
        match &self.value {
            ConstantValue::Float32(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Float32,
            }),
            ConstantValue::Float64(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Float64,
            }),
            ConstantValue::Int32(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Int32,
            }),
            ConstantValue::Int64(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Int64,
            }),
            ConstantValue::Bool(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Bool,
            }),

            ConstantValue::Tensor(tensor_type, _) => Type::Tensor(tensor_type.clone()),
            ConstantValue::Shape(shape_vec) => {
                Type::Shape(ShapeType::new(name.to_string(), shape_vec.len()))
            }
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantNode {
    fn output_types(&self) -> Vec<Type> {
        vec![self.output.clone()]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![]
    }

    fn field_type(&self) -> Option<Type> {
        match &self.value {
            ConstantValue::Tensor(tensor_type, _) => Some(Type::Tensor(tensor_type.clone())),
            _ => None,
        }
    }

    fn field_init(&self) -> Option<TokenStream> {
        match &self.value {
            ConstantValue::Tensor(tensor_type, data) => {
                let ty = tensor_type.ty();
                let name = Ident::new(self.name.as_ref(), Span::call_site());

                assert_eq!(
                    data.shape.len(),
                    tensor_type.rank,
                    "Tensor data shape does not match tensor type rank"
                );

                let shape = data.shape.to_tokens();
                let rank = tensor_type.rank.to_tokens();

                match tensor_type.kind {
                    crate::burn::TensorKind::Int => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank, Int>::zeros(#shape, device),
                            device.clone(),
                            false
                        );
                    }),
                    crate::burn::TensorKind::Float => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank>::zeros(#shape, device),
                            device.clone(),
                            false,
                        );
                    }),
                    crate::burn::TensorKind::Bool => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank, Bool>::empty(#shape, device),
                            device.clone(),
                            false,
                        );
                    }),
                }
            }
            _ => None,
        }
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let name = Ident::new(self.name.as_ref(), Span::call_site());
        let output = self.output.name();

        match &self.value {
            ConstantValue::Tensor(_, _) => {
                quote! {
                    let #output = self.#name.val();
                }
            }
            _ => {
                let val = self.value.val_tokens();
                let ty = self.value.ty_tokens();

                quote! {
                    let #output: #ty = #val;
                }
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Constant(self)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let ConstantValue::Tensor(tensor_type, data) = &self.value {
            let data = match tensor_type.kind {
                TensorKind::Int => data.clone().convert::<PS::IntElem>(),
                TensorKind::Float => data.clone().convert::<PS::FloatElem>(),
                TensorKind::Bool => data.clone(),
            };
            let data = ParamSerde::new(ParamId::new().to_string(), data);
            return data.serialize(serializer);
        }

        S::serialize_none(serializer)
    }
}

impl OnnxIntoNode for ConstantNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        use onnx_ir::ir::{ArgType, AttributeValue, Data, ElementType};

        let output = node.outputs.first().unwrap();

        // Get the value directly from node attributes
        let value_keys = [
            "value",
            "value_float",
            "value_floats",
            "value_int",
            "value_ints",
            "value_string",
            "value_strings",
            "sparse_value",
        ];

        let attr_value = value_keys
            .iter()
            .find_map(|&key| node.attrs.get(key))
            .expect("Constant should have a value attribute");

        // Helper to map elem type to ConstantValue (single scalar)
        fn scalar_from_data(elem: ElementType, data: Data) -> ConstantValue {
            match elem {
                ElementType::Float64 => ConstantValue::Float64(data.into_f64()),
                ElementType::Float32 => ConstantValue::Float32(data.into_f32()),
                ElementType::Int64 => ConstantValue::Int64(data.into_i64()),
                ElementType::Int32 => ConstantValue::Int32(data.into_i32()),
                ElementType::Bool => ConstantValue::Bool(data.into_bool()),
                ElementType::Uint8 => ConstantValue::Int32(data.into_i32()),
                ElementType::Int8 => ConstantValue::Int32(data.into_i32()),
                _ => panic!("Unsupported scalar type: {elem:?}"),
            }
        }

        // Helper to serialize data - hardcoded to f32
        fn serialize_data(data: Data, shape: Vec<usize>) -> TensorData {
            match data {
                Data::Float16s(val) => TensorData::new(val, shape).convert::<f32>(),
                Data::Float32s(val) => TensorData::new(val, shape).convert::<f32>(),
                Data::Float64s(val) => TensorData::new(val, shape).convert::<f32>(),
                Data::Int32s(val) => TensorData::new(val, shape).convert::<f32>(),
                Data::Int64s(val) => TensorData::new(val, shape).convert::<f32>(),
                _ => panic!("Unsupported tensor element type"),
            }
        }

        fn serialize_bool_data(data: Data, shape: Vec<usize>) -> TensorData {
            match data {
                Data::Bools(val) => TensorData::new(val, shape),
                _ => panic!("Expected boolean data for serialize_bool_data"),
            }
        }

        let const_value = match &output.ty {
            ArgType::Shape(rank) => {
                // Extract tensor data from attribute
                let tensor_data = match attr_value {
                    AttributeValue::Tensor(td) => td,
                    _ => panic!("Expected Tensor attribute for Shape constant"),
                };

                let shape_values: Vec<usize> = tensor_data
                    .data
                    .clone()
                    .into_i64s()
                    .into_iter()
                    .map(|v| v as usize)
                    .collect();
                assert_eq!(shape_values.len(), *rank, "Shape constant rank mismatch");
                ConstantValue::Shape(shape_values)
            }

            ArgType::Tensor(tensor) => {
                if tensor.rank == 0 {
                    // Extract scalar data from attribute
                    let data = match attr_value {
                        AttributeValue::Tensor(td) => td.data.clone(),
                        AttributeValue::Float32(f) => Data::Float32(*f),
                        AttributeValue::Int64(i) => Data::Int64(*i),
                        _ => panic!("Unsupported attribute type for scalar constant"),
                    };
                    scalar_from_data(tensor.elem_type.clone(), data)
                } else {
                    let kind: TensorKind = tensor.elem_type.clone().into();
                    let rank = tensor.rank;
                    let name = node.name.clone();

                    // Extract tensor data from attribute
                    let tensor_data = match attr_value {
                        AttributeValue::Tensor(td) => td,
                        _ => panic!("Expected Tensor attribute for tensor constant"),
                    };

                    let tensor_data = match &tensor.elem_type {
                        ElementType::Float32 | ElementType::Float64 | ElementType::Float16 => {
                            serialize_data(tensor_data.data.clone(), tensor_data.shape.clone())
                        }
                        ElementType::Int32
                        | ElementType::Int64
                        | ElementType::Uint8
                        | ElementType::Int8 => serialize_data(tensor_data.data.clone(), tensor_data.shape.clone()),
                        ElementType::Bool => {
                            serialize_bool_data(tensor_data.data.clone(), tensor_data.shape.clone())
                        }
                        other => panic!("Unsupported constant tensor type: {:?} ", other),
                    };

                    ConstantValue::Tensor(TensorType::new(name, rank, kind), tensor_data)
                }
            }

            ArgType::Scalar(elem_type) => {
                // Extract scalar data from attribute
                let data = match attr_value {
                    AttributeValue::Tensor(td) => td.data.clone(),
                    AttributeValue::Float32(f) => Data::Float32(*f),
                    AttributeValue::Int64(i) => Data::Int64(*i),
                    _ => panic!("Unsupported attribute type for scalar constant"),
                };

                match elem_type {
                    ElementType::Float64 => ConstantValue::Float64(data.into_f64()),
                    ElementType::Float32 => ConstantValue::Float32(data.into_f32()),
                    ElementType::Int32 => ConstantValue::Int32(data.into_i32()),
                    ElementType::Int64 => ConstantValue::Int64(data.into_i64()),
                    ElementType::Bool => ConstantValue::Bool(data.into_bool()),
                    other => panic!("Unsupported constant scalar type: {other:?} "),
                }
            }
        };

        let out_ty = match (&output.ty, &const_value) {
            (
                ArgType::Tensor(t),
                ConstantValue::Float32(_)
                | ConstantValue::Float64(_)
                | ConstantValue::Int32(_)
                | ConstantValue::Int64(_)
                | ConstantValue::Bool(_),
            ) if t.rank == 0 => {
                let scalar_kind = match t.elem_type {
                    ElementType::Float32 => {
                        ScalarType::new(output.name.clone(), ScalarKind::Float32)
                    }
                    ElementType::Float64 => {
                        ScalarType::new(output.name.clone(), ScalarKind::Float64)
                    }
                    ElementType::Int32 => ScalarType::new(output.name.clone(), ScalarKind::Int32),
                    ElementType::Int64 => ScalarType::new(output.name.clone(), ScalarKind::Int64),
                    ElementType::Bool => ScalarType::new(output.name.clone(), ScalarKind::Bool),
                    _ => panic!("Unsupported scalar type for rank-0 tensor"),
                };
                Type::Scalar(scalar_kind)
            }
            _ => Type::from(output),
        };

        ConstantNode::new(node.name, const_value, out_ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{
        ScalarKind, ScalarType, ShapeType, TensorType, graph::BurnGraph, node::test::assert_tokens,
    };
    use burn::record::FullPrecisionSettings;
    use burn::tensor::TensorData;
    use quote::ToTokens;

    fn expected_tokens_constant_scalar(
        ty: TokenStream,
        val: TokenStream,
        output: TokenStream,
    ) -> TokenStream {
        quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> #ty {
                    let #output: #ty = #val;
                    #output
                }
            }
        }
    }

    fn assert_codegen_constant_scalar(constant: ConstantValue, scalar_kind: ScalarKind) {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();
        let val = constant.val_tokens();
        let ty = constant.ty_tokens();
        let output = Ident::new("output", Span::call_site());

        graph.register(ConstantNode::new(
            "constant_scalar".to_owned(),
            constant,
            Type::Scalar(ScalarType::new(output.to_string(), scalar_kind)),
        ));

        graph.register_input_output(vec![], vec![output.to_string()]);

        let expected = expected_tokens_constant_scalar(ty, val, output.to_token_stream());
        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_scalar_float32() {
        assert_codegen_constant_scalar(ConstantValue::Float32(3.14f32), ScalarKind::Float32);
    }

    #[test]
    fn test_codegen_constant_scalar_float64() {
        assert_codegen_constant_scalar(
            ConstantValue::Float64(3.111_222_333_444_555_f64),
            ScalarKind::Float64,
        );
    }

    #[test]
    fn test_codegen_constant_scalar_int32() {
        assert_codegen_constant_scalar(ConstantValue::Int32(123i32), ScalarKind::Int32);
    }

    #[test]
    fn test_codegen_constant_scalar_int64() {
        assert_codegen_constant_scalar(ConstantValue::Int64(42i64), ScalarKind::Int64);
    }

    #[test]
    fn test_codegen_constant_scalar_bool() {
        assert_codegen_constant_scalar(ConstantValue::Bool(true), ScalarKind::Bool);
        assert_codegen_constant_scalar(ConstantValue::Bool(false), ScalarKind::Bool);
    }

    #[test]
    fn test_codegen_constant_tensor_float() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_tensor = Ident::new("const_tensor", Span::call_site());
        let dimensions = 1;
        let data = TensorData::from([2f32, 2f32, 2f32, 2f32]);
        let tensor_type = TensorType::new_float(const_tensor.to_string(), dimensions);
        let constant = ConstantValue::Tensor(tensor_type.clone(), data);

        graph.register(ConstantNode::new(
            const_tensor.to_string(),
            constant.clone(),
            Type::Tensor(TensorType::new_float("output", dimensions)),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                const_tensor:  burn::module::Param<Tensor<B, 1>>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let const_tensor: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 1>::zeros([4], device),
                        device.clone(),
                        false
                    );

                    Self {
                        const_tensor,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 1> {
                    let output = self.const_tensor.val();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_tensor_int() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_tensor = Ident::new("const_tensor_int", Span::call_site());
        let dimensions = 1;
        let data = TensorData::from([1i32, 2i32, 3i32]);
        let tensor_type = TensorType::new_int(const_tensor.to_string(), dimensions);
        let constant = ConstantValue::Tensor(tensor_type.clone(), data);

        graph.register(ConstantNode::new(
            const_tensor.to_string(),
            constant.clone(),
            Type::Tensor(TensorType::new_int("output", dimensions)),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                const_tensor_int: burn::module::Param<Tensor<B, 1, Int>>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let const_tensor_int: burn::module::Param<Tensor<B, 1, Int>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 1, Int>::zeros([3], device),
                        device.clone(),
                        false
                    );

                    Self {
                        const_tensor_int,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 1, Int> {
                    let output = self.const_tensor_int.val();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_tensor_bool() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_tensor = Ident::new("const_tensor_3d", Span::call_site());
        let dimensions = 3;
        let data = TensorData::from([[[true, false], [true, false], [true, false]]]);
        let tensor_type = TensorType::new_bool(const_tensor.to_string(), dimensions);
        let constant = ConstantValue::Tensor(tensor_type.clone(), data);

        graph.register(ConstantNode::new(
            const_tensor.to_string(),
            constant.clone(),
            Type::Tensor(TensorType::new_bool("output", dimensions)),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                const_tensor_3d: burn::module::Param<Tensor<B, 3, Bool>>,
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    let const_tensor_3d: burn::module::Param<Tensor<B, 3, Bool>> = burn::module::Param::uninitialized(
                        burn::module::ParamId::new(),
                        move |device, _require_grad| Tensor::<B, 3, Bool>::empty([1, 3, 2], device),
                        device.clone(),
                        false
                    );

                    Self {
                        const_tensor_3d,
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> Tensor<B, 3, Bool> {
                    let output = self.const_tensor_3d.val();
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }

    #[test]
    fn test_codegen_constant_shape() {
        let mut graph = BurnGraph::<FullPrecisionSettings>::default();

        let const_shape = Ident::new("const_shape", Span::call_site());
        let shape_values = vec![2, 3, 4];
        let rank = shape_values.len();
        let constant = ConstantValue::Shape(shape_values.clone());

        graph.register(ConstantNode::new(
            const_shape.to_string(),
            constant.clone(),
            Type::Shape(ShapeType::new("output", rank)),
        ));

        graph.register_input_output(vec![], vec!["output".to_string()]);

        let expected = quote! {
            use burn::prelude::*;

            #[derive(Module, Debug)]
            pub struct Model<B: Backend> {
                phantom: core::marker::PhantomData<B>,
                device: burn::module::Ignored<B::Device>,
            }

            impl<B: Backend> Model<B> {
                #[allow(unused_variables)]
                pub fn new(device: &B::Device) -> Self {
                    Self {
                        phantom: core::marker::PhantomData,
                        device: burn::module::Ignored(device.clone()),
                    }
                }

                #[allow(clippy::let_and_return, clippy::approx_constant)]
                pub fn forward(&self) -> [i64; 3] {
                    let output: [i64; 3] = [2i64, 3i64, 4i64];
                    output
                }
            }
        };

        assert_tokens(graph.codegen(), expected);
    }
}
