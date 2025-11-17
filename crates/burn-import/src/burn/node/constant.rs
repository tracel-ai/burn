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
                            false,
                            #shape.into(),
                        );
                    }),
                    crate::burn::TensorKind::Float => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank>::zeros(#shape, device),
                            device.clone(),
                            false,
                            #shape.into(),
                        );
                    }),
                    crate::burn::TensorKind::Bool => Some(quote! {
                        let #name: burn::module::Param<#ty> = burn::module::Param::uninitialized(
                            burn::module::ParamId::new(),
                            move |device, _require_grad| Tensor::<B, #rank, Bool>::empty(#shape, device),
                            device.clone(),
                            false,
                            #shape.into(),
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
        use onnx_ir::ir::{ArgType, DType};

        let onnx_ir::Node::Constant(n) = &node else {
            panic!("Expected Constant node");
        };
        let input = n.inputs.first().unwrap();
        let output = n.outputs.first().unwrap();

        // Get the tensor data from the central store via the input argument
        let tensor_data = if let Some(data) = input.value() {
            data
        } else {
            panic!("Constant node '{}' input missing tensor data", n.name);
        };

        // Helper to map elem type to ConstantValue (single scalar)
        // Helper to extract scalar from TensorData
        fn scalar_from_tensor_data(
            elem: DType,
            tensor_data: &onnx_ir::TensorData,
        ) -> ConstantValue {
            match elem {
                DType::F64 => {
                    let val = tensor_data.as_slice::<f64>().unwrap()[0];
                    ConstantValue::Float64(val)
                }
                DType::F32 => {
                    let val = tensor_data.as_slice::<f32>().unwrap()[0];
                    ConstantValue::Float32(val)
                }
                DType::I64 => {
                    let val = tensor_data.as_slice::<i64>().unwrap()[0];
                    ConstantValue::Int64(val)
                }
                DType::I32 => {
                    let val = tensor_data.as_slice::<i32>().unwrap()[0];
                    ConstantValue::Int32(val)
                }
                DType::Bool => {
                    let val = tensor_data.as_slice::<bool>().unwrap()[0];
                    ConstantValue::Bool(val)
                }
                DType::U8 => {
                    let val = tensor_data.as_slice::<u8>().unwrap()[0] as i32;
                    ConstantValue::Int32(val)
                }
                DType::I8 => {
                    let val = tensor_data.as_slice::<i8>().unwrap()[0] as i32;
                    ConstantValue::Int32(val)
                }
                _ => panic!("Unsupported scalar type: {elem:?}"),
            }
        }

        let const_value = match &output.ty {
            ArgType::Shape(rank) => {
                let shape_values: Vec<usize> = tensor_data
                    .to_vec::<i64>()
                    .unwrap()
                    .into_iter()
                    .map(|v| v as usize)
                    .collect();
                assert_eq!(shape_values.len(), *rank, "Shape constant rank mismatch");
                ConstantValue::Shape(shape_values)
            }

            ArgType::Tensor(tensor) => {
                if tensor.rank == 0 {
                    // Extract scalar data from tensor_data
                    scalar_from_tensor_data(tensor.dtype, &tensor_data)
                } else {
                    let kind: TensorKind = tensor.dtype.into();
                    let rank = tensor.rank;

                    let serialized_data = match &tensor.dtype {
                        DType::F32 | DType::F64 | DType::F16 => {
                            tensor_data.clone().convert::<f32>()
                        }
                        DType::I32 | DType::I64 | DType::U16 | DType::U8 | DType::I8 => {
                            tensor_data.clone().convert::<i32>()
                        }
                        DType::Bool => tensor_data.clone(),
                        other => panic!("Unsupported constant tensor type: {:?} ", other),
                    };

                    ConstantValue::Tensor(
                        TensorType::new(n.name.clone(), rank, kind),
                        serialized_data,
                    )
                }
            }

            ArgType::Scalar(elem_type) => {
                // Extract scalar data from tensor_data
                scalar_from_tensor_data(*elem_type, &tensor_data)
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
                let scalar_kind = match t.dtype {
                    DType::F32 => ScalarType::new(output.name.clone(), ScalarKind::Float32),
                    DType::F64 => ScalarType::new(output.name.clone(), ScalarKind::Float64),
                    DType::I32 => ScalarType::new(output.name.clone(), ScalarKind::Int32),
                    DType::I64 => ScalarType::new(output.name.clone(), ScalarKind::Int64),
                    DType::Bool => ScalarType::new(output.name.clone(), ScalarKind::Bool),
                    _ => panic!("Unsupported scalar type for rank-0 tensor"),
                };
                Type::Scalar(scalar_kind)
            }
            _ => Type::from(output),
        };

        ConstantNode::new(n.name.clone(), const_value, out_ty)
    }
}
