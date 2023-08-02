use super::{Node, NodeCodegen};
use crate::burn::{ScalarKind, ScalarType, Scope, TensorType, ToTokens, Type};
use burn::{record::PrecisionSettings, tensor::DataSerialize};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ConstantNode<PS: PrecisionSettings> {
    pub name: String,
    pub value: ConstantValue<PS>,
    pub output: Type,
}

#[derive(Debug, Clone)]
pub enum TensorValue<PS: PrecisionSettings> {
    Float(DataSerialize<PS::FloatElem>),
    Int(DataSerialize<PS::IntElem>),
}

#[derive(Debug, Clone, new)]
pub enum ConstantValue<PS: PrecisionSettings> {
    /// Float constant.
    Float32(f32),

    /// Integer constant.
    Int64(i64),

    /// Tensor constant (does not hold a value).
    Tensor(TensorType, TensorValue<PS>),
}

impl<PS: PrecisionSettings> ConstantValue<PS> {
    pub fn ty_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Float32(_) => quote! { f32 },
            ConstantValue::Int64(_) => quote! { i64 },
            ConstantValue::Tensor(tensor_type, _) => {
                let ty = tensor_type.ty();
                quote! { #ty}
            }
        }
    }
    pub fn val_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Float32(val) => quote! { #val },
            ConstantValue::Int64(val) => quote! { #val },
            ConstantValue::Tensor(_, _) => {
                panic!("Tensor constant is not assignable.")
            }
        }
    }
}

impl<PS: PrecisionSettings> ConstantNode<PS> {
    pub fn new(name: String, value: ConstantValue<PS>, output: Type) -> Self {
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
            ConstantValue::Int64(_) => Type::Scalar(ScalarType {
                name,
                kind: ScalarKind::Int64,
            }),
            ConstantValue::Tensor(tensor_type, _) => Type::Tensor(tensor_type.clone()),
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantNode<PS> {
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

    fn field_init(&self, with_record: bool) -> Option<TokenStream> {
        match &self.value {
            ConstantValue::Tensor(tensor_type, _) => {
                let ty = tensor_type.ty();
                let name = Ident::new(self.name.as_ref(), Span::call_site());
                let shape = tensor_type.clone().shape.unwrap().to_tokens();
                let dim = tensor_type.clone().dim.to_tokens();

                if with_record {
                    Some(quote! {
                        let #name = record.#name;
                    })
                } else {
                    Some(quote! {
                        let #name: #ty = Tensor::<B, #dim>::zeros(#shape);
                    })
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
                    let #output = self.#name.clone();
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
        match &self.value {
            ConstantValue::Tensor(_tensor_type, ds) => match ds {
                TensorValue::Float(ds) => ds.serialize(serializer),
                TensorValue::Int(ds) => ds.serialize(serializer),
            },
            _ => S::serialize_none(serializer),
        }
    }
}

// TODO add test missing for constant node (@antimora 8/2/2023)
