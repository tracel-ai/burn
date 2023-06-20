use super::{Node, NodeCodegen};
use crate::burn::{OtherType, Scope, Type};
use burn::record::PrecisionSettings;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

#[derive(Debug, Clone)]
pub struct ConstantNode {
    pub name: String,
    pub value: ConstantValue,
    output_ty: OtherType,
}

#[derive(Debug, Clone, new)]
pub enum ConstantValue {
    Int(i32),
    Float(f32),
    Bool(bool),
}

impl ConstantValue {
    pub fn ty_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Int(_) => quote! { i32 },
            ConstantValue::Float(_) => quote! { f32 },
            ConstantValue::Bool(_) => quote! { bool },
        }
    }
    pub fn val_tokens(&self) -> TokenStream {
        match self {
            ConstantValue::Int(val) => quote! { #val },
            ConstantValue::Float(val) => quote! { #val },
            ConstantValue::Bool(val) => quote! { #val },
        }
    }
}

impl ConstantNode {
    pub fn new(name: String, value: ConstantValue) -> Self {
        let output_ty = OtherType::new(name.clone(), value.ty_tokens());

        Self {
            name,
            value,
            output_ty,
        }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for ConstantNode {
    fn output_types(&self) -> Vec<Type> {
        vec![Type::Other(&self.output_ty)]
    }

    fn input_types(&self) -> Vec<Type> {
        vec![]
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let name = Ident::new(self.name.as_ref(), Span::call_site());
        let val = self.value.val_tokens();
        let ty = self.value.ty_tokens();

        quote! {
            let #name: #ty = #val;
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Constant(self)
    }
}
