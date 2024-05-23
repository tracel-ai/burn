use super::{Node, NodeCodegen};
use crate::burn::{OtherType, Scope, TensorType, Type};
use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone)]
pub struct RandomUniformNode {
    pub high: f64,
    pub low: f64,
    pub output_ty: TensorType,
    pub field_ty: Type,
}

impl RandomUniformNode {
    pub fn new<S: AsRef<str>>(name: S, output_ty: TensorType, high: f64, low: f64) -> Self {
        let field_type = quote! {
            burn::module::Ignored<Distribution>
        };
        Self {
            high,
            low,
            output_ty,
            field_ty: Type::Other(OtherType::new(name, field_type)),
        }
    }

    fn get_output_shape(&self) -> TokenStream {
        let shape_it = self
            .output_ty
            .shape
            .as_ref()
            .expect("RandomUniform output has no shape!")
            .iter();
        quote! { Shape::new([#(#shape_it),*]) }
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for RandomUniformNode {
    fn input_types(&self) -> Vec<Type> {
        Vec::with_capacity(0)
    }

    fn output_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.output_ty.clone())]
    }

    fn field_type(&self) -> Option<Type> {
        Some(self.field_ty.clone())
    }

    fn field_init(&self) -> Option<TokenStream> {
        let field_name = self.field_ty.name();
        let low = self.low;
        let high = self.high;

        Some(quote! {
            let #field_name = burn::module::Ignored(Distribution::Uniform(#low, #high));
        })
    }

    fn forward(&self, _scope: &mut Scope, _node_position: usize) -> TokenStream {
        let output = &self.output_ty.name;
        let field_name = self.field_ty.name();
        let shape = self.get_output_shape();
        quote! {
            let #output = Tensor::random(#shape, self.#field_name.0, self.device.deref());
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::RandomUniform(self)
    }

    fn register_imports(&self, imports: &mut crate::burn::BurnImports) {
        imports.register("core::ops::Deref");
        imports.register("burn::tensor::Distribution");
        imports.register("burn::prelude::Shape");
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        S::serialize_none(serializer)
        // should the Distribution be serialized here?
    }
}
