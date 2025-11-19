use super::{NodeCodegen, SerializationBackend, arg_to_ident, extract_node_data};
use crate::burn::{BurnImports, Field, Scope, ToTokens};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::InstanceNormRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use onnx_ir::Argument;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS>
    for onnx_ir::node::instance_norm::InstanceNormalizationNode
{
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! {
                InstanceNorm<B>
            },
            quote! {
                let #name = InstanceNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();

        let gamma = extract_node_data(&self.inputs, 1).expect("Gamma is required");
        let beta = extract_node_data(&self.inputs, 2).expect("Beta is required");

        let record = InstanceNormRecord::<SerializationBackend> {
            gamma: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(gamma.clone().convert::<PS::FloatElem>(), &device),
            )),
            beta: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(beta.clone().convert::<PS::FloatElem>(), &device),
            )),
            epsilon: ConstantRecord::new(),
            num_channels: ConstantRecord::new(),
            affine: ConstantRecord::new(),
        };

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(self.inputs.first().unwrap(), node_position);
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        quote! {
            let #output = self.#field.forward(#input);
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::InstanceNorm");
        imports.register("burn::nn::InstanceNormConfig");
    }
}
