use super::{NodeCodegen, SerializationBackend, arg_to_ident, extract_node_data};
use crate::burn::{BurnImports, Field, Scope, ToTokens};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::BatchNormRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use onnx_ir::Argument;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::batch_norm::BatchNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        Some(Field::new(
            self.name.clone(),
            quote! {
                BatchNorm<B>
            },
        ))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;
        let momentum = self.config.momentum;

        let tokens = quote! {
            let #name = BatchNormConfig::new(#num_features)
                .with_epsilon(#epsilon)
                .with_momentum(#momentum)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();

        let gamma = extract_node_data(&self.inputs, 1).expect("Gamma is required");
        let beta = extract_node_data(&self.inputs, 2).expect("Beta is required");
        let running_mean = extract_node_data(&self.inputs, 3).expect("Running mean is required");
        let running_var = extract_node_data(&self.inputs, 4).expect("Running var is required");

        let record = BatchNormRecord::<SerializationBackend> {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::from_data(gamma.clone().convert::<PS::FloatElem>(), &device),
            ),
            beta: Param::initialized(
                ParamId::new(),
                Tensor::from_data(beta.clone().convert::<PS::FloatElem>(), &device),
            ),
            running_mean: Param::initialized(
                ParamId::new(),
                Tensor::from_data(running_mean.clone().convert::<PS::FloatElem>(), &device),
            ),
            running_var: Param::initialized(
                ParamId::new(),
                Tensor::from_data(running_var.clone().convert::<PS::FloatElem>(), &device),
            ),
            epsilon: ConstantRecord::new(),
            momentum: ConstantRecord::new(),
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
        imports.register("burn::nn::BatchNorm");
        imports.register("burn::nn::BatchNormConfig");
    }
}
