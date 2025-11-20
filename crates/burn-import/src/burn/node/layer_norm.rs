use super::prelude::*;
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::LayerNormRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::layer_norm::LayerNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_features = self.config.d_model.to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! {
                LayerNorm<B>
            },
            quote! {
                let #name = LayerNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();

        let gamma = extract_node_data(&self.inputs, 1).expect("Gamma is required");
        let beta = extract_node_data(&self.inputs, 2);

        let record = LayerNormRecord::<SerializationBackend> {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::from_data(gamma.clone().convert::<PS::FloatElem>(), &device),
            ),
            beta: Param::initialized(
                ParamId::new(),
                if let Some(beta) = beta {
                    Tensor::from_data(beta.convert::<PS::FloatElem>(), &device)
                } else {
                    Tensor::zeros([self.config.d_model], &device)
                },
            ),
            epsilon: ConstantRecord::new(),
        };

        let item = Record::into_item::<PS>(record);
        item.serialize(serializer)
    }

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
        let output = arg_to_ident(self.outputs.first().unwrap());
        let field = Ident::new(&self.name, Span::call_site());

        if self.config.full_precision {
            quote! {
                let #output = {
                    let dtype = #input.dtype();
                    self.#field.forward(#input.cast(burn::tensor::DType::F32)).cast(dtype)
                };
            }
        } else {
            quote! {
                let #output = self.#field.forward(#input);
            }
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        imports.register("burn::nn::LayerNorm");
        imports.register("burn::nn::LayerNormConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::layer_norm::{
        LayerNormConfig, LayerNormalizationNode, LayerNormalizationNodeBuilder,
    };

    fn create_layer_norm_node(name: &str) -> LayerNormalizationNode {
        let config = LayerNormConfig::new(512, 1e-5, true);

        LayerNormalizationNodeBuilder::new(name)
            .input_tensor("input", 3, DType::F32)
            .output_tensor("output", 3, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_layer_norm_forward() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        let output = {
                let dtype = input.dtype();
                self.layer_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
        ");
    }

    #[test]
    fn test_layer_norm_forward_with_clone() {
        let node = create_layer_norm_node("layer_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        let output = {
                let dtype = input.clone().dtype();
                self.layer_norm1
                    .forward(input.clone().cast(burn::tensor::DType::F32))
                    .cast(dtype)
            };
        ");
    }
}
