use super::prelude::*;
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::BatchNormRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::batch_norm::BatchNormalizationNode {
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
        let momentum = self.config.momentum;

        Some(Field::new(
            self.name.clone(),
            quote! {
                BatchNorm<B>
            },
            quote! {
                let #name = BatchNormConfig::new(#num_features)
                    .with_epsilon(#epsilon)
                    .with_momentum(#momentum)
                    .init(device);
            },
        ))
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

    fn forward(&self, scope: &mut ScopeAtPosition<'_>) -> TokenStream {
        let input = scope.arg(self.inputs.first().unwrap());
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::batch_norm::{
        BatchNormConfig, BatchNormalizationNode, BatchNormalizationNodeBuilder,
    };

    fn create_batch_norm_node(name: &str) -> BatchNormalizationNode {
        let config = BatchNormConfig::new(64, 1e-5, 0.9);

        BatchNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_batch_norm_forward() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.batch_norm1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_batch_norm_forward_with_clone() {
        let node = create_batch_norm_node("batch_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = self.batch_norm1.forward(input.clone());
            output
        }
        ");
    }
}
