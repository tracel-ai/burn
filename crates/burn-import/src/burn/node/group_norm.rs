use super::prelude::*;
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::GroupNormRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::node::group_norm::GroupNormalizationNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let num_groups = self.config.num_groups.to_tokens();
        let num_features = self.config.num_features.to_tokens();
        let epsilon = self.config.epsilon;

        Some(Field::new(
            self.name.clone(),
            quote! {
                GroupNorm<B>
            },
            quote! {
                let #name = GroupNormConfig::new(#num_groups, #num_features)
                    .with_epsilon(#epsilon)
                    .init(device);
            },
        ))
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();

        let gamma = extract_node_data(&self.inputs, 1).expect("Gamma is required");
        let beta = extract_node_data(&self.inputs, 2).expect("Beta is required");

        let record = GroupNormRecord::<SerializationBackend> {
            gamma: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(gamma.clone().convert::<PS::FloatElem>(), &device),
            )),
            beta: Some(Param::initialized(
                ParamId::new(),
                Tensor::from_data(beta.clone().convert::<PS::FloatElem>(), &device),
            )),
            epsilon: ConstantRecord::new(),
            num_groups: ConstantRecord::new(),
            num_channels: ConstantRecord::new(),
            affine: ConstantRecord::new(),
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
        imports.register("burn::nn::GroupNorm");
        imports.register("burn::nn::GroupNormConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::node::group_norm::{
        GroupNormConfig, GroupNormalizationNode, GroupNormalizationNodeBuilder,
    };

    fn create_group_norm_node(name: &str) -> GroupNormalizationNode {
        let config = GroupNormConfig::new(64, 8, 1e-5, true);

        GroupNormalizationNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_group_norm_forward() {
        let node = create_group_norm_node("group_norm1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let dtype = input.dtype();
                self.group_norm1.forward(input.cast(burn::tensor::DType::F32)).cast(dtype)
            };
            output
        }
        ");
    }

    #[test]
    fn test_group_norm_forward_with_clone() {
        let node = create_group_norm_node("group_norm1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
            let output = {
                let dtype = input.clone().dtype();
                self.group_norm1
                    .forward(input.clone().cast(burn::tensor::DType::F32))
                    .cast(dtype)
            };
            output
        }
        ");
    }
}
