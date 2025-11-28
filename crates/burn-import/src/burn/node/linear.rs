use super::prelude::*;
use burn::{
    module::{Param, ParamId},
    nn::LinearRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::linear::LinearNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let d_input = self.config.d_input.to_tokens();
        let d_output = self.config.d_output.to_tokens();
        let bias = self.config.bias;

        Some(Field::new(
            self.name.clone(),
            quote! { Linear<B> },
            quote! {
                let #name = LinearConfig::new(#d_input, #d_output)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();
        let data_weights = extract_node_data(&self.inputs, 1).unwrap();
        let data_bias = extract_node_data(&self.inputs, 2);

        let record = LinearRecord::<SerializationBackend> {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::from_data(data_weights.clone().convert::<PS::FloatElem>(), &device),
            ),
            bias: data_bias.as_ref().map(|bias| {
                Param::initialized(
                    ParamId::new(),
                    Tensor::from_data(bias.clone().convert::<PS::FloatElem>(), &device),
                )
            }),
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
        imports.register("burn::nn::Linear");
        imports.register("burn::nn::LinearConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::ir::{ArgType, Argument, TensorType};
    use onnx_ir::linear::{LinearConfig, LinearNode};

    #[test]
    fn test_linear_forward() {
        let config = LinearConfig::new(128, 64, true);
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let weight = Argument::new(
            "weight",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let bias = Argument::new(
            "bias",
            ArgType::Tensor(TensorType::new(DType::F32, 1, None)),
        );

        let node = LinearNode {
            name: "linear1".to_string(),
            inputs: vec![input, weight, bias],
            outputs: vec![Argument::new(
                "output",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(
            &self,
            input: Tensor<B, 2>,
            weight: Tensor<B, 2>,
            bias: Tensor<B, 1>,
        ) -> Tensor<B, 2> {
            let output = self.linear1.forward(input);
            output
        }
        ");
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let config = LinearConfig::new(128, 64, false);
        let input = Argument::new(
            "input",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );
        let weight = Argument::new(
            "weight",
            ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
        );

        let node = LinearNode {
            name: "linear2".to_string(),
            inputs: vec![input, weight],
            outputs: vec![Argument::new(
                "output",
                ArgType::Tensor(TensorType::new(DType::F32, 2, None)),
            )],
            config,
        };
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @r"
        pub fn forward(&self, input: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
            let output = self.linear2.forward(input);
            output
        }
        ");
    }
}
