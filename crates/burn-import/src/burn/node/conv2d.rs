use super::prelude::*;
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::Conv2dRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::conv2d::Conv2dNode {
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let channels = self.config.channels.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let bias = self.config.bias;

        Some(Field::new(
            self.name.clone(),
            quote! {
                Conv2d<B>
            },
            quote! {
                let #name = Conv2dConfig::new(#channels, #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_dilation(#dilation)
                    .with_groups(#groups)
                    .with_bias(#bias)
                    .init(device);
            },
        ))
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();

        let data_weights = extract_node_data(&self.inputs, 1).unwrap();
        let has_bias = self.inputs.len() == 3;
        let data_bias = if has_bias {
            extract_node_data(&self.inputs, 2)
        } else {
            None
        };
        let record = Conv2dRecord::<SerializationBackend> {
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
            stride: [ConstantRecord::new(); 2],
            kernel_size: [ConstantRecord::new(); 2],
            dilation: [ConstantRecord::new(); 2],
            groups: ConstantRecord::new(),
            padding: ConstantRecord::new(),
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
        imports.register("burn::nn::PaddingConfig2d");
        imports.register("burn::nn::conv::Conv2d");
        imports.register("burn::nn::conv::Conv2dConfig");
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use burn::tensor::DType;
    use insta::assert_snapshot;
    use onnx_ir::conv2d::{Conv2dConfig, Conv2dNode, Conv2dNodeBuilder};
    use onnx_ir::padding::PaddingConfig2d;

    fn create_conv2d_node(name: &str) -> Conv2dNode {
        let config = Conv2dConfig::new(
            [3, 64],
            [3, 3],
            [1, 1],
            PaddingConfig2d::Explicit(1, 1),
            [1, 1],
            1,
            true,
        );

        Conv2dNodeBuilder::new(name)
            .input_tensor("input", 4, DType::F32)
            .output_tensor("output", 4, DType::F32)
            .config(config)
            .build()
    }

    #[test]
    fn test_conv2d_forward() {
        let node = create_conv2d_node("conv1");
        let code = codegen_forward_default(&node);
        assert_snapshot!(code, @"let output = self.conv1.forward(input);");
    }

    #[test]
    fn test_conv2d_forward_with_clone() {
        let node = create_conv2d_node("conv1");
        let code = codegen_forward_with_clone(&node);
        assert_snapshot!(code, @"let output = self.conv1.forward(input.clone());");
    }
}
