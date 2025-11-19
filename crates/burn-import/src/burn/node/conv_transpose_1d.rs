use super::{NodeCodegen, SerializationBackend, arg_to_ident, extract_node_data};
use crate::burn::{BurnImports, Field, Scope, ToTokens};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::conv::ConvTranspose1dRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use onnx_ir::Argument;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS>
    for onnx_ir::node::conv_transpose1d::ConvTranspose1dNode
{
    fn inputs(&self) -> &[Argument] {
        // Filter inputs only dynamic and constant
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());
        let channels_in = self.config.channels_in.to_tokens();
        let channels_out = self.config.channels_out.to_tokens();
        let kernel_size = self.config.kernel_size.to_tokens();
        let stride = self.config.stride.to_tokens();
        let dilation = self.config.dilation.to_tokens();
        let groups = self.config.groups.to_tokens();
        let padding = self.config.padding.to_tokens();
        let padding_out = self.config.padding_out.to_tokens();
        let bias = self.config.bias;

        Some(Field::new(
            self.name.clone(),
            quote! {
                ConvTranspose1d<B>
            },
            quote! {
                let #name = ConvTranspose1dConfig::new([#channels_in, #channels_out], #kernel_size)
                    .with_stride(#stride)
                    .with_padding(#padding)
                    .with_padding_out(#padding_out)
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
        let record = ConvTranspose1dRecord::<SerializationBackend> {
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
            stride: ConstantRecord::new(),
            kernel_size: ConstantRecord::new(),
            dilation: ConstantRecord::new(),
            groups: ConstantRecord::new(),
            padding: ConstantRecord::new(),
            padding_out: ConstantRecord::new(),
            channels: [ConstantRecord::new(), ConstantRecord::new()],
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
        imports.register("burn::nn::conv::ConvTranspose1d");
        imports.register("burn::nn::conv::ConvTranspose1dConfig");
    }
}
