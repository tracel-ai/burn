use super::{NodeCodegen, SerializationBackend, arg_to_ident, extract_node_data};
use crate::burn::{BurnImports, Field, Scope};
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::PReluRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use onnx_ir::Argument;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::prelu::PReluNode {
    fn inputs(&self) -> Vec<&Argument> {
        self.inputs
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
            .collect()
    }

    fn outputs(&self) -> Vec<&Argument> {
        self.outputs.iter().collect()
    }

    fn field(&self) -> Option<Field> {
        Some(Field::new(
            self.name.clone(),
            quote! {
                PRelu<B>
            },
        ))
    }

    fn field_init(&self) -> Option<TokenStream> {
        let name = Ident::new(&self.name, Span::call_site());

        // Get alpha from the second input to determine num_parameters
        let alpha_data = extract_node_data(&self.inputs, 1).expect("PRelu weight is required");
        let num_parameters = alpha_data.shape[0];

        let tokens = quote! {
            let #name = PReluConfig::new(#num_parameters)
                .init(device);
        };

        Some(tokens)
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let device = Default::default();

        let alpha = extract_node_data(&self.inputs, 1).expect("PRelu weight is required");

        let record = PReluRecord::<SerializationBackend> {
            alpha: Param::initialized(
                ParamId::new(),
                Tensor::from_data(alpha.clone().convert::<PS::FloatElem>(), &device),
            ),
            alpha_value: ConstantRecord,
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
        imports.register("burn::nn::PRelu");
        imports.register("burn::nn::PReluConfig");
    }
}
