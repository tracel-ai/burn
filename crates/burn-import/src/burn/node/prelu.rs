use super::prelude::*;
use burn::{
    module::{ConstantRecord, Param, ParamId},
    nn::PReluRecord,
    record::{PrecisionSettings, Record},
    tensor::Tensor,
};
use serde::Serialize;

impl<PS: PrecisionSettings> NodeCodegen<PS> for onnx_ir::prelu::PReluNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn field(&self) -> Option<Field> {
        let name = Ident::new(&self.name, Span::call_site());

        Some(Field::new(
            self.name.clone(),
            quote! {
                PRelu<B>
            },
            quote! {
                let #name = PReluConfig::new()
                    .init(device);
            },
        ))
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
