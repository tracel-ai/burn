use super::{Node, NodeCodegen, OnnxIntoNode};
use crate::burn::{BurnImports, Scope, TensorType, ToTokens, Type};
use burn::record::PrecisionSettings;
use onnx_ir::node::split::SplitConfig;
use proc_macro2::TokenStream;
use quote::quote;

#[derive(Debug, Clone, new)]
pub struct SplitNode {
    pub input: TensorType,
    pub outputs: Vec<TensorType>,
    pub config: SplitConfig,
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for SplitNode {
    fn output_types(&self) -> Vec<Type> {
        self.outputs
            .iter()
            .map(|t| Type::Tensor(t.clone()))
            .collect()
    }

    fn input_types(&self) -> Vec<Type> {
        vec![Type::Tensor(self.input.clone())]
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let input = scope.tensor_use_owned(&self.input, node_position);
        let axis = self.config.axis.to_tokens();

        let outputs = self
            .outputs
            .iter()
            .map(|t| t.name.clone())
            .collect::<Vec<_>>();

        let unpack_outputs = quote! {
            let [#(#outputs),*] = split_tensors.try_into().unwrap();
        };

        if let Some(split_sizes_input) = &self.config.split_sizes {
            // Extract static split sizes from the enum wrapper
            let split_sizes = match split_sizes_input {
                onnx_ir::node::split::SplitSizesInput::Static(sizes) => sizes,
                onnx_ir::node::split::SplitSizesInput::Runtime(_) => {
                    panic!("Runtime split sizes are not supported in burn-import")
                }
            };
            let split_sizes_tokens = split_sizes.iter().map(|s| s.to_tokens());
            quote! {
                let split_tensors = #input.split_with_sizes(vec![#(#split_sizes_tokens),*], #axis);
                #unpack_outputs
            }
        } else {
            let split_size = &self.config.split_size.unwrap();
            let split_size_tokens = split_size.to_tokens();
            quote! {
                let split_tensors = #input.split(#split_size_tokens, #axis);
                #unpack_outputs
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Split(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // When split_sizes is used, we generate vec![...] which needs the vec macro
        if self.config.split_sizes.is_some() {
            imports.register("alloc::vec");
        }
    }
}

impl OnnxIntoNode for SplitNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        let onnx_ir::Node::Split(n) = &node else {
            panic!("Expected Split node");
        };
        let inputs = &n.inputs;
        let outputs = &n.outputs;
        let config = &n.config;
        let input = TensorType::from(inputs.first().unwrap());
        let outputs = outputs.iter().map(TensorType::from).collect();
        Self::new(input, outputs, config.clone())
    }
}
