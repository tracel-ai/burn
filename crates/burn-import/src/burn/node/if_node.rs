use super::prelude::*;
use onnx_ir::Node;

/// Generate inline code for a subgraph
///
/// Converts an OnnxGraph into a TokenStream that can be inserted into an if/else branch.
/// Returns (body_code, output_tuple)
fn generate_subgraph_code<PS: PrecisionSettings + 'static>(
    subgraph: &onnx_ir::OnnxGraph,
    scope: &mut Scope,
    node_position: usize,
) -> (TokenStream, TokenStream) {
    let mut body = quote! {};

    // Register subgraph inputs in scope (they reference parent scope values)
    for input in &subgraph.inputs {
        if let ArgType::Tensor(_) = &input.ty {
            scope.tensor_register_variable(input, node_position);
        }
    }

    // Build scope for subgraph nodes: register outputs and future uses
    for (idx, node) in subgraph.nodes.iter().enumerate() {
        let subgraph_node_pos = node_position + idx + 1;

        // Register node outputs
        for output in <Node as NodeCodegen<PS>>::outputs(node) {
            if let ArgType::Tensor(_) = &output.ty {
                scope.tensor_register_variable(output, subgraph_node_pos);
            }
        }

        // Register future uses of node inputs
        // Filter to only dynamic/constant inputs (exclude static-only initializers)
        for input in <Node as NodeCodegen<PS>>::inputs(node)
            .iter()
            .filter(|arg| arg.is_dynamic() || arg.is_constant())
        {
            if let ArgType::Tensor(_) = &input.ty {
                scope.tensor_register_future_use(input, subgraph_node_pos - 1);
            }
        }
    }

    // Register future uses for subgraph outputs
    for output in &subgraph.outputs {
        if let ArgType::Tensor(_) = &output.ty {
            scope.tensor_register_future_use(output, node_position + subgraph.nodes.len());
        }
    }

    // Generate forward code for each node
    for (idx, node) in subgraph.nodes.iter().enumerate() {
        let node_code = <Node as NodeCodegen<PS>>::forward(node, scope, node_position + idx + 1);
        body.extend(node_code);
    }

    // Generate output tuple
    let output_names: Vec<_> = subgraph.outputs.iter().map(arg_to_ident).collect();

    let output_tuple = if output_names.len() == 1 {
        let out = &output_names[0];
        quote! { #out }
    } else {
        quote! { (#(#output_names),*) }
    };

    (body, output_tuple)
}

impl<PS: PrecisionSettings + 'static> NodeCodegen<PS> for onnx_ir::node::if_node::IfNode {
    fn inputs(&self) -> &[Argument] {
        &self.inputs
    }

    fn outputs(&self) -> &[Argument] {
        &self.outputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        // Get condition input
        let cond_arg = self.inputs.first().unwrap();

        let cond = match &cond_arg.ty {
            ArgType::Scalar(_) => {
                let name = arg_to_ident(cond_arg);
                quote! { #name }
            }
            ArgType::Tensor(_) => {
                let cond_tensor = scope.tensor_use_owned(cond_arg, node_position);
                // Convert tensor to bool - assume it's a scalar tensor
                quote! { #cond_tensor.into_scalar().elem::<bool>() }
            }
            other => panic!("If condition must be scalar or tensor, got {:?}", other),
        };

        // Generate code for then and else branches
        let (then_body, then_output) =
            generate_subgraph_code::<PS>(&self.config.then_branch, scope, node_position);
        let (else_body, else_output) =
            generate_subgraph_code::<PS>(&self.config.else_branch, scope, node_position);

        // Generate output variable declarations
        let output_names: Vec<_> = self.outputs.iter().map(arg_to_ident).collect();

        let output_decls = if self.outputs.len() == 1 {
            let out = &output_names[0];
            quote! { let #out }
        } else {
            quote! { let (#(#output_names),*) }
        };

        quote! {
            #output_decls = if #cond {
                #then_body
                #then_output
            } else {
                #else_body
                #else_output
            };
        }
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from subgraph nodes
        let mut register_subgraph_imports = |subgraph: &onnx_ir::OnnxGraph| {
            for node in &subgraph.nodes {
                <Node as NodeCodegen<PS>>::register_imports(node, imports);
            }
        };

        register_subgraph_imports(&self.config.then_branch);
        register_subgraph_imports(&self.config.else_branch);
    }
}
