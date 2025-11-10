use super::{Node, NodeCodegen, OnnxIntoNode, try_convert_onnx_node};
use crate::burn::{BurnImports, Scope, TensorType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

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
    let mut unsupported_ops = vec![];

    // Helper to extract tensor types
    fn to_tensor(ty: Type) -> Option<TensorType> {
        match ty {
            Type::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    // Convert ONNX nodes to Burn nodes
    let burn_nodes: Vec<_> = subgraph
        .nodes
        .iter()
        .map(|node| {
            try_convert_onnx_node::<PS>(node.clone()).unwrap_or_else(|| {
                unsupported_ops.push(node.node_type.clone());
                panic!("Unsupported op in subgraph: {:?}", node.node_type)
            })
        })
        .collect();

    if !unsupported_ops.is_empty() {
        panic!("Unsupported ops in subgraph: {unsupported_ops:?}");
    }

    // Register subgraph inputs in scope (they reference parent scope values)
    for input in &subgraph.inputs {
        if let Some(tensor) = to_tensor(Type::from(input)) {
            // Register the input variable
            scope.tensor_register_variable(&tensor, node_position);
        }
    }

    // Build scope for subgraph nodes: register outputs and future uses
    for (idx, burn_node) in burn_nodes.iter().enumerate() {
        let subgraph_node_pos = node_position + idx + 1;

        // Register node outputs
        for output in burn_node.output_types() {
            if let Some(tensor) = to_tensor(output) {
                scope.tensor_register_variable(&tensor, subgraph_node_pos);
            }
        }

        // Register future uses of node inputs
        for input in burn_node.input_types() {
            if let Some(tensor) = to_tensor(input) {
                scope.tensor_register_future_use(&tensor, subgraph_node_pos - 1);
            }
        }
    }

    // Register future uses for subgraph outputs
    for output in &subgraph.outputs {
        if let Some(tensor) = to_tensor(Type::from(output)) {
            scope.tensor_register_future_use(&tensor, node_position + burn_nodes.len());
        }
    }

    // Generate forward code for each node
    for (idx, burn_node) in burn_nodes.iter().enumerate() {
        let node_code = burn_node.forward(scope, node_position + idx + 1);
        body.extend(node_code);
    }

    // Generate output tuple
    let output_names: Vec<_> = subgraph
        .outputs
        .iter()
        .map(|output| {
            let name = syn::Ident::new(&output.name, proc_macro2::Span::call_site());
            quote! { #name }
        })
        .collect();

    let output_tuple = if output_names.len() == 1 {
        output_names[0].clone()
    } else {
        quote! { (#(#output_names),*) }
    };

    (body, output_tuple)
}

/// If node - conditional execution with then/else branches
///
/// The If operation executes either the then_branch or else_branch subgraph
/// based on a boolean condition input.
#[derive(Debug, Clone)]
pub struct IfNode {
    pub condition: Type,
    pub inputs: Vec<Type>,
    pub outputs: Vec<Type>,
    pub then_branch: onnx_ir::OnnxGraph,
    pub else_branch: onnx_ir::OnnxGraph,
}

impl IfNode {
    pub fn new(
        condition: Type,
        inputs: Vec<Type>,
        outputs: Vec<Type>,
        then_branch: onnx_ir::OnnxGraph,
        else_branch: onnx_ir::OnnxGraph,
    ) -> Self {
        Self {
            condition,
            inputs,
            outputs,
            then_branch,
            else_branch,
        }
    }
}

impl<PS: PrecisionSettings + 'static> NodeCodegen<PS> for IfNode {
    fn output_types(&self) -> Vec<Type> {
        self.outputs.clone()
    }

    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![self.condition.clone()];
        inputs.extend(self.inputs.clone());
        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let cond = match &self.condition {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            Type::Tensor(tensor) => {
                let cond_tensor = scope.tensor_use_owned(tensor, node_position);
                // Convert tensor to bool - assume it's a scalar tensor
                quote! { #cond_tensor.into_scalar().elem::<bool>() }
            }
            other => panic!("If condition must be scalar or tensor, got {:?}", other),
        };

        // Generate code for then and else branches
        // Field name uniqueness is already handled at the onnx-ir layer
        let (then_body, then_output) =
            generate_subgraph_code::<PS>(&self.then_branch, scope, node_position);
        let (else_body, else_output) =
            generate_subgraph_code::<PS>(&self.else_branch, scope, node_position);

        // Generate output variable declarations
        // Output names must match the Type objects so other nodes can reference them
        let output_names: Vec<_> = self
            .outputs
            .iter()
            .map(|output| match output {
                Type::Tensor(t) => {
                    let name = &t.name;
                    quote! { #name }
                }
                Type::Scalar(s) => {
                    let name = &s.name;
                    quote! { #name }
                }
                _ => panic!("Unsupported output type in If node"),
            })
            .collect();

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

    fn into_node(self) -> Node<PS> {
        Node::If(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from subgraph nodes
        // Helper to register imports from a subgraph
        let mut register_subgraph_imports = |subgraph: &onnx_ir::OnnxGraph| {
            for onnx_node in &subgraph.nodes {
                if let Some(burn_node) = try_convert_onnx_node::<PS>(onnx_node.clone()) {
                    burn_node.register_imports(imports);
                }
            }
        };

        register_subgraph_imports(&self.then_branch);
        register_subgraph_imports(&self.else_branch);
    }
}

impl OnnxIntoNode for IfNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        // Extract condition input (always first input)
        let condition = Type::from(node.inputs.first().unwrap());

        // Get then_branch and else_branch from config
        let config = node.config::<onnx_ir::node::if_node::IfConfig>();
        let then_branch = config.then_branch.clone();
        let else_branch = config.else_branch.clone();

        // IMPORTANT: In ONNX, If nodes have implicit variable capture
        // The ONNX If node itself only lists explicit inputs (e.g., just condition)
        // But subgraphs can reference ANY value from parent scope via their input declarations
        //
        // For example:
        //   If node inputs: [condition]
        //   Subgraph inputs: [x_input]  <- this references a value in parent scope
        //
        // For Burn code generation, we need to make these captured variables EXPLICIT
        // inputs to the If node. This is necessary because:
        // 1. The burn-import graph builder needs to know all inputs to track data flow
        // 2. Identity node elimination needs to be able to rewire captured variables
        //
        // So we collect all subgraph inputs and add them as explicit inputs to the If node.

        // Collect explicit inputs from ONNX If node (after condition)
        let mut inputs: Vec<Type> = node.inputs.iter().skip(1).map(Type::from).collect();

        // Add captured variables from subgraph inputs
        // Use then_branch as canonical (both branches should have same inputs)
        for subgraph_input in &then_branch.inputs {
            inputs.push(Type::from(subgraph_input));
        }

        // Extract outputs from node
        let outputs: Vec<Type> = node.outputs.iter().map(Type::from).collect();

        Self::new(condition, inputs, outputs, then_branch, else_branch)
    }
}

// Tests will be added when full implementation is complete
// For now, integration tests with actual ONNX models will validate the implementation
