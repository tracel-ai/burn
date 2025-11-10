use super::{Node, NodeCodegen, OnnxIntoNode, try_convert_onnx_node};
use crate::burn::{BurnImports, Scope, TensorType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// Generate inline code for a loop body subgraph
///
/// Converts an OnnxGraph into a TokenStream that executes in each loop iteration.
/// Returns the body code
fn generate_loop_body_code<PS: PrecisionSettings + 'static>(
    subgraph: &onnx_ir::OnnxGraph,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
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
                panic!("Unsupported op in loop body: {:?}", node.node_type)
            })
        })
        .collect();

    if !unsupported_ops.is_empty() {
        panic!("Unsupported ops in loop body: {unsupported_ops:?}");
    }

    // Register subgraph inputs in scope (they reference loop variables)
    for input in &subgraph.inputs {
        if let Some(tensor) = to_tensor(Type::from(input)) {
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

    body
}

/// Loop node - iterative execution with loop-carried dependencies
///
/// The Loop operation executes a body subgraph for a specified number of iterations,
/// carrying state between iterations.
#[derive(Debug, Clone)]
pub struct LoopNode {
    pub max_trip_count: Type, // M - optional max iteration count
    pub condition: Type,      // cond - initial loop condition
    pub v_initial: Vec<Type>, // Loop-carried dependency initial values
    pub outputs: Vec<Type>,   // Final values of loop-carried dependencies
    pub body: onnx_ir::OnnxGraph,
}

impl LoopNode {
    pub fn new(
        max_trip_count: Type,
        condition: Type,
        v_initial: Vec<Type>,
        outputs: Vec<Type>,
        body: onnx_ir::OnnxGraph,
    ) -> Self {
        Self {
            max_trip_count,
            condition,
            v_initial,
            outputs,
            body,
        }
    }
}

impl<PS: PrecisionSettings + 'static> NodeCodegen<PS> for LoopNode {
    fn output_types(&self) -> Vec<Type> {
        self.outputs.clone()
    }

    fn input_types(&self) -> Vec<Type> {
        let mut inputs = vec![self.max_trip_count.clone(), self.condition.clone()];
        inputs.extend(self.v_initial.clone());
        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        // Extract max trip count (can be scalar or rank-1 tensor)
        let max_trip_count = match &self.max_trip_count {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            Type::Tensor(tensor) if tensor.rank == 1 => {
                // Rank-1 tensor with single element - extract the scalar value
                let name = &tensor.name;
                quote! { #name.clone().into_scalar().elem::<i64>() }
            }
            _ => panic!("Loop max_trip_count must be scalar i64 or rank-1 tensor"),
        };

        // Extract initial condition (can be scalar or rank-1 tensor)
        let initial_cond = match &self.condition {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! { #name }
            }
            Type::Tensor(tensor) if tensor.rank == 1 => {
                // Rank-1 tensor with single element - extract the scalar value
                let name = &tensor.name;
                quote! { #name.clone().into_scalar().elem::<bool>() }
            }
            _ => panic!("Loop condition must be scalar bool or rank-1 tensor"),
        };

        // Body inputs: [iter_num, cond_in, v_in...]
        // Body outputs: [cond_out, v_out..., scan_outputs...]
        //
        // Note: Not all v_in need to have corresponding v_out. Variables that are only
        // read (not modified) won't have outputs. The number of outputs corresponds to
        // the number of Loop node outputs (self.outputs.len()).

        // Number of loop-carried dependencies passed as input
        let num_loop_vars = self.v_initial.len();

        // Number of loop-carried dependencies that are actually output/modified
        let num_output_vars = self.outputs.len();

        // Body should have 2 + num_loop_vars inputs (iter, cond, v_in...)
        assert_eq!(
            self.body.inputs.len(),
            2 + num_loop_vars,
            "Loop body should have {} inputs, got {}",
            2 + num_loop_vars,
            self.body.inputs.len()
        );

        // Body should have at least 1 + num_output_vars outputs (cond_out, v_out...)
        // May have additional scan outputs
        assert!(
            self.body.outputs.len() > num_output_vars,
            "Loop body should have at least {} outputs, got {}",
            1 + num_output_vars,
            self.body.outputs.len()
        );

        // Initialize loop-carried dependency variables
        let mut init_stmts = quote! {};
        let loop_var_names: Vec<_> = self
            .body
            .inputs
            .iter()
            .skip(2) // Skip iter and cond_in
            .map(|arg| syn::Ident::new(&arg.name, proc_macro2::Span::call_site()))
            .collect();

        for (idx, initial_value) in self.v_initial.iter().enumerate() {
            let var_name = &loop_var_names[idx];
            let init_value = match initial_value {
                Type::Tensor(tensor) => {
                    let tensor_name = &tensor.name;
                    quote! { #tensor_name }
                }
                Type::Scalar(scalar) => {
                    let scalar_name = &scalar.name;
                    quote! { #scalar_name }
                }
                _ => panic!("Unsupported loop-carried dependency type"),
            };

            // Only variables that are updated by the loop body outputs need to be mutable
            // Read-only variables are cloned at the start of each iteration via shadowing
            if idx < num_output_vars {
                init_stmts.extend(quote! {
                    let mut #var_name = #init_value;
                });
            } else {
                init_stmts.extend(quote! {
                    let #var_name = #init_value;
                });
            }
        }

        // Initialize iteration counter and condition
        let iter_name = syn::Ident::new(&self.body.inputs[0].name, proc_macro2::Span::call_site());
        let cond_in_name =
            syn::Ident::new(&self.body.inputs[1].name, proc_macro2::Span::call_site());

        // Create a bool variable for the while condition (handles both scalar and rank-1 tensor)
        let cond_bool_name = syn::Ident::new(
            &format!("{}_bool", self.body.inputs[1].name),
            proc_macro2::Span::call_site(),
        );

        // For tensors, initial_cond already has .into_scalar() applied
        // For scalars, it's just the variable name
        // Either way, we can use it as-is for the bool variable
        init_stmts.extend(quote! {
            let mut #iter_name = 0i64;
            let mut #cond_bool_name = #initial_cond;
        });

        // cond_in_name holds the tensor/scalar version for passing to the loop body
        let cond_var_init = match &self.condition {
            Type::Scalar(scalar) => {
                let name = &scalar.name;
                quote! {
                    let mut #cond_in_name = #name;
                }
            }
            Type::Tensor(tensor) => {
                let name = &tensor.name;
                quote! {
                    let mut #cond_in_name = #name;
                }
            }
            _ => panic!("Unsupported condition type in Loop node"),
        };

        init_stmts.extend(cond_var_init);

        // For read-only variables, shadow them with a clone at the start of each iteration
        // This creates a new binding that shadows the old one for the loop body
        let mut pre_body_stmts = quote! {};
        for (idx, var_name) in loop_var_names.iter().enumerate().skip(num_output_vars) {
            // Check if this is a tensor type that needs cloning
            if let Type::Tensor(_) = &self.v_initial[idx] {
                pre_body_stmts.extend(quote! {
                    let #var_name = #var_name.clone();
                });
            }
        }

        // Generate loop body code
        // Constants in the loop body are automatically registered as model fields
        // by BurnGraph::collect_all_fields() which recursively processes subgraphs
        let body_code = generate_loop_body_code::<PS>(&self.body, scope, node_position);

        // Extract condition output and loop-carried dependency outputs from body
        let cond_out_name =
            syn::Ident::new(&self.body.outputs[0].name, proc_macro2::Span::call_site());

        // Update loop variables from body outputs
        // Only the first num_output_vars loop variables have corresponding body outputs
        let mut update_stmts = quote! {};

        // Update condition (cond_out_name is always from the loop body output, which matches the body input type)
        update_stmts.extend(quote! {
            #cond_in_name = #cond_out_name;
        });

        // Update the bool variable from the tensor/scalar
        let cond_bool_update = match &self.condition {
            Type::Scalar(_) => {
                quote! {
                    #cond_bool_name = #cond_out_name;
                }
            }
            Type::Tensor(_) => {
                quote! {
                    #cond_bool_name = #cond_out_name.clone().into_scalar().elem::<bool>();
                }
            }
            _ => panic!("Unsupported condition type in Loop node"),
        };
        update_stmts.extend(cond_bool_update);

        for (idx, var_name) in loop_var_names.iter().enumerate().take(num_output_vars) {
            let out_name = syn::Ident::new(
                &self.body.outputs[idx + 1].name,
                proc_macro2::Span::call_site(),
            );
            update_stmts.extend(quote! {
                #var_name = #out_name;
            });
        }

        update_stmts.extend(quote! {
            #iter_name += 1;
        });

        // Generate output assignments with block scoping
        // All loop variables are scoped inside the block to avoid polluting the outer scope
        // For scalar conditions, suppress warnings about unused/unread cond variable
        let allow_attr = match &self.condition {
            Type::Scalar(_) => quote! { #[allow(unused_variables, unused_assignments)] },
            _ => quote! {},
        };

        if self.outputs.len() == 1 {
            // Single output: let output_name = { ... loop code ... var_name };
            let output = &self.outputs[0];
            let output_name = match output {
                Type::Tensor(t) => &t.name,
                Type::Scalar(s) => &s.name,
                _ => panic!("Unsupported output type in Loop node"),
            };
            let var_name = &loop_var_names[0];

            quote! {
                #allow_attr
                let #output_name = {
                    #init_stmts

                    while #iter_name < #max_trip_count && #cond_bool_name {
                        #pre_body_stmts
                        #body_code
                        #update_stmts
                    }

                    #var_name
                };
            }
        } else {
            // Multiple outputs: let (out1, out2, ...) = { ... loop code ... (var1, var2, ...) };
            let output_names: Vec<_> = self
                .outputs
                .iter()
                .map(|output| match output {
                    Type::Tensor(t) => &t.name,
                    Type::Scalar(s) => &s.name,
                    _ => panic!("Unsupported output type in Loop node"),
                })
                .collect();

            let var_names: Vec<_> = loop_var_names.iter().take(self.outputs.len()).collect();

            quote! {
                #allow_attr
                let (#(#output_names),*) = {
                    #init_stmts

                    while #iter_name < #max_trip_count && #cond_bool_name {
                        #pre_body_stmts
                        #body_code
                        #update_stmts
                    }

                    (#(#var_names),*)
                };
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Loop(self)
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        // Register imports from body nodes
        for onnx_node in &self.body.nodes {
            if let Some(burn_node) = try_convert_onnx_node::<PS>(onnx_node.clone()) {
                burn_node.register_imports(imports);
            }
        }
    }
}

impl OnnxIntoNode for LoopNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        // Extract M (max trip count) and cond (condition) - first two inputs
        let max_trip_count = Type::from(node.inputs.first().unwrap());
        let condition = Type::from(&node.inputs[1]);

        // Get body graph from config
        let config = node.config::<onnx_ir::node::loop_node::LoopConfig>();
        let body = config.body.clone();

        // Loop-carried dependencies are inputs after M and cond
        let v_initial: Vec<Type> = node.inputs.iter().skip(2).map(Type::from).collect();

        // Outputs are the final values of loop-carried dependencies
        // (and potentially scan outputs, but we'll handle those later)
        let outputs: Vec<Type> = node.outputs.iter().map(Type::from).collect();

        Self::new(max_trip_count, condition, v_initial, outputs, body)
    }
}
