use super::{Node, NodeCodegen, OnnxIntoNode, try_convert_onnx_node};
use crate::burn::{BurnImports, Scope, TensorType, Type};

use burn::record::PrecisionSettings;
use proc_macro2::TokenStream;
use quote::quote;

/// Generate inline code for a scan body subgraph
///
/// Converts an OnnxGraph into a TokenStream that executes for each scan element.
/// Returns the body code
fn generate_scan_body_code<PS: PrecisionSettings + 'static>(
    subgraph: &onnx_ir::OnnxGraph,
    scope: &mut Scope,
    node_position: usize,
) -> TokenStream {
    let mut body = quote! {};

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
            try_convert_onnx_node::<PS>(node.clone())
                .unwrap_or_else(|| panic!("Unsupported op in scan body: {}", node.name()))
        })
        .collect();

    // Register subgraph inputs in scope (they reference scan variables)
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

/// Scan node - iterates over input sequences while maintaining state variables
///
/// The Scan operation processes input sequences element by element, maintaining
/// state variables across iterations and optionally producing output sequences.
#[derive(Debug, Clone)]
pub struct ScanNode {
    pub initial_state_vars: Vec<Type>, // Initial values of state variables
    pub scan_input_sequences: Vec<Type>, // Input sequences to scan over
    pub outputs: Vec<Type>,            // Final state vars + scan output sequences
    pub body: onnx_ir::OnnxGraph,
    pub num_scan_inputs: usize,
    pub scan_input_directions: Vec<i64>,
    pub scan_output_directions: Vec<i64>,
    pub scan_input_axes: Vec<i64>,
    pub scan_output_axes: Vec<i64>,
}

impl ScanNode {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_state_vars: Vec<Type>,
        scan_input_sequences: Vec<Type>,
        outputs: Vec<Type>,
        body: onnx_ir::OnnxGraph,
        num_scan_inputs: usize,
        scan_input_directions: Vec<i64>,
        scan_output_directions: Vec<i64>,
        scan_input_axes: Vec<i64>,
        scan_output_axes: Vec<i64>,
    ) -> Self {
        Self {
            initial_state_vars,
            scan_input_sequences,
            outputs,
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
            scan_input_axes,
            scan_output_axes,
        }
    }
}

impl<PS: PrecisionSettings + 'static> NodeCodegen<PS> for ScanNode {
    fn output_types(&self) -> Vec<Type> {
        self.outputs.clone()
    }

    fn input_types(&self) -> Vec<Type> {
        let mut inputs = self.initial_state_vars.clone();
        inputs.extend(self.scan_input_sequences.clone());
        inputs
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        let num_state_vars = self.initial_state_vars.len();
        let num_scan_inputs = self.scan_input_sequences.len();

        // Body inputs: [state_vars..., scan_inputs...]
        // Body outputs: [state_vars_out..., scan_outputs...]
        assert_eq!(
            self.body.inputs.len(),
            num_state_vars + num_scan_inputs,
            "Scan body should have {} inputs, got {}",
            num_state_vars + num_scan_inputs,
            self.body.inputs.len()
        );

        let num_scan_outputs = self.body.outputs.len() - num_state_vars;

        // Node outputs: [final_state_vars..., scan_output_sequences...]
        assert_eq!(
            self.outputs.len(),
            num_state_vars + num_scan_outputs,
            "Scan should have {} outputs, got {}",
            num_state_vars + num_scan_outputs,
            self.outputs.len()
        );

        // Get sequence length from first scan input
        let first_scan_input = &self.scan_input_sequences[0];
        let seq_len_expr = match first_scan_input {
            Type::Tensor(tensor) => {
                let tensor_name = &tensor.name;
                let axis = if !self.scan_input_axes.is_empty() {
                    self.scan_input_axes[0] as usize
                } else {
                    0
                };
                quote! { #tensor_name.shape().dims[#axis] }
            }
            _ => panic!("Scan input must be a tensor"),
        };

        // Initialize state variables
        let mut init_stmts = quote! {};
        let state_var_names: Vec<_> = self
            .body
            .inputs
            .iter()
            .take(num_state_vars)
            .map(|arg| syn::Ident::new(&arg.name, proc_macro2::Span::call_site()))
            .collect();

        for (idx, initial_value) in self.initial_state_vars.iter().enumerate() {
            let var_name = &state_var_names[idx];
            let init_value = match initial_value {
                Type::Tensor(tensor) => {
                    let tensor_name = &tensor.name;
                    quote! { #tensor_name }
                }
                _ => panic!("Scan state variables must be tensors"),
            };

            init_stmts.extend(quote! {
                let mut #var_name = #init_value;
            });
        }

        // Initialize scan output accumulators (if any)
        let scan_output_names: Vec<_> = self
            .body
            .outputs
            .iter()
            .skip(num_state_vars)
            .map(|arg| {
                let base_name = &arg.name;
                syn::Ident::new(
                    &format!("{}_seq", base_name),
                    proc_macro2::Span::call_site(),
                )
            })
            .collect();

        for output_name in &scan_output_names {
            init_stmts.extend(quote! {
                let mut #output_name = alloc::vec::Vec::new();
            });
        }

        // Generate loop iteration variable
        init_stmts.extend(quote! {
            let seq_len = #seq_len_expr;
        });

        // Extract scan inputs for current iteration
        let scan_input_names: Vec<_> = self
            .body
            .inputs
            .iter()
            .skip(num_state_vars)
            .map(|arg| syn::Ident::new(&arg.name, proc_macro2::Span::call_site()))
            .collect();

        let mut slice_stmts = quote! {};
        for (idx, scan_input) in self.scan_input_sequences.iter().enumerate() {
            let var_name = &scan_input_names[idx];
            let reverse =
                !self.scan_input_directions.is_empty() && self.scan_input_directions[idx] == 1;

            match scan_input {
                Type::Tensor(tensor) => {
                    let tensor_name = &tensor.name;
                    let scan_axis = if !self.scan_input_axes.is_empty() {
                        self.scan_input_axes[idx] as usize
                    } else {
                        0
                    };
                    let iter_expr = if reverse {
                        quote! { seq_len - i - 1 }
                    } else {
                        quote! { i }
                    };

                    // Use slice_dim to slice along the correct scan axis, then squeeze_dim to remove only that dimension
                    // Input rank D, slice along scan_axis, then squeeze → output rank D-1
                    let input_rank = tensor.rank;
                    let output_rank = input_rank - 1;
                    slice_stmts.extend(quote! {
                        let #var_name = #tensor_name
                            .clone()
                            .slice_dim(#scan_axis, #iter_expr..#iter_expr + 1)
                            .squeeze_dim::<#output_rank>(#scan_axis);
                    });
                }
                _ => panic!("Scan input must be a tensor"),
            }
        }

        // Generate body code
        let body_code = generate_scan_body_code::<PS>(&self.body, scope, node_position);

        // Update state variables and collect scan outputs
        let mut update_stmts = quote! {};

        // If there are scan outputs, we need to clone state updates
        // because the same body output might be used for both state and scan output
        let num_scan_outputs = scan_output_names.len();
        let should_clone = num_scan_outputs > 0;

        for (idx, var_name) in state_var_names.iter().enumerate() {
            let out_name =
                syn::Ident::new(&self.body.outputs[idx].name, proc_macro2::Span::call_site());
            if should_clone {
                update_stmts.extend(quote! {
                    #var_name = #out_name.clone();
                });
            } else {
                update_stmts.extend(quote! {
                    #var_name = #out_name;
                });
            }
        }

        // Collect scan outputs
        for (idx, output_name) in scan_output_names.iter().enumerate() {
            let body_out_name = syn::Ident::new(
                &self.body.outputs[num_state_vars + idx].name,
                proc_macro2::Span::call_site(),
            );
            update_stmts.extend(quote! {
                #output_name.push(#body_out_name);
            });
        }

        // Stack scan outputs into tensors
        let mut finalize_stmts = quote! {};
        for (idx, output_name) in scan_output_names.iter().enumerate() {
            let reverse =
                !self.scan_output_directions.is_empty() && self.scan_output_directions[idx] == 1;

            if reverse {
                finalize_stmts.extend(quote! {
                    #output_name.reverse();
                });
            }

            // Determine which axis to stack along (defaults to 0 if not specified)
            let stack_axis = if !self.scan_output_axes.is_empty() {
                self.scan_output_axes[idx] as usize
            } else {
                0
            };

            finalize_stmts.extend(quote! {
                let #output_name = Tensor::stack(#output_name, #stack_axis);
            });
        }

        // Generate output assignments
        let output_names: Vec<_> = self
            .outputs
            .iter()
            .enumerate()
            .map(|(idx, _output)| {
                if idx < num_state_vars {
                    // Final state variables
                    state_var_names[idx].clone()
                } else {
                    // Scan output sequences
                    scan_output_names[idx - num_state_vars].clone()
                }
            })
            .collect();

        let final_output_names: Vec<_> = self
            .outputs
            .iter()
            .map(|output| match output {
                Type::Tensor(t) => &t.name,
                _ => panic!("Scan outputs must be tensors"),
            })
            .collect();

        if self.outputs.len() == 1 {
            let output_name = &final_output_names[0];
            let var_name = &output_names[0];

            quote! {
                let #output_name = {
                    #init_stmts

                    for i in 0..seq_len {
                        #slice_stmts
                        #body_code
                        #update_stmts
                    }

                    #finalize_stmts
                    #var_name
                };
            }
        } else {
            quote! {
                let (#(#final_output_names),*) = {
                    #init_stmts

                    for i in 0..seq_len {
                        #slice_stmts
                        #body_code
                        #update_stmts
                    }

                    #finalize_stmts
                    (#(#output_names),*)
                };
            }
        }
    }

    fn into_node(self) -> Node<PS> {
        Node::Scan(self)
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

impl OnnxIntoNode for ScanNode {
    fn from_onnx(node: onnx_ir::Node) -> Self {
        // Get body graph and config from node
        let onnx_ir::Node::Scan(n) = &node else {
            panic!("Expected Scan node");
        };
        let body = n.config.body.clone();
        let num_scan_inputs = n.config.num_scan_inputs as usize;
        let scan_input_directions = n.config.scan_input_directions.clone();
        let scan_output_directions = n.config.scan_output_directions.clone();
        let scan_input_axes = n.config.scan_input_axes.clone();
        let scan_output_axes = n.config.scan_output_axes.clone();

        // Split inputs into state variables and scan inputs
        let num_state_vars = n.inputs.len() - num_scan_inputs;
        let initial_state_vars: Vec<Type> = n
            .inputs
            .iter()
            .take(num_state_vars)
            .map(Type::from)
            .collect();

        let scan_input_sequences: Vec<Type> = n
            .inputs
            .iter()
            .skip(num_state_vars)
            .map(Type::from)
            .collect();

        // Outputs are final state vars + scan output sequences
        let outputs: Vec<Type> = n.outputs.iter().map(Type::from).collect();

        Self::new(
            initial_state_vars,
            scan_input_sequences,
            outputs,
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
            scan_input_axes,
            scan_output_axes,
        )
    }
}
