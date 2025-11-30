use derive_new::new;
use onnx_ir::{Argument, ir::ArgType};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use std::collections::HashMap;

use super::node_traits::arg_to_ident;

/// The scope struct ensures that ownership rules are respected during the forward pass.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    variables: HashMap<String, TensorVariable>,
}

#[derive(Clone, Debug, new)]
struct TensorVariable {
    references: usize,
    node_position: usize,
}

impl Scope {
    /// Declare a new tensor variable.
    pub fn tensor_register_variable(&mut self, arg: &Argument, node_position: usize) {
        if let Some(variable) = self.variables.get_mut(&arg.name) {
            if variable.node_position == node_position {
                variable.references += 1;
            }
        } else {
            self.variables
                .insert(arg.name.clone(), TensorVariable::new(0, node_position));
        }
    }

    /// Register a future use of a tensor variable.
    ///
    /// # Notes
    ///
    /// We need to know all futures use of a variable in advance.
    /// If the variable doesn't exist yet (e.g., it's created in a nested subgraph),
    /// we register it with the given node_position.
    pub fn tensor_register_future_use(&mut self, arg: &Argument, node_position: usize) {
        if let Some(variable) = self.variables.get_mut(&arg.name) {
            if node_position >= variable.node_position {
                variable.references += 1;
            }
        } else {
            // Variable doesn't exist yet - register it (e.g., from nested subgraph)
            self.variables
                .insert(arg.name.clone(), TensorVariable::new(1, node_position));
        }
    }

    /// Use a tensor variable, cloning it if it was registered multiple times and the tensor will still be used afterward.
    pub fn tensor_use_owned(&mut self, arg: &Argument, node_position: usize) -> TokenStream {
        let name = Ident::new(&arg.name, Span::call_site());

        if let Some(variable) = self.variables.get_mut(&arg.name) {
            let mut count = 0;

            if node_position >= variable.node_position {
                // Only decrement if references > 0 to avoid underflow
                if variable.references > 0 {
                    variable.references -= 1;
                    count = variable.references;
                }
            }

            if count > 0 {
                quote! {
                    #name.clone()
                }
            } else {
                quote! {
                    #name
                }
            }
        } else {
            // Variable doesn't exist in scope (e.g., from nested subgraph) - just use it
            quote! {
                #name
            }
        }
    }

    /// Create a ScopeAtPosition wrapper for ergonomic position-aware operations
    ///
    /// # Arguments
    ///
    /// * `node_position` - The position of the current node in the graph
    ///
    /// # Returns
    ///
    /// A ScopeAtPosition wrapper that encapsulates the position
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
    ///     let mut scope = scope.at_position(node_position);
    ///     let input = scope.arg(&self.inputs[0]);
    ///     // ...
    /// }
    /// ```
    pub fn at_position(&mut self, node_position: usize) -> ScopeAtPosition<'_> {
        ScopeAtPosition {
            scope: self,
            node_position,
        }
    }
}

/// A wrapper around Scope that encapsulates the node_position parameter
///
/// This provides a more ergonomic API for the common pattern of using scope
/// within a node's forward method at a specific position.
///
/// # Benefits
///
/// - Eliminates need to pass node_position to every scope method call
/// - Forces correct usage through the `arg()` method which automatically handles type checking
/// - Reduces boilerplate in node forward implementations
pub struct ScopeAtPosition<'a> {
    scope: &'a mut Scope,
    node_position: usize,
}

impl<'a> ScopeAtPosition<'a> {
    /// Get argument as tokens, automatically handling Tensor/Scalar/Shape types
    ///
    /// This method provides automatic type-conditional handling:
    /// - For Tensor arguments: Uses `tensor_use_owned()` with automatic clone tracking
    /// - For Scalar/Shape arguments: Returns the identifier directly (no clone needed)
    ///
    /// # Arguments
    ///
    /// * `arg` - The argument to convert to tokens
    ///
    /// # Returns
    ///
    /// TokenStream representing the argument usage
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
    ///     let mut scope = scope.at_position(node_position);
    ///
    ///     // Automatically handles Tensor vs Scalar/Shape
    ///     let input = scope.arg(&self.inputs[0]);
    ///     let alpha = scope.arg(&self.inputs[1]); // Could be Scalar or Tensor
    ///
    ///     quote! {
    ///         let output = #input * #alpha;
    ///     }
    /// }
    /// ```
    pub fn arg(&mut self, arg: &Argument) -> TokenStream {
        match &arg.ty {
            ArgType::Tensor(_) => self.scope.tensor_use_owned(arg, self.node_position),
            ArgType::Scalar(_) | ArgType::Shape(_) => {
                let name = arg_to_ident(arg);
                quote! { #name }
            }
        }
    }

    /// Get a reference to the underlying scope (for advanced usage)
    ///
    /// This is needed for complex nodes that need to manipulate the scope directly,
    /// such as if/loop/scan nodes that need to register variables for subgraphs.
    pub fn scope(&mut self) -> &mut Scope {
        self.scope
    }

    /// Get the current node position
    pub fn node_position(&self) -> usize {
        self.node_position
    }
}
