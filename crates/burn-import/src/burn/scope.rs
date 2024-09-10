use super::TensorType;
use derive_new::new;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use std::collections::HashMap;

/// The scope struct ensures that ownership rules are respected during the forward pass.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    variables: HashMap<Ident, TensorVariable>,
}

#[derive(Clone, Debug, new)]
struct TensorVariable {
    references: usize,
    node_position: usize,
}

impl Scope {
    /// Declare a new tensor variable.
    pub fn tensor_register_variable(&mut self, tensor: &TensorType, node_position: usize) {
        if let Some(variable) = self.variables.get_mut(&tensor.name) {
            if variable.node_position == node_position {
                variable.references += 1;
            }
        } else {
            self.variables
                .insert(tensor.name.clone(), TensorVariable::new(0, node_position));
        }
    }

    /// Register a future use of a tensor variable.
    ///
    /// # Notes
    ///
    /// We need to know all futures use of a variable in advance.
    pub fn tensor_register_future_use(&mut self, tensor: &TensorType, node_position: usize) {
        if let Some(variable) = self.variables.get_mut(&tensor.name) {
            if node_position >= variable.node_position {
                variable.references += 1;
            }
        } else {
            panic!("No variable with name {}", tensor.name);
        }
    }

    /// Use a tensor variable, cloning it if it was registered multiple times and the tensor will still be used afterward.
    pub fn tensor_use_owned(&mut self, tensor: &TensorType, node_position: usize) -> TokenStream {
        if let Some(variable) = self.variables.get_mut(&tensor.name) {
            let mut count = 0;
            let name = &tensor.name;

            if node_position >= variable.node_position {
                variable.references -= 1;
                count = variable.references;
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
            panic!("No variable with name {}", &tensor.name);
        }
    }
}
