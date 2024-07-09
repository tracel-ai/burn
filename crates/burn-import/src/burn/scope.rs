use super::TensorType;
use derive_new::new;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use serde_json::map::Iter;
use std::collections::{HashMap, HashSet};

/// The scope struct ensures that ownership rules are respected during the forward pass.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    variables: HashMap<Ident, TensorVariable>,
    constant_set: HashSet<Ident>,
}

#[derive(Clone, Debug, new)]
pub struct TensorVariable {
    references: usize,
    node_position: usize,
}

impl Scope {
    /// Declare a new tensor variable.
    pub fn tensor_register_variable(&mut self, tensor: &TensorType, node_position: usize) {
        if let Some(variable) = self.variables.get_mut(&tensor.name) {
            if variable.node_position == node_position {
                variable.references += 1;
                return;
            }
        } else {
            self.variables
                .insert(tensor.name.clone(), TensorVariable::new(0, node_position));
        }
    }

    pub fn constants(&self) -> impl Iterator<Item = (Ident, &'_ TensorVariable)> {
        self.constant_set
            .iter()
            .map(|ident| (ident.clone(), self.variables.get(ident).unwrap()))
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
            // The tensor originated from an initializer or lifted constant
            self.constant_set.insert(tensor.name.clone());
            // self.tensor_register_variable(&tensor, 0)
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
