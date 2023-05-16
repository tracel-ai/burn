use derive_new::new;
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use std::collections::HashMap;

#[derive(Clone, Debug, new)]
pub struct TensorVariable {
    references: usize,
    node_position: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Scope {
    variables: HashMap<Ident, Vec<TensorVariable>>,
}

impl Scope {
    /// Declare a new variable.
    pub fn declare_tensor(&mut self, name: &Ident, node_position: usize) {
        if let Some(variables) = self.variables.get_mut(name) {
            for variable in variables.iter_mut() {
                if variable.node_position == node_position {
                    variable.references += 1;
                    return;
                }
            }

            variables.push(TensorVariable::new(0, node_position));
        } else {
            self.variables
                .insert(name.clone(), vec![TensorVariable::new(0, node_position)]);
        }
        println!("registered tensor {name}");
    }

    /// We need to know all variables that are going to be used by the program.
    pub fn register_use_owned_tensor(&mut self, name: &Ident, node_position: usize) {
        if let Some(variables) = self.variables.get_mut(name) {
            let mut current_position = 0;

            for variable in variables.iter_mut() {
                if node_position > current_position && node_position < variable.node_position {
                    variable.references += 1;
                    break;
                }

                current_position = variable.node_position;
            }
        } else {
            panic!("No variable with name {name}");
        }
    }

    /// Use a variable, cloning if it was registered multiple times and other function call will be
    /// done.
    pub fn use_owned_tensor(&mut self, name: &Ident, node_position: usize) -> TokenStream {
        if let Some(variables) = self.variables.get_mut(name) {
            let mut current_position = 0;
            let mut count = 0;

            for variable in variables.iter_mut() {
                if node_position > current_position && node_position < variable.node_position {
                    variable.references -= 1;
                    count = variable.references;
                    break;
                }

                current_position = variable.node_position;
            }

            return if count > 0 {
                quote! {
                    #name.clone()
                }
            } else {
                quote! {
                    #name
                }
            };
        } else {
            panic!("No variable with name {name}");
        }
    }
}
