use proc_macro2::{Ident, TokenStream};
use quote::quote;
use std::collections::HashMap;

#[derive(Debug, Clone, new)]
pub struct TensorInput {
    pub name: Ident,
    pub dim: usize,
}

#[derive(Debug, Clone, new)]
pub struct TensorOutput {
    pub name: Ident,
    pub dim: usize,
}

impl TensorInput {
    pub fn ref_count(&mut self, counts: &mut TensorRefCounts) {
        counts.increate(&self.name);
    }
    pub fn to_tokens(&self) -> TokenStream {
        let name = &self.name;

        if 1 > 1 {
            quote! {
                #name.clone()
            }
        } else {
            quote! {
                #name
            }
        }
    }
}

impl TensorOutput {
    pub fn ref_count(&mut self, counts: &mut TensorRefCounts) {
        counts.decrease(&self.name);
    }
}

pub struct TensorRefCounts {
    count: HashMap<Ident, usize>,
}

impl TensorRefCounts {
    pub fn increate(&mut self, name: &Ident) {
        if let Some(count) = self.count.get_mut(name) {
            *count += 1;
        } else {
            self.count.insert(name.clone(), 1);
        }
    }
    pub fn decrease(&mut self, name: &Ident) {
        if let Some(count) = self.count.get_mut(name) {
            *count -= 1;
        }
    }
    pub fn count(&self, name: &Ident) -> usize {
        *self.count.get(name).unwrap_or(&0)
    }
}
