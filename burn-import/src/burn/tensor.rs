use proc_macro2::Ident;
#[cfg(test)] // For now
use proc_macro2::Span;

#[derive(Debug, Clone)]
pub struct TensorDescription {
    pub name: Ident,
    pub dim: usize,
}

#[cfg(test)] // For now
impl TensorDescription {
    pub fn new<S: AsRef<str>>(name: S, dim: usize) -> Self {
        Self {
            name: Ident::new(name.as_ref(), Span::call_site()),
            dim,
        }
    }
}
