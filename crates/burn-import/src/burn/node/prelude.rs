/// Common imports for ONNX node implementations
///
/// This prelude provides all the commonly used types, traits, and functions
/// needed when implementing NodeCodegen for ONNX operators.
///
/// # Usage
///
/// Add this import at the top of your node implementation file:
///
/// ```ignore
/// use super::prelude::*;
/// ```
/// Re-export common traits and helpers from parent module
pub(crate) use super::{NodeCodegen, SerializationBackend, arg_to_ident, extract_node_data};

// Re-export common burn-import types
pub(crate) use crate::burn::scope::ScopeAtPosition;
pub(crate) use crate::burn::{BurnImports, Field, Scope, ToTokens};

// Re-export common burn types
pub(crate) use burn::record::PrecisionSettings;

// Re-export common onnx_ir types
pub(crate) use onnx_ir::{Argument, ir::ArgType};

// Re-export common proc_macro2 types
pub(crate) use proc_macro2::{Ident, Span, TokenStream};

// Re-export quote macro
pub(crate) use quote::quote;
