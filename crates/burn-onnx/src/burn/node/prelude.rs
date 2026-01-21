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
pub(crate) use super::{NodeCodegen, arg_to_ident};

// Re-export common burn-onnx types
pub(crate) use crate::burn::scope::ScopeAtPosition;
pub(crate) use crate::burn::{BurnImports, Field, Scope, ToTokens};

// Re-export common onnx_ir types
pub(crate) use onnx_ir::{
    Argument,
    ir::{ArgType, DType},
};

// Re-export common proc_macro2 types
pub(crate) use proc_macro2::{Ident, Span, TokenStream};

// Re-export quote macro
pub(crate) use quote::quote;
