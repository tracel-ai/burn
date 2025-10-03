use proc_macro2::TokenStream;
use quote::quote;
use serde::Serialize;

// Import the generated Node enum and match_all! macro from registry
use super::node_registry::{Node, match_all};

use burn::record::PrecisionSettings;
use onnx_ir::node::padding::{PaddingConfig1d, PaddingConfig2d, PaddingConfig3d};

use crate::burn::{BurnImports, Scope, Type};

pub type SerializationBackend = burn_ndarray::NdArray<f32>;

/// Trait for converting ONNX IR nodes to Burn nodes
#[allow(dead_code)]
pub trait OnnxIntoNode: Sized {
    /// Convert an ONNX IR node into this Burn node type
    fn from_onnx(node: onnx_ir::Node) -> Self;
}

pub trait NodeCodegen<PS: PrecisionSettings>: std::fmt::Debug {
    /// All types that are used as inputs during the forward pass.
    ///
    /// # Notes
    /// The vec should not include types that are accessible with `self`.
    /// See [field type](NodeCodegen::field_type).
    fn input_types(&self) -> Vec<Type>;

    /// All types that are produced during the forward pass.
    fn output_types(&self) -> Vec<Type>;

    /// The forward pass implementation of the node.
    ///
    /// # Notes
    ///
    /// The [Scope](Scope) struct should be used for [input tensor type](Type::Tensor) access.
    /// The method [use_owned_tensor](Scope::use_owned_tensor) keeps track of tensor reference
    /// count and insert `clone` with necessary.
    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream;

    /// Convert the node implementation into a [node entry](Node).
    fn into_node(self) -> Node<PS>;

    /// Register the necessary imports.
    fn register_imports(&self, _imports: &mut BurnImports) {}

    /// (Optional) Declare the type of the field
    ///
    /// # Notes
    ///
    /// This should be implemented when the node has some parameters.
    /// Just one field per type is possible, if the node has multiple types for its parameters, a
    /// tuple can be used.
    ///
    /// Other field functions should be implemented when this one returns something other than None.
    ///   * [field_init](NodeCodegen::field_init) to initialize parameters.
    ///   * [field_serialize](NodeCodegen::field_serialize) to create the model record.
    fn field_type(&self) -> Option<Type> {
        None
    }

    /// (Optional) Declare how the parameters are initialized.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_init(&self) -> Option<TokenStream> {
        None
    }

    /// (Optional) Declare how the parameters are serialized in a record.
    ///
    /// The function should be implemented along [field_type](NodeCodegen::field_type).
    fn field_serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        panic!("Serialization should be implemented when field_type is not None.");
    }
}

impl<PS: PrecisionSettings> Serialize for Node<PS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.field_serialize(serializer)
    }
}

impl<PS: PrecisionSettings> NodeCodegen<PS> for Node<PS> {
    fn output_types(&self) -> Vec<Type> {
        match_all!(self, NodeCodegen::<PS>::output_types)
    }

    fn input_types(&self) -> Vec<Type> {
        match_all!(self, NodeCodegen::<PS>::input_types)
    }

    fn forward(&self, scope: &mut Scope, node_position: usize) -> TokenStream {
        match_all!(self, |node| NodeCodegen::<PS>::forward(
            node,
            scope,
            node_position
        ))
    }

    fn field_type(&self) -> Option<Type> {
        match_all!(self, NodeCodegen::<PS>::field_type)
    }

    fn field_init(&self) -> Option<TokenStream> {
        match_all!(self, |node| NodeCodegen::<PS>::field_init(node,))
    }

    fn register_imports(&self, imports: &mut BurnImports) {
        match_all!(self, |node| NodeCodegen::<PS>::register_imports(
            node, imports
        ))
    }

    fn into_node(self) -> Node<PS> {
        self
    }

    fn field_serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match_all!(self, |node| NodeCodegen::<PS>::field_serialize(
            node, serializer
        ))
    }
}

// ============================================================================
// Codegen utilities for converting types to TokenStream
// ============================================================================

fn convert_primitive<T: core::fmt::Debug>(primitive: T) -> TokenStream {
    let value = format!("{primitive:?}");

    value.parse().unwrap()
}

fn convert_to_array<'a, I, T>(list: I) -> TokenStream
where
    I: Iterator<Item = &'a T>,
    T: ToTokens + 'a,
{
    let mut body = quote! {};

    list.for_each(|item| {
        let elem = item.to_tokens();
        body.extend(quote! {#elem,});
    });

    quote! {
        [#body]
    }
}

pub trait ToTokens {
    fn to_tokens(&self) -> TokenStream;
}

impl<const N: usize, T: Copy + ToTokens> ToTokens for [T; N] {
    fn to_tokens(&self) -> TokenStream {
        convert_to_array(self.iter())
    }
}

impl<T: Copy + ToTokens> ToTokens for Vec<T> {
    fn to_tokens(&self) -> TokenStream {
        convert_to_array(self.iter())
    }
}

/// Prettier output for `usize`
impl ToTokens for usize {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `i64`
impl ToTokens for i64 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `f64`
impl ToTokens for f64 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Prettier output for `f32`
impl ToTokens for f32 {
    fn to_tokens(&self) -> TokenStream {
        convert_primitive(self)
    }
}

/// Padding configuration
impl ToTokens for PaddingConfig1d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig1d::Valid },
            Self::Explicit(padding) => {
                let padding = padding.to_tokens();
                quote! { PaddingConfig1d::Explicit(#padding) }
            }
        }
    }
}

/// Padding configuration
impl ToTokens for PaddingConfig2d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig2d::Valid },
            Self::Explicit(padding1, padding2) => {
                let padding1 = padding1.to_tokens();
                let padding2 = padding2.to_tokens();
                quote! { PaddingConfig2d::Explicit(#padding1, #padding2) }
            }
        }
    }
}

/// Padding configuration
impl ToTokens for PaddingConfig3d {
    fn to_tokens(&self) -> TokenStream {
        match self {
            Self::Valid => quote! { PaddingConfig3d::Valid },
            Self::Explicit(padding1, padding2, padding3) => {
                let padding1 = padding1.to_tokens();
                let padding2 = padding2.to_tokens();
                let padding3 = padding3.to_tokens();
                quote! { PaddingConfig3d::Explicit(#padding1, #padding2, #padding3) }
            }
        }
    }
}
