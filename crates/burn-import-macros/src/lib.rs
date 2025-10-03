use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr, Token};
use syn::parse::{Parse, ParseStream};

/// Marks a node as an ONNX converter and exports metadata
///
/// # Example
/// ```ignore
/// #[onnx_node("Add")]
/// impl AddNode {
///     pub fn from_onnx(node: onnx_ir::Node) -> Self {
///         // conversion logic
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn onnx_node(attr: TokenStream, item: TokenStream) -> TokenStream {
    let onnx_type = parse_macro_input!(attr as LitStr);
    let input: proc_macro2::TokenStream = item.into();
    let onnx_type_value = onnx_type.value();

    let expanded = quote! {
        #input

        // Export metadata for this ONNX node
        inventory::submit! {
            crate::burn::node::OnnxNodeMetadata {
                onnx_type: #onnx_type_value,
                module_path: module_path!(),
            }
        }
    };

    TokenStream::from(expanded)
}

struct OnnxNodeMapping {
    entries: Vec<OnnxNodeEntry>,
}

struct OnnxNodeEntry {
    onnx_type: syn::Ident,
    #[allow(dead_code)]
    arrow: Token![=>],
    module: syn::Ident,
}

impl Parse for OnnxNodeMapping {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut entries = Vec::new();

        while !input.is_empty() {
            let onnx_type: syn::Ident = input.parse()?;
            let arrow: Token![=>] = input.parse()?;
            let module: syn::Ident = input.parse()?;

            entries.push(OnnxNodeEntry {
                onnx_type,
                arrow,
                module,
            });

            // Optional trailing comma
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(OnnxNodeMapping { entries })
    }
}

/// Generates ONNX node dispatcher from a declarative mapping
///
/// # Example
/// ```ignore
/// onnx_node_registry! {
///     Add => add,
///     Sub => sub,
///     Mul => mul,
/// }
/// ```
#[proc_macro]
pub fn onnx_node_registry(input: TokenStream) -> TokenStream {
    let mapping = parse_macro_input!(input as OnnxNodeMapping);

    let mut match_arms = Vec::new();
    let mut imports = Vec::new();

    for entry in &mapping.entries {
        let onnx_type = &entry.onnx_type;
        let module = &entry.module;

        // Convert module name to PascalCase node type name
        let node_type_name = to_pascal_case(&module.to_string());
        let node_type_ident = syn::Ident::new(&node_type_name, module.span());

        imports.push(quote! {
            use super::#module::#node_type_ident;
        });

        match_arms.push(quote! {
            NodeType::#onnx_type => {
                NodeCodegen::into_node(#node_type_ident::from_onnx(node))
            }
        });
    }

    let expanded = quote! {
        /// Auto-generated ONNX node dispatcher
        #[allow(unused_imports)]
        pub(crate) mod onnx_dispatch {
            use onnx_ir::NodeType;
            use crate::burn::node::Node;
            use crate::burn::node::OnnxIntoNode;
            use burn::record::PrecisionSettings;
            use crate::burn::node::NodeCodegen;

            #(#imports)*

            pub fn convert_onnx_node<PS: PrecisionSettings>(
                node: onnx_ir::Node,
            ) -> Node<PS> {
                match node.node_type {
                    #(#match_arms)*
                    _ => panic!("Unsupported ONNX node type: {:?}", node.node_type),
                }
            }
        }

        pub(crate) use onnx_dispatch::convert_onnx_node;
    };

    TokenStream::from(expanded)
}

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.extend(c.to_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }

    result.push_str("Node");
    result
}
