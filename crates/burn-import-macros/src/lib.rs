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
///
/// Note: Each invocation generates a uniquely-named dispatch module.
#[proc_macro]
pub fn onnx_node_registry(input: TokenStream) -> TokenStream {
    let mapping = parse_macro_input!(input as OnnxNodeMapping);

    // Use the first node type to generate a unique module name
    let first_node = mapping.entries.first().expect("Registry must have at least one entry");
    let module_suffix = first_node.onnx_type.to_string().to_lowercase();
    let module_name = syn::Ident::new(&format!("onnx_dispatch_{}", module_suffix), first_node.onnx_type.span());
    let try_fn_name = syn::Ident::new(&format!("try_convert_onnx_node_{}", module_suffix), first_node.onnx_type.span());

    let mut match_arms = Vec::new();
    let mut imports = Vec::new();
    let mut seen_modules = std::collections::HashSet::new();

    for entry in &mapping.entries {
        let onnx_type = &entry.onnx_type;
        let module = &entry.module;
        let module_str = module.to_string();

        // Convert module name to PascalCase node type name
        let node_type_name = to_pascal_case(&module_str);
        let node_type_ident = syn::Ident::new(&node_type_name, module.span());

        // Only add import once per unique module
        if !seen_modules.contains(&module_str) {
            imports.push(quote! {
                use super::#module::#node_type_ident;
            });
            seen_modules.insert(module_str);
        }

        match_arms.push(quote! {
            NodeType::#onnx_type => {
                Some(NodeCodegen::into_node(#node_type_ident::from_onnx(node)))
            }
        });
    }

    let expanded = quote! {
        /// Auto-generated ONNX node dispatcher
        #[allow(unused_imports)]
        pub(crate) mod #module_name {
            use onnx_ir::NodeType;
            use crate::burn::node::Node;
            use crate::burn::node::OnnxIntoNode;
            use burn::record::PrecisionSettings;
            use crate::burn::node::NodeCodegen;

            #(#imports)*

            /// Try to convert an ONNX node using this registry
            pub fn #try_fn_name<PS: PrecisionSettings>(
                node: onnx_ir::Node,
            ) -> Option<Node<PS>> {
                match node.node_type {
                    #(#match_arms)*
                    _ => None,
                }
            }
        }
    };

    TokenStream::from(expanded)
}

fn to_pascal_case(s: &str) -> String {
    // Special case mappings for names that don't follow simple rules
    let special_cases = [
        ("argmax", "ArgMax"),
        ("argmin", "ArgMin"),
        ("matmul", "Matmul"),
        ("matmul_integer", "MatMulInteger"),
        ("nonzero", "NonZero"),
        ("where_op", "Where"),
        ("bitwisenot", "BitwiseNot"),
        ("bitwiseand", "BitwiseAnd"),
        ("bitwiseor", "BitwiseOr"),
        ("bitwisexor", "BitwiseXor"),
        ("bitshift", "BitShift"),
        ("modulo", "Mod"),
        ("pow", "Pow"),
        ("prelu", "PRelu"),
    ];

    for (pattern, replacement) in &special_cases {
        if s == *pattern {
            return format!("{}Node", replacement);
        }
    }

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
