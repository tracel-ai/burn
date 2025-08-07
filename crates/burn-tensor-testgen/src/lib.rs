use proc_macro::TokenStream;
use quote::{format_ident, quote};

use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::{Attribute, Expr, ItemFn, Lit, Meta, MetaNameValue, parse_macro_input};

// Define a structure to parse the attribute arguments
struct AttributeArgs {
    args: Punctuated<Meta, Comma>,
}

impl Parse for AttributeArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(AttributeArgs {
            args: Punctuated::parse_terminated(input)?,
        })
    }
}

#[allow(clippy::test_attr_in_doctest)]
/// **This is only meaningful when the `reason` is specific and clear.**
///
/// A proc macro attribute that adds panic handling to test functions.
///
/// # Usage
/// ```rust, ignore
/// #[might_panic(reason = "expected panic message prefix")]
/// #[test]
/// fn test_that_might_panic() {
///     // test code that might panic (with acceptable reason)
/// }
/// ```
///
/// # Behavior
/// - If the test does not panic, it passes.
/// - If the test panics with a message starting with the expected prefix, the failure is ignored.
/// - If the test panics with a different message, the test fails.
///
/// # Note
/// This proc macro uses [`std::panic::catch_unwind`]. As such, it does not work in a no-std environment.
/// Make sure it is feature gated when an `"std"` feature is available.
#[proc_macro_attribute]
pub fn might_panic(args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse the attribute arguments
    let args = parse_macro_input!(args as AttributeArgs);
    let input_fn = parse_macro_input!(input as ItemFn);

    // Extract the expected panic reason
    let mut expected_reason = None;
    for arg in args.args.iter() {
        if let Meta::NameValue(MetaNameValue { path, value, .. }) = arg
            && path.is_ident("reason")
            && let Expr::Lit(lit) = value
            && let Lit::Str(ref lit_str) = lit.lit
        {
            expected_reason = Some(lit_str.value());
        }
    }

    let expected_reason = match expected_reason {
        Some(reason) => reason,
        None => {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "The #[might_panic] attribute requires a 'reason' parameter",
            )
            .to_compile_error()
            .into();
        }
    };

    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_generics = &input_fn.sig.generics;
    let fn_block = &input_fn.block;
    let fn_attrs = input_fn
        .attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("test"))
        .collect::<Vec<&Attribute>>();

    // Create a wrapped test function
    let wrapper_name = format_ident!("{}_might_panic", fn_name);

    let expanded = quote! {
        #(#fn_attrs)*
        #fn_vis fn #fn_name #fn_generics() {
            #fn_block
        }

        #[test]
        #fn_vis fn #wrapper_name #fn_generics() {
            use std::panic::{self, AssertUnwindSafe};

            let expected_reason = #expected_reason;
            let result = panic::catch_unwind(AssertUnwindSafe(|| {
                #fn_name();
            }));

            match result {
                Ok(_) => {
                    // Test passed without panic - this is OK
                }
                Err(e) => {
                    // Convert the panic payload to a string
                    let panic_msg = if let Some(s) = e.downcast_ref::<String>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "Unknown panic".to_string()
                    };

                    // Check if the panic message starts with the expected reason
                    if !panic_msg.starts_with(expected_reason) {
                        panic!(
                            "Test '{}' marked as 'might_panic' failed. Expected reason: '{}'",
                            stringify!(#fn_name),
                            expected_reason
                        );
                    }
                }
            }
        }
    };

    expanded.into()
}

#[allow(missing_docs)]
#[proc_macro_attribute]
pub fn testgen(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item: proc_macro2::TokenStream = proc_macro2::TokenStream::from(item);
    let attr: proc_macro2::TokenStream = proc_macro2::TokenStream::from(attr);
    let macro_ident = format_ident!("testgen_{}", attr.to_string());

    let macro_gen = quote! {
        #[allow(missing_docs)]
        #[macro_export]
        macro_rules! #macro_ident {
            () => {
                mod #attr {
                    use super::*;

                    #item
                }
            };
        }
    };

    macro_gen.into()
}
