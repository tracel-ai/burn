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

    quote! {
        #(#fn_attrs)*
        #fn_vis fn #fn_name #fn_generics() { #fn_block }

        #[test]
        #fn_vis fn #wrapper_name #fn_generics() {
            use std::panic::{self, AssertUnwindSafe};
            use std::sync::{Arc, Mutex, OnceLock};

            let get_msg = |p: &(dyn std::any::Any + Send)| -> String {
                p.downcast_ref::<String>().cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "Unknown panic".to_string())
            };

            // An append-only list of all panic messages across the entire process.
            // This is required because cubecl's `CallError` hides the original panic message
            // occurring in the device threads.
            //
            // A global log also prevents parallel tests from overwriting each other's panic hooks.
            static PANIC_LOG: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
            let log = PANIC_LOG.get_or_init(|| Mutex::new(Vec::new()));

            static HOOK: OnceLock<()> = OnceLock::new();
            HOOK.get_or_init(|| {
                let prev = panic::take_hook();
                panic::set_hook(Box::new(move |info| {
                    if let Ok(mut v) = log.lock() {
                        v.push(get_msg(info.payload()));
                    }
                    prev(info);
                }));
            });

            // We only care about panics that occur during this test's execution window, so
            // we start at the number of panics logged before this test starts.
            let start_idx = log.lock().unwrap().len();
            let result = panic::catch_unwind(AssertUnwindSafe(|| #fn_name()));

            if let Err(e) = result {
                let main_msg = get_msg(&*e);
                let panic_logs = log.lock().unwrap();
                let window = &panic_logs[start_idx..];

                let matched = window.iter().chain(std::iter::once(&main_msg))
                    .any(|m| m.contains(#expected_reason));

                let all = window.iter().chain(std::iter::once(&main_msg))
                    .map(|m| format!("- {m}")).collect::<Vec<_>>().join("\n");

                if matched {
                    eprintln!(
                        "\n[SKIPPED - might_panic] Test '{}'\nReason: '{}'\nPanics:\n{}\n",
                        stringify!(#fn_name),
                        #expected_reason,
                        all
                    );
                    return;
                } else {
                    panic!(
                        "\nTest '{}' failed.\nExpected: '{}'\nFound:\n{}\n",
                        stringify!(#fn_name),
                        #expected_reason,
                        all
                    );
                }
            }
        }
    }
    .into()
}
