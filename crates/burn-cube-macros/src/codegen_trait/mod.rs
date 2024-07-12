use proc_macro2::TokenStream;

use crate::codegen_common::signature::{expand_sig, ExpandMode};

pub fn expand_trait_def(mut tr: syn::ItemTrait) -> proc_macro2::TokenStream {
    let mut expand_items = Vec::new();

    for item in tr.items.iter() {
        match item {
            syn::TraitItem::Fn(func) => {
                let expand = expand_sig(
                    &func.sig,
                    &syn::Visibility::Inherited,
                    None,
                    ExpandMode::MethodImpl,
                );
                expand_items.push(syn::parse_quote!(#expand;));
            }
            _ => continue,
        }
    }
    tr.items.append(&mut expand_items);

    quote::quote! {
        #tr
    }
}

pub fn expand_trait_impl(mut tr: syn::ItemImpl) -> proc_macro2::TokenStream {
    let mut expand_items = Vec::new();

    for item in tr.items.iter() {
        match item {
            syn::ImplItem::Fn(func) => {
                let ident = &func.sig.ident;
                let ident = quote::quote! {#ident::__expand};
                let mut inputs = quote::quote!();

                for input in &func.sig.inputs {
                    match input {
                        syn::FnArg::Typed(pat) => {
                            let ident = pat.pat.clone();
                            inputs.extend(quote::quote! {
                                #ident,
                            });
                        }
                        _ => todo!("Only Typed inputs are supported"),
                    }
                }

                let expand = expand_sig(
                    &func.sig,
                    &syn::Visibility::Inherited,
                    None,
                    ExpandMode::MethodImpl,
                );

                let tokens = if !tr.generics.params.is_empty() {
                    let mut func = func.clone();
                    for param in tr.generics.params.iter() {
                        func.sig.generics.params.push(param.clone());
                    }
                    register_expand(&func, &ident, expand, inputs)
                } else {
                    register_expand(func, &ident, expand, inputs)
                };

                expand_items.push(syn::parse2(tokens).unwrap());
            }
            _ => continue,
        }
    }
    tr.items.append(&mut expand_items);

    quote::quote! {
        #tr
    }
}

fn register_expand(
    func: &syn::ImplItemFn,
    name: &TokenStream,
    expand: proc_macro2::TokenStream,
    inputs: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let (func, func_expand) = if func.sig.generics.params.is_empty() {
        (
            quote::quote! { #func },
            quote::quote! {
                #name(context, #inputs)
            },
        )
    } else {
        let (_, gen, _) = &func.sig.generics.split_for_impl();
        (
            quote::quote! { #func },
            quote::quote! {
                #name::#gen(context, #inputs)
            },
        )
    };

    quote::quote! (
        #expand {
            #[cube]
            #func
            #func_expand
        }
    )
}
