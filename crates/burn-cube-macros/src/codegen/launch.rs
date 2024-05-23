use proc_macro2::TokenStream;
use quote::ToTokens;

pub fn codegen_launch(sig: &syn::Signature) -> TokenStream {
    let mut inputs = quote::quote!();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ty = &pat.ty;
                let ident = pat.pat.clone();

                inputs.extend(quote::quote! {
                    #ident: <#ty as burn_cube::CubeType>::ExpandType,
                });
            }
            _ => todo!("Only Typed inputs are supported"),
        }
    }

    let mut output = quote::quote!();

    match &sig.output {
        syn::ReturnType::Default => output.extend(quote::quote! { ()}),
        syn::ReturnType::Type(_, ty) => {
            output.extend(quote::quote! {
                <#ty as burn_cube::CubeType>::ExpandType
            });
        }
    }

    let ident = &sig.ident;
    let ident = syn::Ident::new(format!("{ident}_launch").as_str(), ident.span());

    let generics = sig.generics.clone().into_token_stream();

    quote::quote! {
        pub fn #ident #generics (context: &mut burn_cube::CubeContext, #inputs) -> #output
    }
}
