use quote::ToTokens;

use crate::tracker::VariableTracker;

#[derive(Copy, Clone, Debug)]
pub enum ExpandMode {
    FuncImpl,
    StructImpl,
    TraitImpl,
}

pub fn expand_sig(
    sig: &syn::Signature,
    visibility: &syn::Visibility,
    mut variable_tracker: Option<&mut VariableTracker>,
    mode: ExpandMode,
) -> proc_macro2::TokenStream {
    let mut inputs = quote::quote!();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ident = pat.pat.clone();

                if let syn::Pat::Ident(ident) = ident.as_ref() {
                    if let Some(vars) = &mut variable_tracker {
                        vars.codegen_declare(ident.ident.to_string(), 0);
                    }
                }

                let ty = no_ref(pat.ty.as_ref());
                inputs.extend(quote::quote! {
                    #ident: <#ty as burn_cube::frontend::CubeType>::ExpandType,
                });
            }
            _ => todo!("Only Typed inputs are supported"),
        }
    }

    let mut output = quote::quote!();

    match &sig.output {
        syn::ReturnType::Default => output.extend(quote::quote! { ()}),
        syn::ReturnType::Type(_, ty) => {
            let ty = no_ref(ty.as_ref());
            output.extend(quote::quote! {
                <#ty as burn_cube::frontend::CubeType>::ExpandType
            });
        }
    }

    let ident = &sig.ident;
    let ident = match mode {
        ExpandMode::FuncImpl => syn::Ident::new(format!("__expand").as_str(), ident.span()),
        _ => syn::Ident::new(format!("__expand_{ident}").as_str(), ident.span()),
    };

    let generics = sig.generics.clone().into_token_stream();

    quote::quote! {
        /// Expanded Cube function
        #visibility fn #ident #generics (context: &mut burn_cube::frontend::CubeContext, #inputs) -> #output
    }
}

pub fn no_ref(ty: &syn::Type) -> &syn::Type {
    match ty {
        syn::Type::Reference(val) => &val.elem,
        _ => ty,
    }
}
