mod statement;
use std::collections::HashMap;

use proc_macro::TokenStream;
use statement::parse_statement;

/// Derive macro for the module.
#[proc_macro_attribute]
pub fn cube(_attr: TokenStream, tokens: TokenStream) -> TokenStream {
    let func: syn::ItemFn = syn::parse(tokens).unwrap();
    let mut variables = VariableAnalyses::create(&func);

    codegen_cube(&func, &mut variables)
}

#[derive(Hash, PartialEq, Eq, Debug, Clone)]
struct VariableKey {
    name: String,
}

impl From<&syn::Ident> for VariableKey {
    fn from(value: &syn::Ident) -> Self {
        VariableKey {
            name: value.to_string(),
        }
    }
}

struct VariableAnalysis {
    num_used: usize,
    loop_level_declared: usize,
}

impl VariableAnalysis {
    pub fn should_clone(&self, loop_level: usize) -> bool {
        if self.num_used == 1 && self.loop_level_declared >= loop_level {
            return false;
        }

        true
    }
}

struct VariableAnalyses {
    analyses: HashMap<VariableKey, VariableAnalysis>,
}

impl VariableAnalyses {
    pub fn should_clone(&self, ident: &syn::Ident, loop_level: usize) -> bool {
        let key: VariableKey = ident.into();
        if let Some(var) = self.analyses.get(&key) {
            return var.should_clone(loop_level);
        }

        false
    }
    pub fn create(func: &syn::ItemFn) -> Self {
        Self {
            analyses: Default::default(),
        }
    }
}

fn codegen_cube(func: &syn::ItemFn, variables: &mut VariableAnalyses) -> TokenStream {
    let signature = expand_sig(&func.sig);
    let mut body = quote::quote! {};

    for statement in func.block.stmts.iter() {
        let tokens = parse_statement(statement, 0);
        body.extend(tokens);
    }

    let code = quote::quote! {
        #func

        #[allow(unused_mut)]
        #signature {
            #body
        }
    }
    .into();

    code
}

fn expand_sig(sig: &syn::Signature) -> proc_macro2::TokenStream {
    let mut inputs = quote::quote!();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat) => {
                let ty = &pat.ty;
                let ident = pat.pat.clone();

                inputs.extend(quote::quote! {
                    #ident: <#ty as burn_cube::RuntimeType>::ExpandType,
                });
            }
            _ => todo!(),
        }
    }

    let mut output = quote::quote!();

    match &sig.output {
        syn::ReturnType::Default => output.extend(quote::quote! { ()}),
        syn::ReturnType::Type(_, ty) => {
            output.extend(quote::quote! {
                <#ty as burn_cube::RuntimeType>::ExpandType
            });
        }
    }

    let ident = &sig.ident;
    let ident = syn::Ident::new(format!("{ident}_expand").as_str(), ident.span());

    quote::quote! {
        pub fn #ident(context: &mut burn_cube::CubeContext, #inputs) -> #output
    }
    .into()
}
