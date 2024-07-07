use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use syn::{GenericParam, Generics, Ident, Lifetime, LifetimeParam, TypeParam};

pub(crate) struct GenericsCodegen {
    arg_lifetime: syn::Generics,
    arg_runtime: syn::Generics,
    type_gens: syn::Generics,
}

impl GenericsCodegen {
    pub(crate) fn new(type_gens: syn::Generics) -> Self {
        Self {
            arg_lifetime: Self::arg_lifetime(),
            arg_runtime: Self::arg_runtime(),
            type_gens,
        }
    }

    fn arg_lifetime() -> Generics {
        let mut generics = Generics::default();
        let lifetime =
            GenericParam::Lifetime(LifetimeParam::new(Lifetime::new("'a", Span::call_site())));
        generics.params.push(lifetime);
        generics
    }

    fn arg_runtime() -> Generics {
        let mut generics = Generics::default();
        let mut runtime_param = TypeParam::from(Ident::new("R", Span::call_site()));
        runtime_param
            .bounds
            .push(syn::parse_str("Runtime").unwrap());
        let runtime = GenericParam::Type(runtime_param);
        generics.params.push(runtime);
        generics
    }

    pub(crate) fn type_definitions(&self) -> TokenStream {
        self.type_gens.to_token_stream()
    }

    pub(crate) fn type_in_use(&self) -> TokenStream {
        generics_in_use_codegen(self.type_gens.clone())
    }

    pub(crate) fn runtime_definitions(&self) -> TokenStream {
        let mut generics = self.arg_runtime.clone();
        generics.params.extend(self.arg_lifetime.params.clone());
        generics.to_token_stream()
    }

    pub(crate) fn all_definitions(&self) -> TokenStream {
        let mut generics = self.arg_lifetime.clone();
        generics.params.extend(self.arg_runtime.params.clone());
        generics.params.extend(self.type_gens.params.clone());
        generics.to_token_stream()
    }

    pub(crate) fn all_in_use(&self) -> TokenStream {
        let mut generics = self.arg_lifetime.clone();
        generics.params.extend(self.arg_runtime.params.clone());
        generics.params.extend(self.type_gens.params.clone());
        generics_in_use_codegen(generics)
    }
}

fn generics_in_use_codegen(generics: Generics) -> TokenStream {
    let mut tokens = quote::quote! {<};
    for generic in generics.params.iter() {
        let ident = match generic {
            GenericParam::Lifetime(param) => param.lifetime.to_token_stream(),
            GenericParam::Type(param) => param.ident.to_token_stream(),
            GenericParam::Const(_) => todo!("Const generic not supported"),
        };
        tokens.extend(quote::quote! { #ident, })
    }
    tokens.extend(quote::quote! {>});

    tokens
}
