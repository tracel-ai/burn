use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use syn::{GenericParam, Generics, Ident, Lifetime, LifetimeParam, TypeParam};

pub(crate) struct GenericsCodegen {
    runtime_gens: syn::Generics,
    type_gens: syn::Generics,
}

impl GenericsCodegen {
    pub(crate) fn new(type_gens: syn::Generics) -> Self {
        Self {
            runtime_gens: Self::runtime_gens(),
            type_gens,
        }
    }

    fn runtime_gens() -> Generics {
        let mut runtime_gens = Generics::default();

        let lifetime =
            GenericParam::Lifetime(LifetimeParam::new(Lifetime::new("'a", Span::call_site())));
        runtime_gens.params.push(lifetime);

        let mut runtime_param = TypeParam::from(Ident::new("R", Span::call_site()));
        runtime_param
            .bounds
            .push(syn::parse_str("Runtime").unwrap());
        let runtime = GenericParam::Type(runtime_param);
        runtime_gens.params.push(runtime);

        runtime_gens
    }

    pub(crate) fn type_definitions(&self) -> TokenStream {
        self.type_gens.to_token_stream()
    }

    pub(crate) fn type_in_use(&self) -> TokenStream {
        generics_in_use_codegen(self.type_gens.clone())
    }

    pub(crate) fn runtime_definitions(&self) -> TokenStream {
        self.runtime_gens.to_token_stream()
    }

    pub(crate) fn all_definitions(&self) -> TokenStream {
        let mut generics = self.runtime_gens.clone();
        generics.params.extend(self.type_gens.params.clone());
        generics.to_token_stream()
    }

    pub(crate) fn all_in_use(&self) -> TokenStream {
        let mut generics = self.runtime_gens.clone();
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
