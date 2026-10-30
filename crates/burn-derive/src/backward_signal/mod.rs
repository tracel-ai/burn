use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Field, Fields, Index, Member};

/// One field of the struct, named or positional, and whether it is cloned through.
struct SignalField {
    member: Member,
    skip: bool,
}

impl SignalField {
    fn parse(index: usize, field: &Field) -> syn::Result<Self> {
        let member = match &field.ident {
            Some(ident) => Member::Named(ident.clone()),
            None => Member::Unnamed(Index::from(index)),
        };
        let mut skip = false;
        for attr in &field.attrs {
            if attr.path().is_ident("backward_signal") {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") {
                        skip = true;
                        Ok(())
                    } else {
                        Err(meta.error("the only `backward_signal` attribute is `skip`"))
                    }
                })?;
            }
        }
        Ok(Self { member, skip })
    }
}

pub(crate) fn derive_impl(input: &DeriveInput) -> TokenStream {
    match generate(input) {
        Ok(tokens) => tokens,
        Err(error) => error.to_compile_error(),
    }
}

fn generate(input: &DeriveInput) -> syn::Result<TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "`BackwardSignal` can only be derived for a struct",
        ));
    };
    let fields = match &data.fields {
        Fields::Named(fields) => &fields.named,
        Fields::Unnamed(fields) => &fields.unnamed,
        Fields::Unit => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "`BackwardSignal` needs a struct with fields",
            ));
        }
    };
    let fields = fields
        .iter()
        .enumerate()
        .map(|(index, field)| SignalField::parse(index, field))
        .collect::<syn::Result<Vec<_>>>()?;

    let signal = quote! { burn::module::parallel::BackwardSignal };
    let detached = fields.iter().map(|SignalField { member, skip }| {
        if *skip {
            quote! { #member: self.#member }
        } else {
            quote! { #member: #signal::detach_require_grad(self.#member) }
        }
    });
    let gradients = fields.iter().map(|SignalField { member, skip }| {
        if *skip {
            quote! { #member: ::core::clone::Clone::clone(&self.#member) }
        } else {
            quote! { #member: #signal::grad(&self.#member, grads) }
        }
    });
    let products =
        fields
            .iter()
            .filter(|field| !field.skip)
            .map(|SignalField { member, .. }| {
                quote! { #signal::grad_product(self.#member, grad.#member) }
            });

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        #[allow(unused_variables)]
        impl #impl_generics #signal for #name #ty_generics #where_clause {
            fn detach_require_grad(self) -> Self {
                Self { #(#detached),* }
            }

            fn grad(&self, grads: &burn::tensor::Gradients) -> Self {
                Self { #(#gradients),* }
            }

            fn grad_product(self, grad: Self) -> Option<burn::tensor::Tensor<1>> {
                burn::module::parallel::sum_products([#(#products),*])
            }
        }
    })
}
